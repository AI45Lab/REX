"""Passive routing mass on the removable bank, F0 vs F1, on training corpora only.

Samples deterministic token windows from the forget and retain training
corpora once, then runs F0 and F1 over the same windows and records, for every
valid token at every MoE layer, the normalized top-k probability mass on the
added experts. Launched with torchrun: each rank loads a full model copy on its
GPU and observes a disjoint share of the windows.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from common.io_utils import read_json, write_json_atomic
from common.run_record import write_run_record
from common.tokenizers import ensure_pad_token
from moe.model_adapters import adapter_from_checkpoint, diagnostic_topk_weights, model_adapter

WINDOWS_PER_CORPUS = 512
WINDOW_TOKENS = 2048
RETAIN_DOCUMENTS = 4096
SEED = 42
BATCH_SIZE = 4
CDF_POINTS = 1001
DEFINITION = ("float32 normalized selected top-k probability mass on added experts "
              "before dispatch casting and expert scaling; no masking")


def corpus_windows(tokenizer, directory, corpus, *, seed=42, windows=512, length=2048, documents=None):
    """Reservoir sampling over nonoverlapping windows, including long-document tails."""
    if windows < 1 or length < 2 or (documents is not None and documents < 1):
        raise ValueError("Invalid sampling budget")
    rng, rows = random.Random(seed), []
    docs = tokens = eligible = 0
    sources = {}
    paths = sorted(Path(directory).glob('*.jsonl'))
    if not paths:
        raise ValueError(f"No training JSONL files: {directory}")
    for path in paths:
        digest, consumed = hashlib.sha256(), 0
        with path.open('rb') as handle:
            for line_index, line in enumerate(handle):
                if documents is not None and docs >= documents:
                    break
                raw = json.loads(line)
                text = raw.get('text', raw.get('content'))
                if not isinstance(text, str):
                    raise ValueError(f"Missing corpus text: {path}:{line_index}")
                digest.update(line)
                consumed += 1
                ids = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
                docs += 1
                tokens += len(ids)
                for offset in range(0, len(ids), length):
                    chunk = ids[offset:offset+length]
                    if len(chunk) < 2:
                        continue
                    item = dict(id=f'{corpus}:{path.name}:{line_index}:{offset}', corpus=corpus,
                                source=str(path.resolve()), line_index=line_index, token_offset=offset, input_ids=chunk)
                    eligible += 1
                    if len(rows) < windows:
                        rows.append(item)
                    else:
                        replacement = rng.randrange(eligible)
                        if replacement < windows:
                            rows[replacement] = item
        sources[str(path.resolve())] = dict(consumed_documents=consumed, consumed_prefix_sha256=digest.hexdigest(), file_bytes=path.stat().st_size)
        if documents is not None and docs >= documents:
            break
    if not rows:
        raise ValueError("Empty routing sample")
    return sorted(rows, key=lambda r: r['id']), dict(seed=seed, requested_windows=windows, sampled_windows=len(rows),
        window_length=length, documents_in_pool=docs, tokens_in_pool=tokens, eligible_windows=eligible,
        sampled_tokens=sum(len(r['input_ids']) for r in rows), sources=sources,
        tokenization='plain training text; no BOS; append EOS; no document-head truncation',
        coverage='all forget documents in sampling pool, evaluated subset only' if documents is None
                 else f'first {documents} training retain documents in sorted file order, evaluated subset only')


class RoutingMassProbe:
    """Hooks return None and observe every valid token at every MoE layer."""
    def __init__(self, model, harmful_experts):
        self.adapter = model_adapter(model)
        self.layers = list(self.adapter.moe_layers(model))
        self.harmful = tuple(harmful_experts)
        if not self.layers or not self.harmful or len(set(self.harmful)) != len(self.harmful):
            raise ValueError("Missing MoE layers or invalid bank")
        self.active = False
        self.handles = [layer.router.register_forward_hook(self.hook(layer.index)) for layer in self.layers]

    def begin(self, attention_mask):
        if self.active or not attention_mask.bool().any():
            raise ValueError("Invalid observation start")
        self.mask, self.values, self.active = attention_mask.bool(), {}, True

    def hook(self, layer):
        def observe(_module, _inputs, output):
            if not self.active:
                return
            if layer in self.values:
                raise ValueError(f"Repeated router in one forward: {layer}")
            weights, selected = diagnostic_topk_weights(self.adapter, output)
            weights = weights.reshape(*self.mask.shape, -1)
            selected = selected.reshape_as(weights)
            bank = torch.tensor(self.harmful, device=selected.device)
            harmful = (selected[..., None] == bank).any(-1)
            mass = (weights * harmful).sum(-1)[self.mask]
            if not torch.isfinite(mass).all() or (mass < 0).any() or (mass > 1.000001).any():
                raise ValueError("Invalid routing mass")
            self.values[layer] = mass.detach().cpu().numpy().copy()
        return observe

    def end(self):
        if not self.active:
            raise ValueError("No active routing observation")
        self.active = False
        if set(self.values) != {layer.index for layer in self.layers}:
            raise ValueError("Incomplete routing observation")
        return np.stack([self.values[layer.index] for layer in self.layers], axis=1)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.active = False


def tensors(rows, pad_token, device):
    ids = torch.full((len(rows), max(len(r['input_ids']) for r in rows)), pad_token, device=device, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for index, row in enumerate(rows):
        length = len(row['input_ids'])
        ids[index, :length] = torch.tensor(row['input_ids'], device=device)
        mask[index, :length] = 1
    return dict(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)


def prepare(tokenizer_dir, forget, retain, output):
    """Sample the shared windows of both corpora into output/{inputs.jsonl,metadata.json}."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True, local_files_only=True)
    ensure_pad_token(tokenizer)
    output.mkdir(parents=True, exist_ok=False)
    rows, coverage = [], {}
    for corpus, directory, limit in [('forget', forget, None), ('retain', retain, RETAIN_DOCUMENTS)]:
        sample, coverage[corpus] = corpus_windows(tokenizer, directory, corpus, windows=WINDOWS_PER_CORPUS,
                                                  length=WINDOW_TOKENS, seed=SEED, documents=limit)
        rows.extend(sample)
    path = output / 'inputs.jsonl'
    with path.open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    write_json_atomic(output / 'metadata.json', dict(tokenizer=str(Path(tokenizer_dir).resolve()), pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id, input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), coverage=coverage, training_corpora_only=True))


def collect(checkpoint, all_rows, pad_token_id, output, *, rank, world, local_rank):
    """Observe this rank's share of the windows; write output/rank-<rank>.npz."""
    from transformers import AutoConfig
    from moe.runtime import load_expanded_moe_manifest_metadata
    torch.manual_seed(SEED)
    device = torch.device('cuda', local_rank)
    rows = []
    for corpus in ('forget', 'retain'):
        subset = sorted([r for r in all_rows if r['corpus'] == corpus], key=lambda r: (-len(r['input_ids']), r['id']))
        rows.extend(subset[rank::world])
    assert rows
    adapter = adapter_from_checkpoint(checkpoint)
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    model, loading = adapter.auto_model_class(config).from_pretrained(checkpoint, dtype=torch.bfloat16, device_map={'': device},
        attn_implementation='sdpa', trust_remote_code=True, local_files_only=True, output_loading_info=True)
    assert not any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')), loading
    model.eval().requires_grad_(False)
    old, harmful, count = load_expanded_moe_manifest_metadata(checkpoint)
    assert tuple(harmful) == tuple(range(old, count)) and adapter.config_num_experts(config) == count
    # The probe must be passive: same logits with hooks, no parameter writes.
    versions = [(p, p._version) for p in model.parameters()]
    batch = tensors(rows[:BATCH_SIZE], pad_token_id, device)
    with torch.inference_mode():
        plain = model(**batch).logits
        probe = RoutingMassProbe(model, harmful)
        probe.begin(batch['attention_mask'])
        observed = model(**batch).logits
        probe.end()
        assert torch.equal(plain, observed) and torch.isfinite(observed).all(), 'Hooks changed logits'
    values = []
    try:
        for start in range(0, len(rows), BATCH_SIZE):
            batch = tensors(rows[start:start+BATCH_SIZE], pad_token_id, device)
            with torch.inference_mode():
                probe.begin(batch['attention_mask'])
                logits = model(**batch).logits
                assert torch.isfinite(logits).all()
                values.append(probe.end())
            if start % 32 == 0:
                print(f'rank={rank} windows={start}/{len(rows)}', flush=True)
    finally:
        probe.close()
    assert all(p._version == version for p, version in versions), 'Parameter mutation'
    lengths = np.array([len(r['input_ids']) for r in rows], dtype=np.int64)
    mass = np.concatenate(values, axis=0)
    assert mass.shape == (int(lengths.sum()), len(probe.layers))
    with (output / f'rank-{rank}.npz').open('xb') as handle:
        np.savez_compressed(handle, z=mass, ids=np.array([r['id'] for r in rows]), corpora=np.array([r['corpus'] for r in rows]),
            lengths=lengths, offsets=np.concatenate([[0], lengths.cumsum()]), layers=np.array([layer.index for layer in probe.layers]))
    return dict(model_family=adapter.name, old_num_experts=old, new_num_experts=count, harmful_experts=list(harmful))


def summarize(directory, world):
    """Token-weighted per-layer means and CDFs over all ranks of one node."""
    layers = None
    grid = np.linspace(0., 1., CDF_POINTS)
    sums, counts, cdf_counts, mean_cdf_counts = {}, {}, {}, {}
    windows = 0
    for rank in range(world):
        with np.load(directory / f'rank-{rank}.npz', allow_pickle=False) as data:
            z, corpora, offsets = data['z'], data['corpora'], data['offsets']
            if layers is None:
                layers = data['layers'].tolist()
            assert data['layers'].tolist() == layers
            windows += len(data['ids'])
            for corpus in ('forget', 'retain'):
                blocks = [z[int(offsets[i]):int(offsets[i+1])] for i, c in enumerate(corpora) if str(c) == corpus]
                assert blocks
                values = np.concatenate(blocks, axis=0)
                sums[corpus] = sums.get(corpus, np.zeros(len(layers))) + values.sum(axis=0, dtype=np.float64)
                counts[corpus] = counts.get(corpus, 0) + len(values)
                cdf_counts[corpus] = cdf_counts.get(corpus, np.zeros(len(grid), dtype=np.int64)) + np.searchsorted(np.sort(values, axis=None), grid, side='right')
                mean_cdf_counts[corpus] = mean_cdf_counts.get(corpus, np.zeros(len(grid), dtype=np.int64)) + np.searchsorted(np.sort(values.mean(axis=1, dtype=np.float64)), grid, side='right')
    return dict(status='complete', layers=layers, windows=windows, definition=DEFINITION,
        corpora={c: dict(valid_tokens=counts[c], token_layer_observations=counts[c]*len(layers), mean_z=float(sums[c].sum()/(counts[c]*len(layers))),
            layer_mean_z=(sums[c]/counts[c]).tolist(), cdf_grid=grid.tolist(),
            token_layer_cdf=(cdf_counts[c]/(counts[c]*len(layers))).tolist(),
            token_layer_mean_cdf=(mean_cdf_counts[c]/counts[c]).tolist()) for c in ('forget', 'retain')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('f0', 'f1', 'forget', 'retain', 'output'):
        parser.add_argument('--' + flag, required=True, type=Path)
    args = parser.parse_args()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.set_num_threads(4)
    torch.cuda.set_device(local_rank)
    # Only used to wait for the other ranks; rank 0 tokenizes the full forget
    # corpus while the others wait, hence the long timeout.
    dist.init_process_group('gloo', timeout=timedelta(hours=6))
    nodes = {'F0': args.f0.resolve(), 'F1': args.f1.resolve()}
    sampling = args.output / 'sampling'
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        prepare(nodes['F0'], args.forget, args.retain, sampling)
        for node in nodes:
            (args.output / node).mkdir()
    dist.barrier()
    metadata = read_json(sampling / 'metadata.json')
    all_rows = [json.loads(line) for line in (sampling / 'inputs.jsonl').open()]
    assert len({r['id'] for r in all_rows}) == len(all_rows)
    structure = {}
    for node, checkpoint in nodes.items():
        structure[node] = collect(checkpoint, all_rows, metadata['pad_token_id'], args.output / node,
                                  rank=rank, world=world, local_rank=local_rank)
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
    if rank == 0:
        summaries = {}
        for node in nodes:
            summary = summarize(args.output / node, world)
            summary.update(node=node, **structure[node],
                           uniform_bank_reference=len(structure[node]['harmful_experts']) / structure[node]['new_num_experts'])
            summaries[node] = summary
        assert summaries['F0']['layers'] == summaries['F1']['layers']
        protocol = dict(windows_per_corpus=WINDOWS_PER_CORPUS, window_tokens=WINDOW_TOKENS, retain_documents=RETAIN_DOCUMENTS,
                        seed=SEED, add_bos=False, forget_sampling_pool='all training forget documents')
        inference = dict(attention='sdpa', batch_size_per_gpu=BATCH_SIZE, dtype='bfloat16', world_size=world)
        write_json_atomic(args.output / 'routing_mass.json', dict(
            status='complete', kind='routing_mass', training_corpora_only=True, sampling=metadata,
            nodes=summaries, protocol=protocol, inference=inference), sort_keys=True)
        write_run_record(
            args.output, step='eval_routing_mass',
            inputs={'F0': nodes['F0'], 'F1': nodes['F1'], 'forget': args.forget, 'retain': args.retain},
            args=dict(protocol=protocol, inference=inference, tokenizer='F0'),
            results={node: {c: dict(mean_z=v['mean_z'], valid_tokens=v['valid_tokens']) for c, v in s['corpora'].items()}
                     for node, s in summaries.items()})
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
