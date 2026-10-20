# Repository discipline

See README.md for installation, public data examples and entrypoints.

If `internal/RULES.md` exists in a local checkout, also follow its local execution
and storage rules. `internal/` is private and excluded from Git. A local `report/`
workspace is a separate repository and is ignored independently.

## 命名（唯一定义）

方法名为 **REX**（论文 `\sysname{}`）。代码、配置与脚本中不再使用 SGTM 指代本方法；SGTM 只指外部工作（Shilov et al., 2025）。

下表的 256E/264E 与 experts 256–263 是 Intern-S2 配方的具体数值。跨模型 Node 含义相同，基础专家数 N 与新增数 K 读取 checkpoint 的 `config.json` 与 `expert_expansion_manifest.json`；Gemma 4 本地输入 N=128，Qwen3.5 N=256。三个模型的 `B` 均按 base 版本对待（Intern 的 base 版本基于 Qwen instruct 构建）。

| Node | 定义 | 形态 |
| --- | --- | --- |
| `B` | 原始 base checkpoint | standard 256E |
| `F0` | 选择 + 克隆 + 缩放（零训练） | expanded 264E |
| `A0` | `E(F0)` | standard 256E |
| `F1` | `F0` 经 REX | expanded 264E |
| `A1` | `E(F1)`，与 `B` 同构 | standard 256E |
| `H1` | `F1` 的 experts 256–263 与对应 router rows | **非独立 checkpoint** |
| `A2` | `A1` 经定制 residual RMU | standard 256E |
| `F2` | `A2` 保留参数 + `H1` | expanded 264E |
| `F3(t)` | `S_std(F2)` 在 `t` tokens | expanded 264E |
| `A3(t)` | `E(F3(t))` | standard 256E |
| `A3-direct(t)` | `S_std(A2)` 在 `t` tokens | standard 256E |
| `R` | `B` 经同 operator/预算的 Pure RMU | standard 256E |
| `R-SFT(t)` | `S_std(R)` 在 `t` tokens | standard 256E |

`E(·)` = 固定专家切除；`S_std(·)` = 预先固定的标准全参 SFT；`T(a,b)(·)` = 协议 `a`、预算 `b` 的微调压力测试。

### 命名纪律

- `F0`/`F1`/`A1`/`A2`/`F2`/`F3` 是**不同的真实 checkpoint**，各自带 `run.json`，不是同一 checkpoint 的 phase。
- **`H1` 不是独立 checkpoint**：重拼从同源 `F1` 恢复 RMU 未更新的部分。不得写成可加载模型。不引入 `H3`，`F3` 中对应位置称「`F3` 的 removed bank」。
- **停用 `Ours (ablate)`**：物理切除是方法的必要组成。L1 的 Ours 指 `A2`，L2 的 Ours 指 `A3`。
- `source_expert_scale`（构造期缩放）与 `std_expert_lr_multiplier`（训练期学习率倍率）不可混淆；`harmful_router_lr_multiplier`（基础倍率）与 `harmful_router_std_step_lr_multiplier`（仅 std step 的附加倍率）不可混称。
- standard checkpoint（`B`/`R`/`A1`/`A2`）只有自身形态，不虚构第二次切除。

---

## 代码边界

- 公开入口只有 `scripts/<动词_对象>.sh`，每个脚本单一用途、位置参数给出输入与输出，只调用一个 Python 模块的 `main()`；流水线 `pipelines/*.sh` 只串联 scripts。不新增多模式 CLI 或子命令。公开脚本假定已在 GPU 环境内执行，不含集群路径、rjob 或存储桶名。
- 依赖方向自上而下：`common/` → `moe/` → `surgery/`；`common/`、`moe/` → `training/` → `methods/{rex,rmu}/`。`training/` 不依赖方法包；方法包不依赖 `storage/`；`surgery/` 不依赖 `training/`。
- 模型架构、参数角色与 checkpoint 布局集中于 `moe/model_adapters.py`；MoE runtime 在 `moe/runtime.py`，离线变换在 `surgery/`。不得在训练循环、评测或序列化代码中新增各自的 expert key 匹配。
- REX 与 RMU 共用 `training/` 的 setup、参考模型加载、配对数据流与 optimizer step，共用 `common/checkpoint_io.py` 保存。方法 loss、split、预算与 scheduler 语义留在各方法包。
- 配方配置不含机器路径；输入、输出路径由脚本参数给出。
- 每个产物只写一份记录 `run.json`（`common/run_record.py`），只作记录，不回读校验。另保留结构性的 `expert_expansion_manifest.json`，供加载器识别可移除专家。不新增其它 manifest 或血缘校验。
- 产物以节点目录为单位：`<run_id>/<node>/` 下放权重（临时节点 A0/A1/A2/A3 等只保留记录与评测，不存权重）、`run.json`、`eval/<protocol>/<task>/`、SFT 节点的 `online_eval/tokens-<N>/`、F1 的 `diagnostics/`。本地 run 目录与对象存储布局一一对应，对象存储只增不改：已存在且内容不同的对象不得覆盖，`run.json` 最后上传。
- 评测协议定义在 `eval/protocols.yaml`。协议（如 `lm-eval-vllm-v1`）固定共用运行设定（后端与模型参数、种子、chat template/thinking、lm-eval 与 vLLM 版本），任一改变须升版本号；协议下的任务规格只增不改，追加指标不改版本号，调整已有任务设定只能新增任务名。数据并行规模不属于协议，记录在 `run.json`。
- `dataset/xtuner_vendor/` 是 vendored XTuner，除底层行为确需同步外不修改。`sft/` 基于外部 ms-swift Megatron，只通过 `sft/swift_plugin.py` 接入，不修改 swift 本身。
- 已知例外：SFT 由 ms-swift 自行导出 HF 权重，不经 `common/checkpoint_io`（见 `sft/swift_checkpoint.py`）。
- 保留跨域适配代码不等于批准对应域实验。

---

## REX 训练语义（`F0 → F1`）

输入必须是 expanded MoE。参数分为：

| 组 | 参数 |
| --- | --- |
| `theta_harmful` | `mlp.harmful_gate`、`mlp.harmful_experts.*` |
| `theta_std` | `mlp.gate`、`mlp.experts.*` |
| `theta_shared` | attention、embedding、norm、shared expert/gate、lm head、视觉模块及其它 |

`theta_shared` **始终冻结并排除在 optimizer 之外**。两个 split 的更新范围固定：

| split | 输入与激活 | 可更新参数 |
| --- | --- | --- |
| `std` | retain；pre-top-k 屏蔽 harmful experts | 两组 router + std experts |
| `harmful` | target；不屏蔽 | 两组 router + harmful experts |

主方法是纯 scale + REX，不含交错 RMU。Router 统计在 mask **前**收集，实际 dispatch 使用 mask **后**重新 top-k 的结果。

- `rex_variant` 默认 `standard`。诊断值 `gradient_mask_only` 关闭 std activation mask 与全部路由质量/rank loss；`without_std_activation_mask` 仅关闭 std activation mask；`without_routing_loss` 仅关闭两 split 的路由质量 loss 与 std rank loss。三者保留梯度掩码、共享参数冻结、同源 F0、相同学习率、预算与顺序，不替代主线配方。
- `harmful_expert_lr_multiplier=0` 表示新增专家零参数更新，但保留 `requires_grad`、optimizer 状态与 harmful step 的全局梯度裁剪贡献，不等同于关闭反向传播。其它学习率倍率须为正数。
- 上述参数名是 Qwen/Intern-S2 布局。Gemma 的 routed experts 与 router 位于 decoder 层下，常开 dense MLP 属于冻结共享参数。Gemma `router.per_expert_scale` 默认按专家拆入两组 router 并使用对应 router 学习率；可选 `gemma_harmful_expert_scale_lr`：原有专家 scale 冻结，新增专家 scale 单独使用该初始学习率（随原 scheduler 衰减、无 weight decay），仅 harmful step 更新，值为 0 时两组 scale 均冻结。跨专家共享的 `router.scale` 始终冻结。统计 mass 使用逐专家缩放前的归一化 top-k 权重，rank loss 使用原始 logits，实际 dispatch 保留缩放。
- 保存时 merge 回标准 packed HF state-dict key；runtime-only harmful key 不得进入最终权重。实际 HF 文件布局按 loader 检查，不能只凭 logical layout 推断。

---

## RMU 语义（`A1 → A2` 与 `B → R`）

`rmu_monitor_layer_indices`（提取 activation、计算 loss）与 `rmu_trainable_layer_indices`（允许更新的层）必须分开配置。参数 scope 显式声明，不从参数名隐式猜测：

- `selected_routed_and_shared_expert_mlp`：更新指定层 routed expert 全部 MLP projection 与 shared expert MLP；router、shared expert gate、attention、embedding、norm、lm head、视觉参数冻结。Gemma 对应更新 routed experts 与常开 dense MLP 的 projection。
- `selected_std_expert_down_proj_only`：只更新指定层 std expert 的 `down_proj`，常开 dense MLP 冻结（Gemma Cyber 的 A2 与 matched R 使用此 scope）。

### matched 约束（硬约束）

`A2` 与 `R` 使用**同一份** `configs/<model>/wmdp-<domain>/rmu.yaml`，即完全相同的 RMU operator、优化超参与 forget budget，**只允许 parent / retain reference / order 不同**。

- retain reference 永远是当前 pre-RMU parent（`A2` → `A1`；`R` → `B`），由 `scripts/train_rmu.sh` 的 PARENT 参数给出。
- `rmu.yaml` 若改动，必须从 `B` 用同一配方补跑 `R`，不得套用旧 `R` 数值。
- 实际 forget / retain tokens 分开记录在 `run.json`；paired retain tokens 不决定预算。

---

## 血缘记录

每个 checkpoint 一份 `run.json`：生成它的脚本与 git commit、输入路径、完整配方与参数、实际计数，并内嵌输入目录的 `run.json`，因此完整血缘随产物传递。**不从目录名推断血缘。** `F2` 重拼必须用产生该 `A1` 的同源 `F1`（`pipelines/f1_to_f2.sh` 保证）。

---

## 数据与证据边界

- Selection 的 target/control routing statistics 只使用训练语料，不读取 MMLU/WMDP 评测 prompt。
- 语料遍历范围、长文处理与有效 tokens 必须一起记录；不能把首段截断统计称为完整 token 覆盖。
- 完整 forget corpus 可用于 expansion、REX、RMU 与攻击；不另划 target train/test。
- WMDP 与 MMLU 等下游指标用于调参与验证，不描述为 held-out 或 confirmatory test。

### 指标边界

- WMDP 是 4 选 1，随机水平 0.25。贴地板后的细小差异不自动代表更强遗忘，须结合 stderr 与能力结果解释。
- MMLU 的任务知识评估不能替代指令遵循评估（IFEval 使用 chat template，thinking 关）。
- 报告 `Rec` 时必须并报绝对 ΔW。
- 「最坏情况」在有限实验中一律称「最强已测试攻击」；未达阈值写「> b_max」，不写「无法恢复」或无穷。
- 独立评测统一使用 lm-eval vLLM 后端，只评 checkpoint 的真实形态；切除后评测 = 先切除再评测。

---

## 比较约束

### 三层不可互相推导

L1（遗忘后基座）、L2（标准 SFT 后）、L3（发布后恶意微调）分别评价不同性质。L3 不是前两层的推论，三层也不要求都全面领先。若声称某组件改善 L3 鲁棒性，必须报告它在 L3 的结果。

### 对照配对

- `A2` 与 `R`、`A3` 与 `R-SFT` 分别记录 matched 比较；`A2 → A3` 记录同一路线不同节点的实际指标，不预设变化方向。
- `A3-direct`（`S_std(A2)`）是 **− recombination** 对照，单独命名，不与 `A3` 曲线混称。
- 方法主线是 `A2` → 重拼 `F2` → 全参 SFT → 再次物理切除交付。完整方法的收益不能表述成 `A2` 自身的耐久性。
- `F2`/`F3` 的 full 结果与 excised 结果分别报告；standard 模型只评其真实形态。

### SFT 与攻击

- 所有耐久性训练先完成 unlearning。
- 每条轨迹是一条 250M cosine trajectory（`configs/sft/` 的 S_std 配方）；匹配比较使用相同数据顺序与 seed；只保存终点权重，milestone 做训练中驻留评测。`A3`、`A3-direct` 与 `R-SFT` 使用完全相同的 `S_std`，只差起点。
- 禁止拼接独立预算为一条曲线；禁止 SFT 后补 RMU 支持耐久性 claim。0-token 点是输入 checkpoint，不是训练任务。
- 良性 SFT（L2，开发者行为）与恶意微调（L3，攻击者行为）分开记录，两者 token 预算不可直接比较强度。
- 攻击协议（数据、预算网格、可更新参数范围、lr、搜索预算、阈值、验证/测试划分）必须在运行前记录用户确认的协议，不得事后按结果调整。测试集不得参与 lr / seed / checkpoint 选择。
- 每条 SFT 轨迹使用唯一的 S3 前缀，避免远端对象覆盖。

### 选择性 claim

声称「切除位置有选择性」必须有同层同数量、多随机种子的随机专家切除对照（`scripts/excise_experts.sh --random-original-experts-seed`）。「不做 REX」类消融不能只是省去训练并悄悄减少 token 暴露；无法匹配的因素必须明确说明。

### 排期与授权


---

## 修改与验证

- 修改前检索真实调用关系；保留工作区中与任务无关的改动。删除前明确目标，**绝不递归操作仓库根目录、`checkpoints/` 或 `data/`**。
- 不以单元测试作为验证手段。脚本修改检查 `bash -n` 与 `--help`；Python 修改至少确认可导入；checkpoint 变换与训练修改用真实 checkpoint 对比新旧代码输出（逐 tensor 比较、短预算 loss 曲线），需要 GPU 时先取得授权。
- 文档修改检查来源、引用与当前/历史状态。不能把静态检查描述为 GPU 或真实 checkpoint 验证。交付时列出实际执行与未执行的检查。
- 代码 ready 前不提交大规模训练。
