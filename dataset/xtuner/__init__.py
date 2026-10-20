"""REX-local XTuner-style data pipeline."""

__all__ = [
    "SUPPORTED_DATA_FORMAT_TYPES",
    "create_xtuner_dataloaders",
    "load_dataset_sources",
]


def __getattr__(name: str):
    if name == "create_xtuner_dataloaders":
        from .loader import create_xtuner_dataloaders

        return create_xtuner_dataloaders
    if name in {"SUPPORTED_DATA_FORMAT_TYPES", "load_dataset_sources"}:
        from .recipe import SUPPORTED_DATA_FORMAT_TYPES, load_dataset_sources

        return {
            "SUPPORTED_DATA_FORMAT_TYPES": SUPPORTED_DATA_FORMAT_TYPES,
            "load_dataset_sources": load_dataset_sources,
        }[name]
    raise AttributeError(name)
