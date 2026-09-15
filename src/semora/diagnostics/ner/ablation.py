"""Fresh-process control adapters for measured optimizations, never production."""


def original_tensors(**options):
    """Retain original Torch row construction for an otherwise identical A/B run."""
    from classla.models.common.data import get_long_tensor
    from classla.models.ner import data

    from .adapters import ClasslaAdapter

    data.get_long_tensor = get_long_tensor
    adapter = ClasslaAdapter(**options)
    adapter.metadata["tensor_backend"] = "original_torch_rows"
    return adapter
