"""Bounded CPU/CUDA timeline: python -m semora.diagnostics.ner.profile_classla."""

import argparse
from contextlib import closing
from functools import wraps
from pathlib import Path
from unittest.mock import patch

from .adapters import ClasslaAdapter
from .sample import readonly


def main():
    import torch
    from classla.models.ner import data, model, trainer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--articles", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.articles < 1 or args.batch_size < 1:
        parser.error("Articles and batch size must be positive")
    if args.output.exists():
        parser.error("Output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with closing(readonly(args.sample)) as source:
        texts = [
            r["content"]
            for r in source.execute("SELECT content FROM articles ORDER BY position LIMIT ?", (args.articles,))
        ]
    adapter = ClasslaAdapter(device="cuda")
    try:
        adapter.predict(texts[:2], args.batch_size)

        def marked(function, name):
            @wraps(function)
            def call(*a, **kw):
                with torch.profiler.record_function(name):
                    return function(*a, **kw)

            return call

        with (
            patch.object(data.DataLoader, "__init__", marked(data.DataLoader.__init__, "ner.vocabulary")),
            patch.object(data.DataLoader, "__getitem__", marked(data.DataLoader.__getitem__, "ner.cpu_tensors")),
            patch.object(trainer, "unpack_batch", marked(trainer.unpack_batch, "ner.transfer_inputs")),
            patch.object(model.NERTagger, "forward", marked(model.NERTagger.forward, "ner.neural_forward")),
            patch.object(trainer, "viterbi_decode_batch", marked(trainer.viterbi_decode_batch, "ner.cpu_decode")),
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            ) as profile,
        ):
            adapter.predict(texts, args.batch_size)
        profile.export_chrome_trace(str(args.output))
        print(profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))
        print(profile.key_averages().table(sort_by="self_device_time_total", row_limit=15))
        print(f"Timeline saved to {args.output}; profiling timings are not throughput benchmarks.")
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
