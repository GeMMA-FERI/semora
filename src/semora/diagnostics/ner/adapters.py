"""Lazy model adapters. Extra adapters may be supplied as module:factory."""

import importlib
import re
import time
from typing import Any

from .base import Adapter, Entity, Prediction


def load_adapter(name: str, **options: Any) -> Adapter:
    factories = {"classla": ClasslaAdapter, "transformers": TransformersAdapter}
    if name in factories:
        return factories[name](**options)
    module, separator, factory = name.partition(":")
    if not separator:
        raise ValueError("Adapter must be classla, transformers, or module:factory")
    return getattr(importlib.import_module(module), factory)(**options)


class ClasslaAdapter:
    def __init__(
        self, *, device: str, model: str | None = None, resources: str | None = None, download: bool = False, **_: Any
    ):
        import classla
        import torch

        self.torch = torch
        self.device = device
        options: dict[str, Any] = {"processors": "tokenize,ner", "type": model or "default"}
        if resources:
            options["dir"] = resources
        if download:
            classla.download("sl", **options)
        self.pipeline = classla.Pipeline("sl", **options, use_gpu=device == "cuda", verbose=False)
        self.metadata = {
            "adapter": "classla",
            "package": model or "default",
            "processors": "tokenize,ner",
            "classla_version": classla.__version__,
            "ner_config": self.pipeline.processors["ner"].config,
        }

    def predict(self, texts: list[str], batch_size: int) -> Prediction:
        from classla.models.common.doc import Document

        started = time.perf_counter()
        documents = [self.pipeline.processors["tokenize"].process(text) for text in texts]
        tokenized = time.perf_counter()
        # Batch sentences while retaining each sentence's original document pointer.
        # This preserves article boundaries and exact offsets without text sentinels.
        combined = Document([])
        combined.sentences = [sentence for document in documents for sentence in document.sentences]
        combined.num_tokens = sum(document.num_tokens for document in documents)
        combined.num_words = sum(document.num_words for document in documents)
        self.pipeline.processors["ner"].config["batch_size"] = batch_size
        if combined.sentences:
            with self.torch.inference_mode():
                self.pipeline.processors["ner"].process(combined)
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        inferred = time.perf_counter()
        result = [
            [Entity(e.start_char, e.end_char, e.type, e.text) for e in document.build_ents()] for document in documents
        ]
        return Prediction(
            result,
            {
                "tokenize_seconds": tokenized - started,
                "inference_seconds": inferred - tokenized,
                "postprocess_seconds": time.perf_counter() - inferred,
            },
        )


class TransformersAdapter:
    def __init__(
        self,
        *,
        device: str,
        model: str | None = None,
        revision: str = "main",
        max_length: int = 256,
        stride: int = 64,
        pretokenized: bool = False,
        precision: str = "float32",
        **_: Any,
    ):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

        model = model or "ivlcic/xlmr-ner-slavic-pline"
        tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, use_fast=True)
        network = AutoModelForTokenClassification.from_pretrained(
            model,
            revision=revision,
            dtype=getattr(torch, precision),
        )
        maximum = min(tokenizer.model_max_length, getattr(network.config, "max_position_embeddings", max_length))
        if max_length > maximum or not 0 < stride < max_length - tokenizer.num_special_tokens_to_add():
            raise ValueError("Invalid max-length/stride for the model")
        tokenizer.model_max_length = max_length
        self.pipeline = pipeline(
            "token-classification",
            model=network,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1,
            aggregation_strategy="simple",
        )
        self.pretokenized = pretokenized or model == "ivlcic/xlmr-ner-slavic-pline"
        self.stride = stride
        self.torch = torch
        self.device = device
        self.metadata = {
            "adapter": "transformers",
            "model": model,
            "revision": network.config._commit_hash,
            "max_length": max_length,
            "stride": stride,
            "pretokenized": self.pretokenized,
            "precision": precision,
        }

    def predict(self, texts: list[str], batch_size: int) -> Prediction:
        # Pipeline handles overlapping overflow windows and merges duplicate spans.
        # The Slavic checkpoint author prescribes this lossless pretokenization.
        started = time.perf_counter()
        inputs = [re.findall(r"\s+|\w+|[^\w\s]", text) for text in texts] if self.pretokenized else texts
        options = {"is_split_into_words": True, "delimiter": ""} if self.pretokenized else {}
        with self.torch.inference_mode():
            output = self.pipeline(inputs, batch_size=batch_size, stride=self.stride, **options)
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        result = [
            [
                Entity(
                    int(e["start"]),
                    int(e["end"]),
                    e["entity_group"],
                    text[int(e["start"]) : int(e["end"])],
                    float(e["score"]),
                )
                for e in entities
            ]
            for text, entities in zip(texts, output, strict=True)
        ]
        # The high-level pipeline interleaves tokenize/inference/postprocessing.
        return Prediction(result, {"pipeline_seconds": time.perf_counter() - started})
