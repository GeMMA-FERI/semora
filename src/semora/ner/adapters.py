"""Lazy model adapters. Extra adapters may be supplied as module:factory."""

import hashlib
import importlib
import re
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from .base import Adapter, Entity, Prediction
from .decoding import word_pipeline_class

# Obeliks TokRulesPart1 explicitly discards C0/C1 control characters.
# They remain in articles.content: permit gaps, never delete/rebase source text.
_TOKENIZER_GAP = re.compile(r"[\s\x00-\x1f\x7f-\x9f]*\Z")


def _match_across_controls(text: str, word: str, cursor: int) -> tuple[int, int]:
    """Match a token merged across discarded controls, without rebasing offsets."""
    start = cursor
    while start < len(text) and _TOKENIZER_GAP.fullmatch(text[start]):
        start += 1
    end = start
    for expected in word:
        while end < len(text) and text[end] != expected:
            char = text[end]
            # Whitespace separates tokens; only non-whitespace C0/C1 controls
            # can disappear inside a token (e.g. J\x03. becomes J.).
            if char.isspace() or not (ord(char) < 32 or 127 <= ord(char) <= 159):
                return -1, -1
            end += 1
        if end == len(text):
            return -1, -1
        end += 1
    return start, end


def align_classla_tokens(document: Any, text: str) -> None:
    """Obeliks offsets restart per paragraph; reconstruct and verify article offsets."""
    # Obeliks normalizes soft hyphens to ASCII hyphens before tokenization.
    # This substitution is one-to-one, so offsets still address the unmodified
    # document text used by build_ents() and stored in the database.
    matching_text = text.replace('\xad', '-')
    cursor = 0
    for token in document.iter_tokens():
        start = matching_text.find(token.text, cursor)
        end = start + len(token.text)
        if start < 0 or not _TOKENIZER_GAP.fullmatch(matching_text[cursor:start]):
            start, end = _match_across_controls(matching_text, token.text, cursor)
            if start < 0:
                raise ValueError(f"Cannot align CLASSLA token {token.text!r} at article character {cursor}")
        # CLASSLA exposes read-only properties backed by these fields.
        token._start_char = start
        token._end_char = end
        cursor = token.end_char
    if not _TOKENIZER_GAP.fullmatch(text[cursor:]):
        raise ValueError(f"CLASSLA tokenization omitted text after article character {cursor}")


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
        self,
        *,
        device: str,
        model: str | None = None,
        resources: str | None = None,
        download: bool = False,
        tokenizer_workers: int = 0,
        sentence_order: str = "input",
        ner_pipeline_depth: int = 1,
        ner_decoder: str = "numpy",
        **_: Any,
    ):
        import classla
        import torch

        self.torch = torch
        self.device = device
        if tokenizer_workers < 0:
            raise ValueError("Tokenizer workers cannot be negative")
        self.tokenizer_workers = tokenizer_workers
        self.tokenizer_pool: ProcessPoolExecutor | None = None
        options: dict[str, Any] = {"processors": "tokenize,ner", "type": model or "default", "ner_decoder": ner_decoder}
        if resources:
            options["dir"] = resources
        if download:
            classla.download("sl", **options)
        self.pipeline = classla.Pipeline("sl", **options, use_gpu=device == "cuda", verbose=False)
        if sentence_order not in ("input", "length"):
            raise ValueError("Sentence order must be input or length")
        self.pipeline.processors["ner"].config["sort_during_eval"] = sentence_order == "length"
        if ner_pipeline_depth < 1:
            raise ValueError("NER pipeline depth must be positive")
        self.pipeline.processors["ner"].config["pipeline_depth"] = ner_pipeline_depth
        self.metadata = {
            "adapter": "classla",
            "package": model or "default",
            "processors": "tokenize,ner",
            "classla_version": classla.__version__,
            "ner_config": self.pipeline.processors["ner"].config,
            "tokenizer_workers": tokenizer_workers,
            "source_sha256": classla_source_hash(classla),
        }
        digest = hashlib.sha256()
        with Path(self.pipeline.processors["ner"].config["model_path"]).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        self.metadata["model_sha256"] = digest.hexdigest()

    def prepare(self, texts: list[str]):
        """CPU-only stage; one caller owns the tokenizer and its optional pool."""
        started = time.perf_counter()
        tokenizer = self.pipeline.processors["tokenize"]
        if self.tokenizer_workers:
            if self.tokenizer_pool is None:
                self.tokenizer_pool = ProcessPoolExecutor(
                    max_workers=self.tokenizer_workers, mp_context=get_context("spawn")
                )
            documents = [
                tokenizer.process_tokenized(*value) for value in self.tokenizer_pool.map(tokenize_article, texts)
            ]
        else:
            documents = [tokenizer.process(text) for text in texts]
        for document, text in zip(documents, texts, strict=True):
            align_classla_tokens(document, text)
        return documents, time.perf_counter() - started

    def predict(self, texts: list[str], batch_size: int) -> Prediction:
        return self.predict_prepared(self.prepare(texts), batch_size)

    def predict_prepared(self, prepared, batch_size: int) -> Prediction:
        from classla.models.common.doc import Document

        documents, tokenize_seconds = prepared
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
                "tokenize_seconds": tokenize_seconds,
                "inference_seconds": inferred - tokenized,
                "postprocess_seconds": time.perf_counter() - inferred,
            },
        )

    def close(self):
        if self.tokenizer_pool is not None:
            self.tokenizer_pool.shutdown(wait=True, cancel_futures=True)
            self.tokenizer_pool = None


def tokenize_article(text: str):
    """Spawn-safe CPU worker: no model or CUDA tensors cross process boundaries."""
    from classla.utils.obeliks import ObeliksTrainer

    return ObeliksTrainer.tokenize(text)


def classla_source_hash(module: Any) -> str:
    """Fingerprint editable code so summaries do not merge different optimizations."""
    root = Path(module.__file__).parent
    digest = hashlib.sha256()
    for relative in (
        "models/ner/data.py",
        "models/ner/model.py",
        "models/ner/trainer.py",
        "models/ner/inference.py",
        "models/ner/numba_decode.py",
        "pipeline/ner_processor.py",
        "models/common/crf.py",
    ):
        digest.update(relative.encode())
        path = root / relative
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") if path.exists() else b"<absent>")
    return digest.hexdigest()


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
        aggregation: str | None = None,
        **_: Any,
    ):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

        model = model or "ivlcic/xlmr-ner-slavic-pline"
        aggregation = aggregation or ("first" if model == "ivlcic/xlmr-ner-slavic-pline" else "simple")
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
            aggregation_strategy=aggregation,
            pipeline_class=word_pipeline_class(),
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
            "aggregation": aggregation,
            "word_id_aggregation": True,
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
