from __future__ import annotations

import contextlib
import re
import sys
import types
from pathlib import Path

import pytest

from semora.diagnostics.classla import profile_classla
from semora.text import ClasslaLemmatizer, download_classla_models
from semora.text.lemmatization import _lemmas_from_document


def test_classla_adapter_uses_minimal_pipeline_without_offsets(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    class FakeTokenizer:
        def process(self, text: str):
            tokens = []
            for match in re.finditer(r"[^\W\d_]+", text):
                lemma = "gledališče" if match.group() == "gledališča" else match.group().casefold()
                word = types.SimpleNamespace(lemma=lemma, text=match.group())
                tokens.append(
                    types.SimpleNamespace(
                        start_char=match.start(),
                        end_char=match.end() - match.start(),
                        words=[word],
                        text=match.group(),
                    )
                )
            return types.SimpleNamespace(sentences=[types.SimpleNamespace(tokens=tokens)])

    class FakeProcessor:
        @staticmethod
        def process(document):
            return document

    class FakePipeline:
        def __init__(self, language: str, **options) -> None:
            calls["pipeline"] = (language, options)
            self.processors = {
                "tokenize": FakeTokenizer(),
                "pos": FakeProcessor(),
                "lemma": FakeProcessor(),
            }

    def fake_download(language: str, **options) -> None:
        calls["download"] = (language, options)

    monkeypatch.setitem(
        sys.modules,
        "classla",
        types.SimpleNamespace(Pipeline=FakePipeline, download=fake_download),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            inference_mode=contextlib.nullcontext,
        ),
    )

    lemmatizer = ClasslaLemmatizer(
        device="auto",
        resources_dir=tmp_path,
        pos_batch_size=10_000,
        lemma_batch_size=200,
    )
    assert lemmatizer.lemmatize("gledališča") == "gledališče"
    documents = lemmatizer.lemmatize_many(["Prvi dokument", "Drugi @@EOD@@ dokument"])
    assert documents == [
        "prvi dokument",
        "drugi eod dokument",
    ]
    staged = lemmatizer.lemmatize_batches(
        [["Prvi dokument"], ["Drugi dokument", "Tretji dokument"]],
        pipeline_depth=2,
    )
    assert [document for batch in staged for document in batch] == [
        "prvi dokument",
        "drugi dokument",
        "tretji dokument",
    ]
    assert lemmatizer.last_profile is not None
    assert lemmatizer.last_profile.documents == 3
    assert lemmatizer.last_profile.wall_seconds is not None
    language, options = calls["pipeline"]
    assert language == "sl"
    assert options["processors"] == "tokenize,pos,lemma"
    assert options["use_gpu"] is False
    assert options["dir"] == str(tmp_path.resolve())
    assert options["pos_batch_size"] == 10_000
    assert options["lemma_batch_size"] == 200

    download_classla_models(resources_dir=tmp_path)
    language, options = calls["download"]
    assert language == "sl"
    assert options["processors"] == "tokenize,pos,lemma"


def test_operator_profiler_rejects_dangerously_large_traces(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="limited to five articles"):
        profile_classla(["Besedilo."] * 6, tmp_path / "trace.json")


def test_classla_lemma_collection_does_not_require_source_positions() -> None:
    word = types.SimpleNamespace(lemma="-", text="-")
    token = types.SimpleNamespace(
        text="-",
        start_char=0,
        end_char=1,
        words=[word],
    )
    document = types.SimpleNamespace(
        sentences=[types.SimpleNamespace(tokens=[token])],
    )

    result = _lemmas_from_document(["Pred—vojno"], "UNUSED_BOUNDARY", document)

    assert result.documents == ["-"]
    assert result.tokens == 1


def test_classla_adapter_releases_only_unused_cuda_cache() -> None:
    calls = []
    lemmatizer = object.__new__(ClasslaLemmatizer)
    lemmatizer._use_gpu = True
    lemmatizer._torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(empty_cache=lambda: calls.append("empty_cache"))
    )

    lemmatizer.release_cuda_cache()

    assert calls == ["empty_cache"]
