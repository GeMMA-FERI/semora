from __future__ import annotations

import contextlib
import re
import sys
import types
from pathlib import Path

from semora.text import ClasslaLemmatizer, download_classla_models


def test_classla_adapter_uses_minimal_pipeline_and_offsets(monkeypatch, tmp_path: Path) -> None:
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
    assert lemmatizer.annotate("gledališča")[0].start == 0
    documents = lemmatizer.annotate_many(["Prvi dokument", "Drugi @@EOD@@ dokument"])
    assert [[lemma for token in document for lemma in token.lemmas] for document in documents] == [
        ["prvi", "dokument"],
        ["drugi", "eod", "dokument"],
    ]
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
