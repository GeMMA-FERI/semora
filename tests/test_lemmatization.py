from __future__ import annotations

import sys
import types
from pathlib import Path

from semora.text import ClasslaLemmatizer, download_classla_models


def test_classla_adapter_uses_minimal_pipeline_and_offsets(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    class FakePipeline:
        def __init__(self, language: str, **options) -> None:
            calls["pipeline"] = (language, options)

        def __call__(self, _text: str):
            word = types.SimpleNamespace(lemma="gledališče", text="gledališča")
            token = types.SimpleNamespace(start_char=0, end_char=10, words=[word], text="gledališča")
            sentence = types.SimpleNamespace(tokens=[token])
            return types.SimpleNamespace(sentences=[sentence])

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
        types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
    )

    lemmatizer = ClasslaLemmatizer(device="auto", resources_dir=tmp_path)
    assert lemmatizer.lemmatize("gledališča") == "gledališče"
    assert lemmatizer.annotate("gledališča")[0].start == 0
    language, options = calls["pipeline"]
    assert language == "sl"
    assert options["processors"] == "tokenize,pos,lemma"
    assert options["use_gpu"] is False
    assert options["dir"] == str(tmp_path.resolve())

    download_classla_models(resources_dir=tmp_path)
    language, options = calls["download"]
    assert language == "sl"
    assert options["processors"] == "tokenize,pos,lemma"
