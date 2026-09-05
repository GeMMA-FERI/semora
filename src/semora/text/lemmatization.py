"""Slovene lemmatization through the optional CLASSLA pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class LemmaToken:
    start: int
    end: int
    lemmas: tuple[str, ...]


class Lemmatizer(Protocol):
    def annotate(self, text: str) -> list[LemmaToken]: ...

    def lemmatize(self, text: str) -> str: ...


class ClasslaLemmatizer:
    """Minimal tokenize/POS/lemma CLASSLA pipeline for standard Slovene."""

    def __init__(
        self,
        *,
        pipeline_type: str = "default",
        device: str = "auto",
        resources_dir: str | Path | None = None,
    ) -> None:
        try:
            import classla
            import torch
        except ImportError as exc:
            raise RuntimeError("Slovene lemmatization requires the 'classla' extra.") from exc
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("CLASSLA device must be auto, cpu, or cuda.")
        use_gpu = torch.cuda.is_available() if device == "auto" else device == "cuda"
        if device == "cuda" and not use_gpu:
            raise RuntimeError("CLASSLA was asked to use CUDA, but CUDA is unavailable.")
        options: dict[str, Any] = {
            "type": pipeline_type,
            "processors": "tokenize,pos,lemma",
            "use_gpu": use_gpu,
            "verbose": False,
        }
        if resources_dir is not None:
            options["dir"] = str(Path(resources_dir).resolve())
        self.pipeline = classla.Pipeline("sl", **options)

    def annotate(self, text: str) -> list[LemmaToken]:
        if not text.strip():
            return []
        document = self.pipeline(text)
        result: list[LemmaToken] = []
        for sentence in document.sentences:
            for token in sentence.tokens:
                if token.start_char is None or token.end_char is None:
                    continue
                lemmas = tuple(
                    str(word.lemma or word.text).strip()
                    for word in token.words
                    if str(word.lemma or word.text).strip() not in {"", "_"}
                )
                if not lemmas and str(token.text).strip():
                    lemmas = (str(token.text).strip(),)
                if lemmas:
                    result.append(
                        LemmaToken(
                            start=int(token.start_char),
                            end=int(token.end_char),
                            lemmas=lemmas,
                        )
                    )
        return result

    def lemmatize(self, text: str) -> str:
        return " ".join(lemma for token in self.annotate(text) for lemma in token.lemmas)


def download_classla_models(
    *,
    pipeline_type: str = "default",
    resources_dir: str | Path | None = None,
) -> None:
    try:
        import classla
    except ImportError as exc:
        raise RuntimeError("CLASSLA model download requires the 'classla' extra.") from exc
    options: dict[str, Any] = {"type": pipeline_type, "processors": "tokenize,pos,lemma"}
    if resources_dir is not None:
        options["dir"] = str(Path(resources_dir).resolve())
    classla.download("sl", **options)
