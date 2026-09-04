import re
from abc import ABC, abstractmethod
from typing import Any, cast

try:
    import langchain_text_splitters as _langchain_splitters
except ImportError:  # pragma: no cover - handled when split processor is used
    _langchain_splitters = None  # type: ignore[assignment]

RecursiveCharacterTextSplitter: Any = (
    _langchain_splitters.RecursiveCharacterTextSplitter if _langchain_splitters else None
)


class TextProcessor(ABC):
    """Abstract text processor that turns source text into one or more chunks."""

    @abstractmethod
    def process(self, source_id: str, text: str) -> list[tuple[str, str]]:
        """Return list of (chunk_suffix, chunk_text)."""


class SplitTextProcessor(TextProcessor):
    def __init__(
        self,
        split_text: bool = True,
        text_batch_size: int = 2000,
        chunk_overlap: int = 200,
    ):
        self.split_text = split_text
        self.text_batch_size = text_batch_size
        self.chunk_overlap = chunk_overlap

        if self.split_text and RecursiveCharacterTextSplitter is None:
            raise ImportError(
                "langchain-text-splitters is required for split processor. "
                "Install with: pip install langchain-text-splitters"
            )

        if self.text_batch_size <= 0:
            raise ValueError("text_batch_size must be a positive integer")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")

        if self.chunk_overlap >= self.text_batch_size:
            raise ValueError("chunk_overlap must be smaller than text_batch_size")

    @staticmethod
    def _preprocess_text(text: str) -> str:
        lines = text.split("\n")
        # Preserve legacy filtering before splitting.
        lines = [line for line in lines if not line.strip().startswith("#")]
        lines = [line for line in lines if line.strip() != ""]
        return "\n".join(lines)

    def process(self, source_id: str, text: str) -> list[tuple[str, str]]:
        if not self.split_text:
            return [("0", text)]

        cleaned_text = self._preprocess_text(text)
        if not cleaned_text.strip():
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.text_batch_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        text_batches = splitter.split_text(cleaned_text)
        text_batches = [batch.strip() for batch in text_batches if batch and batch.strip()]
        return [(str(i), batch) for i, batch in enumerate(text_batches)]


class NoopProcessor(TextProcessor):
    def process(self, source_id: str, text: str) -> list[tuple[str, str]]:
        return [("0", text)]


class ParagraphProcessor(TextProcessor):
    """Split text into paragraph chunks separated by blank lines."""

    def process(self, source_id: str, text: str) -> list[tuple[str, str]]:
        paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n+", text) if paragraph.strip()]
        return [(str(i), paragraph) for i, paragraph in enumerate(paragraphs)]


class SentenceWindowProcessor(TextProcessor):
    """Split text into overlapping windows of sentences."""

    def __init__(self, sentence_count: int = 5, sentence_overlap: int = 1):
        if sentence_count <= 0:
            raise ValueError("sentence_count must be a positive integer")
        if sentence_overlap < 0:
            raise ValueError("sentence_overlap must be >= 0")
        if sentence_overlap >= sentence_count:
            raise ValueError("sentence_overlap must be smaller than sentence_count")

        self.sentence_count = sentence_count
        self.sentence_overlap = sentence_overlap

    def process(self, source_id: str, text: str) -> list[tuple[str, str]]:
        sentences = _split_sentences(text)
        if not sentences:
            return []

        chunks: list[tuple[str, str]] = []
        step = self.sentence_count - self.sentence_overlap
        for start in range(0, len(sentences), step):
            window = sentences[start : start + self.sentence_count]
            if not window:
                continue
            chunks.append((str(len(chunks)), " ".join(window)))
            if start + self.sentence_count >= len(sentences):
                break
        return chunks


class TokenWindowProcessor(TextProcessor):
    """Split text into fixed-size overlapping token windows."""

    def __init__(self, model_id: str, token_count: int = 512, token_overlap: int = 50):
        if token_count <= 0:
            raise ValueError("token_count must be a positive integer")
        if token_overlap < 0:
            raise ValueError("token_overlap must be >= 0")
        if token_overlap >= token_count:
            raise ValueError("token_overlap must be smaller than token_count")

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for token_window processor. " "Install with: pip install transformers"
            ) from exc

        self.model_id = model_id
        self.token_count = token_count
        self.token_overlap = token_overlap
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def process(self, source_id: str, text: str) -> list[tuple[str, str]]:
        token_ids = cast(list[int], self.tokenizer.encode(text, add_special_tokens=False))
        if not token_ids:
            return []

        chunks: list[tuple[str, str]] = []
        step = self.token_count - self.token_overlap
        for start in range(0, len(token_ids), step):
            window = token_ids[start : start + self.token_count]
            if not window:
                continue
            chunk_text = cast(
                str,
                self.tokenizer.decode(
                    window,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ),
            ).strip()
            if chunk_text:
                chunks.append((str(len(chunks)), chunk_text))
            if start + self.token_count >= len(token_ids):
                break
        return chunks


def _split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    # This deliberately remains a lightweight, language-agnostic splitter.
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    return [sentence.strip() for sentence in sentences if sentence.strip()]
