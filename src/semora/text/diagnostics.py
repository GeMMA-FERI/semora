"""Bounded diagnostics for tokenizers that print source text to stdout."""


class TokenAlignmentWarnings:
    """Count known warnings while discarding their potentially large OCR payloads."""

    def __init__(self) -> None:
        self.count = 0

    def write(self, message: str) -> int:
        self.count += message.count("Warning: Cannot compute token index.")
        return len(message)

    def flush(self) -> None:
        pass
