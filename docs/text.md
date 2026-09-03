# Text chunking

Chunkers implement the `TextProcessor` interface and return `(suffix, text)`
pairs. Available implementations include no-op, paragraph, sentence-window,
token-window, and recursive character splitting.

```python
from semora.text import ParagraphProcessor, remove_markdown_images

text = remove_markdown_images(markdown)
chunks = ParagraphProcessor().process("document-1", text)
```

Token-window chunking requires the `chunking` optional dependency group.
