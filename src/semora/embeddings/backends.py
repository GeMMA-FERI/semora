import json
import os
from typing import Any

import torch

from semora.embeddings.base import BaseEmbedder


class CausalEmbedder(BaseEmbedder):
    """Embedder for causal LM models (uses hidden states mean pooling)."""

    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.tokenizer: Any = None
        self.model: Any = None

    def load(self) -> "CausalEmbedder":
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("The transformers library is not installed. pip install transformers") from exc

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype="auto", device_map="auto")
        return self

    def _embed(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]  # [1, seq_len, dim]
        embedding = last_hidden_state.mean(dim=1).squeeze()  # [dim]
        return embedding.cpu()

    def embed_query(self, text: str) -> torch.Tensor:
        return self._embed(text)

    def embed_sentence(self, text: str) -> torch.Tensor:
        return self._embed(text)


class SentenceEmbedder(BaseEmbedder):
    """Embedder for `sentence_transformers` style models.

    Uses `encode_query` / `encode_document` when available and falls back to
    `encode(..., convert_to_tensor=True)` if needed.
    """

    def __init__(self, model_id: str, transformer_kwargs: dict | None = None):
        super().__init__(model_id)
        self.model: Any = None
        self.is_e5 = "e5" in model_id.lower()
        self.transformer_kwargs = transformer_kwargs or {}

    def load(self) -> "SentenceEmbedder":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("sentence_transformers not installed. pip install sentence-transformers") from exc

        self.model = SentenceTransformer(self.model_id, **self.transformer_kwargs)
        return self

    def embed_query(self, text: str) -> torch.Tensor:
        if hasattr(self.model, "encode_query"):
            emb = self.model.encode_query(text, convert_to_tensor=True)
        else:
            if self.is_e5:
                text = "query: " + text
            emb = self.model.encode(text, convert_to_tensor=True, show_progress_bar=False)
        if not torch.is_tensor(emb):
            raise TypeError(f"Expected torch.Tensor embedding output, got {type(emb).__name__}")
        return emb.detach().cpu().reshape(-1)

    def embed_sentence(self, text: str) -> torch.Tensor:
        if hasattr(self.model, "encode_document"):
            emb = self.model.encode_document(text, convert_to_tensor=True)
        else:
            if self.is_e5:
                text = "passage: " + text
            emb = self.model.encode(text, convert_to_tensor=True, show_progress_bar=False)
        if not torch.is_tensor(emb):
            raise TypeError(f"Expected torch.Tensor embedding output, got {type(emb).__name__}")
        return emb.detach().cpu().reshape(-1)

    def embed_documents(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, 0))
        if hasattr(self.model, "encode_document"):
            emb = self.model.encode_document(texts, convert_to_tensor=True)
        else:
            if self.is_e5:
                texts = ["passage: " + text for text in texts]
            emb = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        if not torch.is_tensor(emb):
            raise TypeError(f"Expected torch.Tensor embedding output, got {type(emb).__name__}")
        tensor = emb.detach().cpu()
        if tensor.dim() == 1:
            return tensor.reshape(1, -1)
        if tensor.dim() > 2:
            return tensor.reshape(tensor.shape[0], -1)
        return tensor


class NvidiaLlamaEmbedder(SentenceEmbedder):
    """Specialized embedder for `nvidia/llama-embed-nemotron-8b`."""

    def load(self) -> "NvidiaLlamaEmbedder":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("sentence_transformers not installed. pip install sentence-transformers") from exc

        attn_implementation = "eager"  # Or "flash_attention_2"
        self.model = SentenceTransformer(  # type: ignore[call-arg]
            self.model_id,
            trust_remote_code=True,
            model_kwargs={
                "attn_implementation": attn_implementation,
                "torch_dtype": "float16",
            },
            tokenizer_kwargs={"padding_side": "left"},
        )
        return self

    def embed_query(self, text: str) -> torch.Tensor:
        instruction = "Given a question, retrieve passages that answer the question"
        return super().embed_sentence(f"Instruct: {instruction}\nQuery: {text}")


class HFAutoEmbedder(BaseEmbedder):
    """Generic HF AutoModel embedder with mean-pooling and PEFT/LoRA adapter support.

    This supports using a local PEFT adapter directory (contains `adapter_config.json`)
    or a standard HF model id. When an adapter dir is given the adapter's
    `adapter_config.json` is inspected to find `base_model_name_or_path` and the
    base model is loaded and wrapped with the adapter.
    """

    def __init__(self, model_id: str, is_e5: bool = True):
        super().__init__(model_id)
        self.model: Any = None
        self.tokenizer: Any = None
        self.is_e5 = is_e5  # enable automatic query:/passage: prefixing

    def load(self):
        from transformers import AutoModel, AutoTokenizer

        peft_cfg = os.path.join(self.model_id, "adapter_config.json")

        # --- PEFT adapter loading ---
        if os.path.isdir(self.model_id) and os.path.isfile(peft_cfg):
            with open(peft_cfg) as f:
                cfg = json.load(f)
            base = cfg["base_model_name_or_path"]

            from peft import PeftModel

            base_model = AutoModel.from_pretrained(base, device_map="auto")
            self.model = PeftModel.from_pretrained(base_model, self.model_id, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(base)

            return self

        # --- normal model loading ---
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id, device_map="auto")

        return self

    def _pool(self, out, attention_mask):
        # CLS pooling if available
        if hasattr(self.model.config, "pooler_type") and self.model.config.pooler_type == "cls":
            return out.last_hidden_state[:, 0]

        # Mean pooling fallback
        mask = attention_mask.unsqueeze(-1).float()
        return (out.last_hidden_state * mask).sum(1) / mask.sum(1)

    def _embed(self, text: str) -> torch.Tensor:
        enc = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        with torch.no_grad():
            out = self.model(**enc)

        emb = self._pool(out, enc["attention_mask"])

        return emb.cpu().squeeze(0)

    def embed_query(self, text: str) -> torch.Tensor:
        if self.is_e5:
            text = "query: " + text
        return self._embed(text)

    def embed_sentence(self, text: str) -> torch.Tensor:
        if self.is_e5:
            text = "passage: " + text
        return self._embed(text)


class FlagEmbedder(BaseEmbedder):
    """Embedder for loading FlagEmbedding-based models."""

    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.model: Any = None

    def load(self) -> "BaseEmbedder":
        """Load model/tokenizer as needed and return self."""
        try:
            from FlagEmbedding import BGEM3FlagModel

            self.model = BGEM3FlagModel(self.model_id)
        except ImportError as exc:
            raise ImportError("FlagEmbedding library is not installed. pip install FlagEmbedding") from exc
        return self

    def embed_query(self, text: str) -> torch.Tensor:
        """Embed a short query string."""
        return self.embed_sentence(text)

    def embed_sentence(self, text: str) -> torch.Tensor:
        """Embed a (longer) sentence/document."""
        return torch.from_numpy(self.model.encode([text])["dense_vecs"])

    def embed_document(self, text: str) -> torch.Tensor:
        """Return the embedding for a single document.
        Alias for embed_sentence()"""
        return self.embed_sentence(text)

    def embed_documents(self, texts: list[str]) -> torch.Tensor:
        """Return embeddings for multiple documents.
        The returned tensor has shape [num_documents, embedding_dimension].
        Subclasses may override this method to implement efficient batched
        embedding."""
        return torch.from_numpy(self.model.encode(texts)["dense_vecs"])


