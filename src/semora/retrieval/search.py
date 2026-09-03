import argparse

import faiss
import torch

from semora.embeddings.io import load_embeddings
from semora.embeddings.registry import get_embedder
from semora.text.ids import extract_chunk_path


def get_query_vector(text, embed_func):
    out = embed_func(text)  # may be (tensor, tokens) or just tensor
    if isinstance(out, tuple):  # causal-LM case
        out = out[0]
    emb = out.detach()
    emb = emb.unsqueeze(0)

    return emb.float().cpu().numpy().astype("float32")


def build_index(embeddings):
    embeddings = embeddings.numpy()
    d = embeddings.shape[1]
    # normalize stored embeddings
    faiss.normalize_L2(embeddings)

    # flat inner-product index
    index = faiss.IndexFlatIP(d)

    # # or HNSW with inner product (M can be larger than 3, e.g. 32)
    # index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    index.add(embeddings)
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to a .pt file or folder with batch .pt files.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="openai/gpt-oss-20b",
        help="Model to embed user queries.",
    )
    parser.add_argument(
        "--model-kind",
        type=str,
        choices=["causal", "sentence"],
        default=None,
        help="Specify model kind: 'causal' or 'sentence'. If omitted, the script will try to infer from the model id.",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    file_paths, emb_matrix = load_embeddings(args.embeddings)
    print(f"Loaded {len(file_paths)} embeddings, shape={emb_matrix.shape}")
    print("Building FAISS index...")
    index = build_index(emb_matrix)
    print("Index ready.")

    # Create and load an embedder using the new interface
    embedder = get_embedder(args.model_id, kind=args.model_kind)
    embedder.load()
    def embed_func(text: str) -> torch.Tensor:
        return embedder.embed_query(text).to(device).float()

    print("\nEnter a prompt (or 'quit' to exit).")
    while True:
        query = input("> ")
        if query.lower() in {"quit", "q", "exit"}:
            break
        if not query.strip():
            continue

        q_emb = get_query_vector(query, embed_func)
        faiss.normalize_L2(q_emb)
        distances, indices = index.search(q_emb, k=10)

        print("\nTop 10 matches:")
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0], strict=True), 1):
            filepath = extract_chunk_path(file_paths[idx])
            print(f"{rank:2d}. {filepath:<140}  dist={dist:.4f}")
        print()


if __name__ == "__main__":
    main()
