from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from semora.embeddings.serialization import tensor_from_float32blob
from semora.storage import Database, EmbeddingProjection, Run

DEFAULT_TRANSFORM_BATCH_SIZE = 10_000


class _ProjectionMLP(torch.nn.Module):
    def __init__(self, input_dimension: int) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 3),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class _TorchMLPProjector:
    def __init__(
        self,
        model: _ProjectionMLP,
        *,
        device: torch.device,
        target_mean: torch.Tensor,
        target_scale: torch.Tensor,
    ) -> None:
        self.model = model.eval()
        self.device = device
        self.target_mean = target_mean
        self.target_scale = target_scale

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        inputs = torch.from_numpy(vectors).to(self.device)
        inputs = F.normalize(inputs, p=2, dim=1)
        with torch.inference_mode():
            coordinates = self.model(inputs)
            coordinates = coordinates * self.target_scale + self.target_mean
        return coordinates.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Jointly project compatible embedding runs into 3D and store them in SQLite."
    )
    parser.add_argument(
        "--db-path",
        default="data/derived/newspapers.sqlite",
        help="SQLite database path.",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--embedding-run-ids",
        nargs="+",
        help="One or more compatible embedding runs to project together.",
    )
    selection.add_argument(
        "--model-id",
        help="Project all embedding runs whose model id exactly matches this value.",
    )
    parser.add_argument(
        "--method",
        choices=("umap", "umap-mlp", "pca"),
        default="umap",
        help="Projection algorithm (default: umap). Use umap-mlp for fast PyTorch inference.",
    )
    parser.add_argument(
        "--metric",
        default="cosine",
        help="UMAP input-space distance metric (default: cosine).",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP neighborhood size (default: 15; capped at sample count - 1).",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP minimum output-space distance (default: 0.1).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for sampling and single-threaded UMAP (default: 42).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help=(
            "UMAP CPU workers: 1 preserves deterministic output, -1 uses all cores, "
            "and values above 1 use that many cores (default: -1). Parallel UMAP is nondeterministic."
        ),
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help=(
            "Fit the projector on this many randomly selected embeddings, then "
            "transform and store coordinates for every selected embedding."
        ),
    )
    parser.add_argument(
        "--transform-workers",
        type=int,
        default=1,
        help=(
            "Number of embedding batches to transform concurrently (default: 1). "
            "SQLite writes remain on the main thread."
        ),
    )
    parser.add_argument(
        "--mlp-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for umap-mlp training and inference (default: auto).",
    )
    parser.add_argument(
        "--mlp-epochs",
        type=int,
        default=30,
        help="Maximum umap-mlp training epochs (default: 30).",
    )
    parser.add_argument(
        "--mlp-batch-size",
        type=int,
        default=2048,
        help="umap-mlp training batch size (default: 2048).",
    )
    parser.add_argument(
        "--mlp-learning-rate",
        type=float,
        default=0.001,
        help="umap-mlp Adam learning rate (default: 0.001).",
    )
    parser.add_argument(
        "--mlp-patience",
        type=int,
        default=5,
        help="umap-mlp validation-loss early-stopping patience (default: 5).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional pipeline run id. Defaults to a timestamped id.",
    )
    args = parser.parse_args()
    _validate_args(args)

    db = Database(args.db_path)
    try:
        db.initialize()
        run_rows = _resolve_embedding_runs(db, args)
        embedding_run_ids = [row["embedding_run_id"] for row in run_rows]
        embedding_count = db.count_embeddings_for_projection(embedding_run_ids)
        if not embedding_count:
            raise ValueError("No embeddings found for the selected embedding runs")

        print("Sampling embeddings to fit the projector...")
        sample_count = min(args.n_samples or embedding_count, embedding_count)
        sample_rows = _reservoir_sample(
            tqdm(
                db.iter_embeddings_for_projection(embedding_run_ids),
                total=embedding_count,
                desc="Sampling embeddings",
                unit="embedding",
            ),
            sample_count,
            random_state=args.random_state,
        )
        sample_matrix = _embedding_matrix_from_rows(sample_rows)
        del sample_rows
        projector = _fit_projector(sample_matrix.numpy(), args)
        del sample_matrix
        print("Fitted projector, transforming and storing coordinates for all embeddings...")

        db.prepare_embedding_projection_staging()
        projected_count = 0
        cursor = db.iter_embeddings_for_projection(embedding_run_ids)
        with tqdm(
            total=embedding_count,
            desc="Transforming embeddings",
            unit="embedding",
        ) as progress:
            for projections in _transform_batches(
                cursor,
                projector,
                worker_count=args.transform_workers,
            ):
                db.stage_embedding_projections(projections)
                projected_count += len(projections)
                progress.update(len(projections))

        db.apply_staged_embedding_projections(expected_count=embedding_count)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = args.run_id or f"projection_{args.method}_{timestamp}"
        db.insert_run(Run(run_id=run_id, run_type="projection"))
        db.log(run_id, "INFO", f"Projection config: {json.dumps(vars(args), sort_keys=True)}")
        db.log(
            run_id,
            "INFO",
            f"Jointly projected {projected_count} embeddings from "
            f"{len(embedding_run_ids)} embedding runs with {args.method}, "
            f"fitted on {sample_count} samples.",
        )
        print(
            f"Stored {projected_count} {args.method.upper()} coordinates for "
            f"{len(embedding_run_ids)} embedding runs in {args.db_path} "
            f"(fitted on {sample_count} samples)"
        )
    finally:
        db.close()


def _validate_args(args: argparse.Namespace) -> None:
    if args.n_neighbors < 2:
        raise ValueError("n-neighbors must be at least 2")
    if not 0.0 <= args.min_dist:
        raise ValueError("min-dist must be non-negative")
    if args.n_samples is not None and args.n_samples < 3:
        raise ValueError("n-samples must be at least 3")
    if args.n_jobs == 0 or args.n_jobs < -1:
        raise ValueError("n-jobs must be -1 or a positive integer")
    if args.transform_workers < 1:
        raise ValueError("transform-workers must be a positive integer")
    if args.mlp_epochs < 1:
        raise ValueError("mlp-epochs must be a positive integer")
    if args.mlp_batch_size < 1:
        raise ValueError("mlp-batch-size must be a positive integer")
    if args.mlp_learning_rate <= 0:
        raise ValueError("mlp-learning-rate must be positive")
    if args.mlp_patience < 1:
        raise ValueError("mlp-patience must be a positive integer")
    if args.method == "umap-mlp" and args.transform_workers != 1:
        raise ValueError(
            "umap-mlp performs batched PyTorch inference internally; "
            "use --transform-workers 1"
        )
    if args.transform_workers > 1 and args.n_jobs != 1:
        print(
            "Warning: both transform-workers and UMAP n-jobs enable parallelism; "
            "their product can oversubscribe the available CPU cores."
        )


def _resolve_embedding_runs(db: Database, args: argparse.Namespace):
    if args.model_id:
        run_rows = db.get_embedding_runs_by_model(args.model_id)
        if not run_rows:
            raise ValueError(f"No embedding runs found for model: {args.model_id}")
        return run_rows

    requested_ids = list(dict.fromkeys(args.embedding_run_ids))
    run_rows = db.get_embedding_runs_by_ids(requested_ids)
    found_ids = {row["embedding_run_id"] for row in run_rows}
    missing_ids = [run_id for run_id in requested_ids if run_id not in found_ids]
    if missing_ids:
        raise ValueError(f"Embedding runs not found: {', '.join(missing_ids)}")

    model_ids = {row["model_id"] for row in run_rows}
    if len(model_ids) != 1:
        raise ValueError(
            "Selected embedding runs use different models and cannot share a projection: "
            + ", ".join(sorted(model_ids))
        )
    return run_rows


def _embedding_matrix_from_rows(rows) -> torch.Tensor:
    if not rows:
        raise ValueError("Cannot build an embedding matrix from no rows")

    first = tensor_from_float32blob(rows[0]["tensor_blob"])
    if first.numel() == 0:
        raise ValueError("Encountered an empty embedding tensor")
    expected_dimension = first.numel()
    matrix = torch.empty((len(rows), expected_dimension), dtype=torch.float32)
    matrix[0].copy_(first)
    for index, row in enumerate(rows[1:], start=1):
        tensor = tensor_from_float32blob(row["tensor_blob"])
        if tensor.numel() != expected_dimension:
            raise ValueError(
                "Embedding dimensions are inconsistent: "
                f"expected {expected_dimension}, found {tensor.numel()}"
            )
        matrix[index].copy_(tensor)

    if not torch.isfinite(matrix).all():
        raise ValueError("Embeddings contain NaN or infinite values")
    return matrix


def _fit_projector(vectors: np.ndarray, args: argparse.Namespace):
    if min(vectors.shape) < 3:
        raise ValueError(
            "At least three embeddings and three input dimensions are required for a 3D projection"
        )

    if args.method == "pca":
        try:
            from sklearn.decomposition import PCA
        except ImportError as exc:
            raise RuntimeError("PCA projection requires scikit-learn") from exc
        return PCA(n_components=3).fit(vectors)

    reducer = _fit_umap(vectors, args)
    if args.method == "umap-mlp":
        return _fit_torch_mlp(vectors, np.asarray(reducer.embedding_), args)
    return reducer


def _fit_umap(vectors: np.ndarray, args: argparse.Namespace):

    try:
        import umap
    except ImportError as exc:
        raise RuntimeError(
            "UMAP projection requires the optional 'umap-learn' package; "
            "install it with: pip install umap-learn"
        ) from exc

    if args.metric == "cosine" and np.any(np.einsum("ij,ij->i", vectors, vectors) == 0):
        raise ValueError("Cosine UMAP cannot project zero-length embedding vectors")

    n_neighbors = min(args.n_neighbors, len(vectors) - 1)
    umap_random_state = args.random_state if args.n_jobs == 1 else None
    if args.n_jobs != 1:
        print(
            f"Parallel UMAP enabled with n_jobs={args.n_jobs}; "
            "the fitted projection will not be exactly reproducible."
        )
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=umap_random_state,
        n_jobs=args.n_jobs,
        init="random",
    )
    return reducer.fit(vectors)


def _fit_torch_mlp(
    vectors: np.ndarray,
    target_coordinates: np.ndarray,
    args: argparse.Namespace,
) -> _TorchMLPProjector:
    device = _resolve_mlp_device(args.mlp_device)
    print(f"Training UMAP projection MLP on {device}")
    torch.manual_seed(args.random_state)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.random_state)

    inputs = F.normalize(torch.from_numpy(vectors), p=2, dim=1)
    targets = torch.from_numpy(target_coordinates.astype(np.float32, copy=False))
    target_mean = targets.mean(dim=0)
    target_scale = targets.std(dim=0).clamp_min(1e-6)
    targets = (targets - target_mean) / target_scale

    generator = torch.Generator().manual_seed(args.random_state)
    indices = torch.randperm(len(inputs), generator=generator)
    validation_count = max(1, int(len(inputs) * 0.1))
    if validation_count >= len(inputs):
        validation_count = 1
    validation_indices = indices[:validation_count]
    training_indices = indices[validation_count:]

    model = _ProjectionMLP(inputs.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.mlp_learning_rate)
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0

    for epoch in range(args.mlp_epochs):
        model.train()
        shuffled = training_indices[
            torch.randperm(len(training_indices), generator=generator)
        ]
        for start in range(0, len(shuffled), args.mlp_batch_size):
            batch_indices = shuffled[start : start + args.mlp_batch_size]
            batch_inputs = inputs[batch_indices].to(device)
            batch_targets = targets[batch_indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(batch_inputs), batch_targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            validation_loss = 0.0
            validation_items = 0
            for start in range(0, len(validation_indices), args.mlp_batch_size):
                batch_indices = validation_indices[start : start + args.mlp_batch_size]
                batch_inputs = inputs[batch_indices].to(device)
                batch_targets = targets[batch_indices].to(device)
                batch_loss = F.mse_loss(model(batch_inputs), batch_targets, reduction="sum")
                validation_loss += float(batch_loss.item())
                validation_items += batch_targets.numel()
        validation_loss /= validation_items
        print(
            f"MLP epoch {epoch + 1}/{args.mlp_epochs}: "
            f"standardized_validation_mse={validation_loss:.6f}"
        )

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.mlp_patience:
                print(f"MLP early stopping after {epoch + 1} epochs")
                break

    if best_state is None:
        raise RuntimeError("MLP training did not produce a model state")
    model.load_state_dict(best_state)
    model.to(device)
    return _TorchMLPProjector(
        model,
        device=device,
        target_mean=target_mean.to(device),
        target_scale=target_scale.to(device),
    )


def _resolve_mlp_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested for umap-mlp but is not available")
    return torch.device(value)


def _transform_batch(projector, batch) -> list[EmbeddingProjection]:
    matrix = _embedding_matrix_from_rows(batch)
    coordinates = projector.transform(matrix.numpy())
    _validate_coordinates(coordinates, len(batch))
    return [
        EmbeddingProjection(
            embedding_id=row["embedding_id"],
            x=float(point[0]),
            y=float(point[1]),
            z=float(point[2]),
        )
        for row, point in zip(batch, coordinates, strict=True)
    ]


def _transform_batches(cursor, projector, *, worker_count: int):
    """Transform bounded batches concurrently without writing SQLite from workers."""
    if worker_count == 1:
        while batch := cursor.fetchmany(DEFAULT_TRANSFORM_BATCH_SIZE):
            yield _transform_batch(projector, batch)
        return

    max_pending = worker_count * 2
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="embedding-transform",
    ) as executor:
        pending: set[Future[list[EmbeddingProjection]]] = set()
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < max_pending:
                batch = cursor.fetchmany(DEFAULT_TRANSFORM_BATCH_SIZE)
                if not batch:
                    exhausted = True
                    break
                pending.add(executor.submit(_transform_batch, projector, batch))

            if pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    yield future.result()


def _reservoir_sample(rows, sample_count: int, *, random_state: int) -> list:
    """Select a reproducible uniform sample without loading every blob into memory."""
    generator = random.Random(random_state)
    sample = []
    for index, row in enumerate(rows):
        if index < sample_count:
            sample.append(row)
            continue
        replacement_index = generator.randrange(index + 1)
        if replacement_index < sample_count:
            sample[replacement_index] = row
    return sample


def _validate_coordinates(coordinates: np.ndarray, expected_count: int) -> None:
    if coordinates.shape != (expected_count, 3):
        raise ValueError(f"Projection returned unexpected shape: {coordinates.shape}")
    if not np.isfinite(coordinates).all():
        raise ValueError("Projection produced NaN or infinite coordinates")


if __name__ == "__main__":
    main()
