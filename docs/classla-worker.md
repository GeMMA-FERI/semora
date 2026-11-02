# Optional CLASSLA NER worker

The worker serves NER inference without access to the corpus or database. It
loads the existing CLASSLA adapter once, warms it up, then accepts authenticated,
bounded v1 batches. The additive NER pipeline can distribute requests across
these workers; see [execution backends](execution-backends.md) for coordinator
configuration and compatibility rules. Lemmatization remains a later milestone.
Ordinary local processing is unchanged and does not import this service.

## Run without Docker

Install Semora's `classla-worker` extra and the tested CLASSLA optimization fork
at commit `61d2981e36c35585ba4e828b9fda373fe6606599`. The extra isolates API/model
dependencies from the base package and does not install retrieval/FAISS models.
An ordinary PyPI CLASSLA installation is not evidence that this fork is installed.
Keep your chosen environment's resolved dependencies for reproducibility.

Prepare CLASSLA Slovene tokenize/NER resources separately, including
`resources.json`; serving does not download or update models. Create a private
token file containing at least 32 non-whitespace characters, using a securely
generated token. Pass its path, not its contents, on the command line:

```sh
semora-classla-worker --resources /path/to/classla_resources --token-file /path/to/token
```

The default bind address is `127.0.0.1:8000` and the default device is CPU.
`--device cuda` fails startup if CUDA is unavailable; it does not fall back to CPU.
`--decoder numba` requires Numba in that environment. Use `--help` for request
limits and inference tuning. Start one process per assigned device; the provided
launcher fixes the server process count at one.

## Build and run the CPU container

From the Semora repository:

```sh
docker build --platform linux/amd64 -f docker/classla-worker/Dockerfile -t semora-classla-worker:dev-cpu .
docker run --rm -p 127.0.0.1:18081:8000 \
  --mount type=bind,source=/absolute/classla_resources,target=/models,readonly \
  --mount type=bind,source=/absolute/token,target=/run/secrets/worker-token,readonly \
  --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges \
  semora-classla-worker:dev-cpu
```

The recipe targets Linux amd64, pins the Python base image digest and verifies
the immutable CLASSLA source archive checksum. The runtime lock pins all
dependencies with artifact hashes, including Python 3.11-compatible CPU PyTorch
2.6.0 and NumPy 2.2.6. Build tools are pinned in the builder stage. The
Dockerfile-specific context allowlist keeps local models, archives, virtual
environments and secrets out of the build. Neither models nor tokens are baked
into the image. The final image runs as UID/GID 10001; mounted files must be
readable by that user. Preserve upstream CLASSLA/package notices. Verify model
redistribution permissions separately before bundling or publishing resources.

This recipe is CPU-only. A separate pinned [CUDA image and GPU deployment](classla-worker-gpu.md)
are also available and have passed cross-machine synthetic qualification. These local
development images have not been published to a registry. Record its image ID or
registry digest when distributing a tested release.

For transfer without a registry, export the image with
`docker save -o classla-worker-cpu.tar semora-classla-worker:dev-cpu`, copy the
archive to the other computer, then run `docker load -i classla-worker-cpu.tar`
there. Supply the verified model directory and a private token file separately.
The image's standalone runtime needs neither a Semora checkout nor a corpus mount.

NORA provides a standalone `compose.classla-worker.yaml`. Set
`CLASSLA_RESOURCES_DIR` and `CLASSLA_WORKER_TOKEN_FILE` to existing absolute paths,
then run `docker compose -f compose.classla-worker.yaml up -d` from NORA. It is
not included in NORA's normal Compose deployment. It defaults to host loopback
port 18081; change `CLASSLA_WORKER_BIND` only for a private/VPN interface with
appropriate access control. Use TLS termination when crossing untrusted networks.
Models/token mount read-only and the service receives no corpus mount.

## API

| Endpoint | Authentication | Behavior |
| --- | --- | --- |
| `GET /health/live` | None | Process liveness, without resource details. |
| `GET /v1/capabilities` | Bearer token | Readiness, task identity/fingerprint, limits, tuning, capacity and inference counters. |
| `POST /v1/ner` | Bearer token | Strict `Request` JSON; returns `Response` JSON from the v1 contracts. |
| `POST /v1/lemma` | Bearer token | Explicit 501 unsupported-task response. |

Use `Authorization: Bearer <token>` and `Content-Type: application/json`.
Obtain the NER fingerprint from capabilities and construct requests with
`semora.execution.contracts.Document.create` and `Request`. A request's task
must be `ner`, with exactly the worker's advertised fingerprint. Consume results
with `Response.from_dict(...).validate_against(request)`.

One model-owning thread performs load, warmup, inference and close. No inference
queue is maintained. A concurrent request returns 503/busy with `Retry-After`;
health/capabilities remain responsive. Cancelling a caller does not make model
capacity available before its inference actually finishes. Request IDs correlate
responses; the worker does not cache results or deduplicate repeated requests.
It never writes article-completion records.

Default limits are 32 documents, 1 MiB of UTF-8 input text, 2 MiB of raw request
JSON, 4 MiB of response JSON and 30 seconds to receive a request body. The launcher
also bounds concurrent connections/body readers. The worker rejects oversized
inputs rather than splitting text or changing batch composition. Empty batches,
empty texts and zero-entity results are valid.

Errors use `{"error":{"code":"...","message":"..."}}`: 400 invalid contract,
401 unauthorized, 408 body timeout, 409 incompatible fingerprint, 413 request/
input/response size limit, 415 unsupported media type, 500 inference failure,
501 unsupported task, and 503 busy/not-ready. Inference errors omit source text
and traceback details. No partial result batch is returned.

## Identity and reproducibility

The versioned NER identity hashes CLASSLA's package source/data tree, the separate
Obeliks package and tokenizer rules, Semora's adapter/alignment and entity code,
and resource files referenced by loaded processor configurations. Package paths
are replaced by logical names; Python CRLF/LF differences are normalized. Missing
required files, unrecognized resource references and symlinks fail startup.
Disabled optional components retain explicit nulls. Large model files are hashed
incrementally. Device/batching tuning is advertised separately.

After choosing and verifying resources, pass `--expected-fingerprint <digest>`
to reject an unintended resource/code change at startup. A fingerprint computed
now is not proof of the exact historical environment and does not automatically
permit appending to an adopted run. Changing hardware, runtime or batch
composition may change near-tied neural predictions; qualification must compare
fixed batches and record those settings. Keep model mounts immutable throughout
the service lifetime.

The worker follows FastAPI's [lifespan model](https://fastapi.tiangolo.com/advanced/events/)
for startup/shutdown. This isolates model loading from module import and test
collection. API/service tests use an injected model; real-model checks are
separate from the default dependency-light suite. CI runs the worker API tests
against the built wheel with API dependencies installed and CLASSLA/Torch absent.
To run the opt-in CPU parity test, set `SEMORA_TEST_CLASSLA_WORKER=1` and
`SEMORA_CLASSLA_RESOURCES` to the existing model directory, then run
`pytest tests/test_execution_worker_model.py`. It uses synthetic inputs, fixed
batches, controls/soft hyphens and an offset-preserving HTML table mask.
