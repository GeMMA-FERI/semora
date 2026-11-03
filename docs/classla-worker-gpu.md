# Optional CUDA CLASSLA worker

The CUDA image runs the same NER worker API with `--device cuda`. Local Semora
and the CPU worker remain independent. CUDA startup fails clearly when GPU access
is unavailable; it never silently selects CPU.

## Host and image

Use Linux amd64 containers with an NVIDIA GPU. Docker Desktop on Windows needs
the WSL 2 backend and a compatible Windows NVIDIA driver. Test passthrough before
loading models:

```console
docker run --rm --gpus all --entrypoint nvidia-smi semora-classla-worker:dev-cpu
```

This uses an already-installed CPU worker image only to inspect the injected GPU
driver tools; it does not make that image CUDA-capable.

Build the separate CUDA variant from the Semora checkout:

```console
docker build --platform linux/amd64 -f docker/classla-worker/Dockerfile.cuda -t semora-classla-worker:dev-cuda .
```

The recipe uses the same immutable Python base and CLASSLA fork archive as the
original qualified CPU worker. Its separate hash-locked requirements select
PyTorch 2.6.0+cu124 and CUDA 12.4 runtime libraries, retaining the common package
versions from that qualified CPU image. The current CPU recipe/lock can evolve
independently. A GPU does not require installing a CUDA toolkit into the Windows
host; the image supplies its userspace libraries and Docker supplies driver access.

Source references: [Docker Desktop GPU support](https://docs.docker.com/desktop/features/gpu/),
[Compose GPU reservations](https://docs.docker.com/compose/how-tos/gpu-support/),
[official PyTorch 2.6 installation variants](https://docs.pytorch.org/get-started/previous-versions/).

## Standalone deployment

NORA supplies `compose.classla-worker-gpu.yaml`, separate from both ordinary NORA
and the CPU worker. Set `CLASSLA_RESOURCES_DIR` and `CLASSLA_WORKER_TOKEN_FILE` to
existing absolute paths. The same private token can authenticate a CPU/GPU pool.
Then run:

```console
docker compose -f compose.classla-worker-gpu.yaml up -d
```

It reserves one NVIDIA GPU, mounts models/token read-only and defaults to remote
loopback port 18082. The Compose command selects CUDA and defaults to sentence batch size 64;
`CLASSLA_GPU_BATCH_SIZE` overrides this startup setting. The image alone defaults
to sentence batch size 16. The optional
`CLASSLA_GPU_WORKER_IMAGE`, `CLASSLA_GPU_WORKER_BIND` and
`CLASSLA_GPU_WORKER_PORT` variables configure only this service; CPU image/port
variables do not override it. `CLASSLA_GPU_CPU_THREADS` defaults to 2 for CPU-side
work. Prefer SSH forwarding or a private/VPN interface with appropriate transport
protection. No corpus directory is mounted.

For a second-computer SSH deployment, forwarding coordinator port 19082 to that
computer's `127.0.0.1:18082` exposes `http://127.0.0.1:19082`. Existing bearer-token
configuration and the remote NER client work without a new API. Confirm readiness,
`runtime.device == "cuda"`, fingerprint and real inference before scheduling work.
Keep a working CPU deployment available while qualifying GPU operation.

## Validation and limits

The model's neural inference uses CUDA. Tokenization, alignment and the NumPy CRF
decoder still use CPU; GPU presence alone does not establish a speedup. Measure
end-to-end fixed batches, transfer overhead and memory on representative OCR.
Synthetic comparison is correctness evidence, not a general throughput claim.

Device and runtime versions are provenance, outside semantic identity. CPU and
GPU workers may advertise matching semantic fingerprints yet differ numerically
on near-tied outputs. Compare fixed batches and preserve the execution reports;
do not automatically append a new device profile to an unqualified historical run.
Oversized documents and CUDA out-of-memory errors fail explicitly. Select suitable
worker batch size/limits before long runs; the 8 GB reference card does not qualify
all possible document lengths or concurrency settings.

The existing opt-in real-model test now accepts CPU and CUDA cases. Set
`SEMORA_TEST_CLASSLA_WORKER=1`, `SEMORA_CLASSLA_RESOURCES` to existing cached
resources, and additionally `SEMORA_TEST_CLASSLA_CUDA=1` on a CUDA-capable test
environment. Its CUDA case compares local-GPU and API-GPU results; separate
cross-device checks qualify CPU/GPU parity. Ordinary CI skips model/GPU tests.

Rollback stops only this standalone GPU project:

```console
docker compose -f compose.classla-worker-gpu.yaml stop
```

Continue using the CPU endpoint and keep the same documented identity/new-run
rules. Images are local artifacts until a registry publication is separately
requested. Remote lemma remains a later capability.

## Recorded RTX 4060 qualification

On 2026-11-02 the image built and ran on a second Windows/Docker Desktop WSL 2
machine with an RTX 4060 (8 GB), driver 610.88. Image ID:
`sha256:9c08d7c9a53f5af74bda0a04e691dc2e9fad66330742c21ab27bb7a6f6acabe9`.
Its readiness reported CUDA and the same task fingerprint as the qualified CPU
reference. The standalone GPU worker remains running alongside the CPU worker.

Eight synthetic articles produced 11 mentions with exact persisted-output parity
against the pinned CPU reference. Coordinator interruption after one commit
resumed with one skipped and seven added. A separate fixed 16-document, 39,680-byte
batch returned identical CPU/GPU entities (1,280 mentions) across four calls per
device. Medians of the last three round trips were 0.526 s CPU and 0.431 s GPU
(about 1.22x), including SSH transport and CPU tokenization/decoding. These small
synthetic measurements do not establish representative OCR throughput or peak
VRAM capacity. No production corpus was processed.

## Follow-up GPU placement and stage profiling

A follow-up check verified Docker's NVIDIA device reservation, PyTorch
`2.6.0+cu124`, CUDA runtime 12.4 and `torch.cuda.is_available() == True` on the RTX
4060. All 27 NER parameter tensors (11,418,228 elements), observed model-forward
inputs and outputs were on `cuda:0`. CUDA stream events around 20 forward calls
recorded positive elapsed time (44.9 ms summed in the verification run); a CUDA
matrix multiply also returned a CUDA tensor. CUPTI/PyTorch-profiler device events
were unavailable in this environment, so no pure kernel-busy-time claim is made.

Separate pre-tokenized stage profiling ran CPU and CUDA modes from the same GPU
image, both with two CPU threads, using 16 synthetic documents/1,280 short
sentences/8,000 tokens. Medians of three warmed `predict_prepared` calls:

| Sentence batch size | CPU NER stage | CUDA NER stage | Stage speedup |
| --- | ---: | ---: | ---: |
| 16 | 283 ms | 166 ms | 1.71x |
| 64 | 216 ms | 74 ms | 2.91x |
| 256 | 224 ms | 111 ms | 2.02x |

The NER stage includes batching, neural forward, device copies and NumPy decoding;
it excludes tokenization and HTTP/SSH. Median tokenization was approximately
175–179 ms, dominating much of the accelerated path. The earlier 1.22x number
measures the full API round trip, not isolated GPU compute. These microbenchmarks
are synthetic and batch-order/length dependent; they do not qualify a production
batch-size change. At profiling time the worker used sentence batch size 16. Mitko subsequently
selected 64; the GPU worker was recreated with 64, and the eight-article
parity/restart check passed again. The CPU worker was stopped.

## Batch configuration boundary

The current v1 request schema has no per-request `batch_size` field; an added
field is rejected with HTTP 400. Configure sentence batching with the worker's
`--batch-size` launch argument or the GPU Compose `CLASSLA_GPU_BATCH_SIZE`
variable, then recreate the service. Authenticated capabilities report the actual
value under `runtime.batch_size`. This is distinct from `limits.max_documents`,
which remains 32 documents per API request by default.


## Measured CPU preparation overlap

For the 2026-11-03 OCR experiment, the selected profile is sentence batch64,
four tokenizer processes, NumPy decoding, up to128 documents per HTTP request,
and `--prepare-batch-documents 32`. The model-owner thread infers one32-document
chunk while one preparation thread tokenizes the next. At most two prepared
chunks are resident and one HTTP inference slot remains in use. This is separate
from `--ner-pipeline-depth`, which stays1 in the selected profile. The default
preparation chunk is0 (disabled), preserving ordinary operation.

Warm repeated1,040-article tests with production sharing the test GPUs gave
about10.2 versus4.6 seconds on the RTX4090 host, and10.6 versus5.7 seconds on the
RTX4060 host. All compared outputs matched exactly. Eight tokenizer processes,
Numba0.61.2 decoding and additional tensor-stage depth did not provide a useful
consistent gain, so the selected profile keeps four processes and NumPy. No
additional JIT dependency is required. Parallel tokenizer diagnostics are
suppressed in child processes and their warning counts returned to the owner.

NORA's optional GPU Compose exposes `CLASSLA_GPU_TOKENIZER_WORKERS`,
`CLASSLA_GPU_DECODER`, `CLASSLA_GPU_PIPELINE_DEPTH`, `CLASSLA_GPU_MAX_DOCUMENTS`
and `CLASSLA_GPU_PREPARE_DOCUMENTS`; sentence batch remains64. Increase the
coordinator's bounded page to256 for two128-document workers. Request input and
response byte bounds still apply. The coordinator reuses each available worker
and sizes compact JSON batches in linear time while retaining exact byte limits.

Qualify outputs and memory on the target corpus before selecting this profile.
Stop both old workers and wait for the coordinator's durable paused checkpoint
before replacing either production endpoint. Record the new fingerprint/runtime
qualification, then resume with trusted checkpoint counts. Do not hot-swap a
worker underneath an active coordinator. Keep the prior image/configuration
for rollback; failed candidates must not update the entity dictionary or publish.
