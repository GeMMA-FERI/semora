# Initial NER comparison: 14–15 September 2026

This is a practical, unannotated comparison on the local newspaper archive, not
an accuracy benchmark. The source database was opened read-only. All generated
samples, predictions and reports are under `artifacts/ner-benchmark/` and ignored
by Git. See [the commands and adapter guide](ner-benchmark.md) to reproduce it.

## Workload and environment

- 10,000 uniformly sampled valid, nonempty articles; seed 42.
- 12,927,278 characters from `articles.content`; titles were not prepended.
- Sample fingerprint: `c98d779ac2f59f726229223dd7901f09d4b038f372b9eb38283b0763e5a78976`.
- Dates represented: 1860s–2020s; 1,776 articles have no usable date.
- Length mix: 6,353 below 1,000 characters; 3,215 between 1,000 and 4,999;
  432 at least 5,000 characters. The sample is not balanced by period or OCR quality.
- Windows 11, RTX 4090, Python 3.14.0, PyTorch 2.14.0+cu130,
  Transformers 5.17.0, local CLASSLA fork reporting version 2.2.2.
- CLASSLA used the standard Slovenian NER 2.2 resource, with tokenize/NER only.
- XLM-R used `ivlcic/xlmr-ner-slavic-pline`, revision
  `22d0901eeadcdca674512de283211a0b2406ccbf`, FP16, 256-token windows,
  64-token stride, lossless pretokenization and `first` aggregation using word IDs.
- Both adapters used an outer read/write batch of 32 articles. Runs were sequential.

## Full-sample results

| Configuration | Articles | Seconds | Articles/s | Characters/s | Estimated hours for 6,318,092 articles |
|---|---:|---:|---:|---:|---:|
| CLASSLA, sentence batch 16 | 10,000 | 272.44 | 36.71 | 47,451 | 47.81 |
| CLASSLA, sentence batch 32 | 10,000 | 293.53 | 34.07 | 44,040 | 51.52 |
| CLASSLA, sentence batch 64 | 10,000 | 353.09 | 28.32 | 36,612 | 61.97 |
| XLM-R, window batch 64, word-level decoder | 10,000 | 377.18 | 26.51 | 34,274 | 66.20 |

The fastest observed CLASSLA configuration was batch 16. XLM-R batch 64 was
selected from a separate 2,000-article prefix sweep: sizes 16, 32 and 64 measured
18.72, 20.48 and 22.21 articles/s respectively. The full run was faster than that
short prefix run. These were single runs per setting, so no repeat standard
deviation or statistically reliable speed ranking is claimed.

For the selected full runs:

| Measurement | CLASSLA, batch 16 | XLM-R, batch 64 |
|---|---:|---:|
| Model load, cached resources | 1.12 s | 6.08 s |
| Warm-up | 0.25 s | 0.26 s |
| Peak process RSS | 1.25 GiB | 2.42 GiB |
| Peak PyTorch CUDA allocated memory | 101 MiB | 813 MiB |
| Extracted spans, all labels | 145,079 | 144,688 |
| Experimental prediction database | 35.32 MB | 38.76 MB |

CUDA allocated memory is not total device usage. Both result databases store
JSON predictions plus indexed individual mentions; these are benchmark artifacts,
not the proposed final production entity schema.

Timings include sample SQLite reads, extraction, validation, indexed writes and
commits, excluding startup/warm-up. The small sample is cache-friendly. Estimates
do not include the full production database's I/O behavior, competing GPU jobs,
interruptions or a future optimized indexing pipeline. No hour-long stability
test was performed. Treat roughly two to three days as an initial planning scale,
not a guaranteed completion time.

## Practical extraction observations

The final `review/report.html` contains 40 short passages sampled from articles
where the models disagree, plus literal-name checks. Hover highlights for labels;
the lists below each passage expose exact entity boundaries. There were 8,213
articles with at least one PER/LOC/ORG span or label disagreement. This is **not**
an error rate: articles often contain many names, label inventories differ, and
there is no manually annotated reference.

Literal-name checks found:

| Query | Literal occurrences | CLASSLA full-span coverage, any label | XLM-R full-span coverage, any label |
|---|---:|---:|---:|
| Ivan Cankar | 18 | 18 | 18 |
| Josip Broz | 1 | 1 | 1 |
| Marija Abram | 0 | 0 | 0 |

CLASSLA labelled 17 of the Ivan Cankar spans PER; XLM-R labelled nine PER and nine
ORG. Inspecting the nine ORG contexts showed schools, societies and a cooperative
named after him. Lower person-label counts therefore did not mean missed names.
These checks are literal matches, not recall; inflected forms require additional
queries, and an absent sample name is not a model failure.

Selected disagreements also exposed plausible CLASSLA false positives, such as
sentence-initial “Nato” as ORG and OCR “Liudje” as PER. XLM-R omitted those labels
in the inspected passage. Other examples involve addresses, publication titles,
historical spellings and entity boundaries. These observations justify reviewing
the short report, not declaring either model globally more accurate.

## Corrections discovered during the comparison

1. Obeliks offsets restart at paragraph boundaries. The CLASSLA adapter now
   reconstructs positions from ordered tokens, checks all text alignment and
   retains article boundaries when batching sentences. Initial incorrect outputs
   are quarantined under `invalid/` and excluded from the final report.
2. The XLM-R model card's simple aggregation frequently produced subword fragments
   as separate entities. The adapter now supports explicit word-ID aggregation;
   the Slavic checkpoint defaults to `first`. The completed simple-decoder baseline
   remains in `xlmr-sweep/batch-16-repeat-1` (437.14 s), separately from the final
   word-level result. Its following run was deliberately stopped.
3. Windows virtualenv launchers can leave their child model process running after
   the launcher is terminated. Sweep timeout/interruption cleanup now kills and
   reaps the worker tree. A regression test exercises a real spawned descendant.

## Local artifacts and next use

- `artifacts/ner-benchmark/review/report.html`: final offline review.
- `artifacts/ner-benchmark/review/name_checks.json`: contexts, labels and spans.
- `artifacts/ner-benchmark/timing-summary/summary.csv`: configurations, sample
  sizes and timings; JSON retains full configuration metadata.
- `artifacts/ner-benchmark/classla-aligned-sweep/`: three full-sample configurations.
- `artifacts/ner-benchmark/xlmr-word-tuning/`: three 2,000-article configurations.
- `artifacts/ner-benchmark/xlmr-final/`: selected XLM-R full-sample result.

Open the report locally:

```powershell
Invoke-Item C:/GitHub/semora/artifacts/ner-benchmark/review/report.html
```

For future models, use the Transformers adapter with another checkpoint or a
`module:factory` adapter. Reuse the sample and regenerate reports from saved
predictions. Before committing to full-corpus indexing, select the model using
the report and, if needed, run repeated timings or a longer stability test with
the same commands. No production NER index or Nora tool has been enabled.
