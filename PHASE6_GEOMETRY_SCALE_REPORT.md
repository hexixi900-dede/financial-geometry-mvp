# Phase 6: FinMME Geometry Scale Check

This is an auxiliary module-isolation result, not the final FinMME question-answering accuracy.

## Setup

- Data: existing FinMME charts already scanned in Phase 1; no new dataset or image search.
- Scope: simple single-y-axis Bar/Line charts with visible value labels.
- Construction: each of 346 viable labels was independently masked and recovered.
- Primary correctness rule: absolute error no larger than 2% of the displayed y-axis span.
- Construction failures count as incorrect in `accuracy_all`.
- No GPU or VLM was used.

## Results

| Type | Candidate labels | Recovered | Coverage | Correct among recovered (2%) | Accuracy including failures (2%) |
|---|---:|---:|---:|---:|---:|
| All | 346 | 259 | 74.9% | 237/259 = 91.5% | 237/346 = 68.5% |
| Bar | 308 | 229 | 74.4% | 223/229 = 97.4% | 223/308 = 72.4% |
| Line | 38 | 30 | 78.9% | 14/30 = 46.7% | 14/38 = 36.8% |

Sensitivity over all chart types:

| Threshold | Correct among recovered | Accuracy including failures |
|---|---:|---:|
| 1% axis span | 226/259 = 87.3% | 226/346 = 65.3% |
| 2% axis span | 237/259 = 91.5% | 237/346 = 68.5% |
| 5% axis span | 247/259 = 95.4% | 247/346 = 71.4% |

## Failure accounting at the 2% threshold

| Type | Construction failure | Recovered but over 2% | Correct |
|---|---:|---:|---:|
| Bar | 79 | 6 | 223 |
| Line | 8 | 16 | 14 |
| All | 87 | 22 | 237 |

## Interpretation

The Geometry measurement idea is strong for accepted Bar samples: 97.4% are within 2% of the y-axis span and all are within 5%. The old Line point recovery is the weak component. Phase 3 already showed that semantic X grounding plus local line fitting fixes this on its small retained subset, so that implementation should be used in the final QA path instead of the older line reader.

These numbers must not be reported as final FinMME QA accuracy. The next experiment should score original FinMME questions end to end using the official Gold/tolerance, with masked-value results kept only as regression evidence.
