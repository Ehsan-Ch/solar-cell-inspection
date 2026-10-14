# Solar Cell Inspection Desk

A complete classical computer vision project for screening **electroluminescence (EL) images of solar cells**: data verification, image features, model selection, held-out evaluation, visual error analysis, portable inference, and a human review queue.

**Result:** the fixed screening threshold finds 132 of 143 labeled defects in 445 held-out images (**92.3% recall**) and raises 134 false alarms (**49.6% precision**). This demonstrates a review-prioritization trade-off, not automatic quality certification.

## Explore

Download the repository and open [reports/benchmark/report.html](reports/benchmark/report.html). The report works offline and includes embedded image previews, a threshold explorer, missed defects, false alarms, occlusion sensitivity, subgroup comparisons, brightness stress tests and a sortable-by-score inspection queue. GitHub displays HTML source; download the file to interact with it.

This is an image classifier using HOG, local binary patterns and logistic regression. It does not locate or segment cracks and does not use a neural network. Its inputs are grayscale EL cell crops, not ordinary solar-panel photographs.

## Reproduce

Python 3.11+; no GPU needed.

```sh
python -m venv .venv
# Activate the environment using your shell's activation command.
python -m pip install -e .
python -m unittest discover -s tests -v
python -m solarvision.cli download --data data
python -m solarvision.cli run --data data --out reports/my-run
```

The downloader retrieves a pinned public dataset archive (about 94 MB). Use a fresh data directory and an empty output directory: existing prepared data and benchmark outputs are protected from accidental replacement. A local archive can be supplied with `download --archive path/to/archive.zip`.

The benchmark used Python 3.12.14. Its dependency versions are recorded in `requirements-benchmark.txt` and `metrics.json`; on Python 3.12, install that requirements file before `pip install -e .` for closer numerical reproduction. The broader version ranges in `pyproject.toml` support installation on other Python versions. Small floating-point differences can occur across environments.

Predict one downloaded EL image using the saved JSON model:

```sh
python -m solarvision.cli predict data/images/cell0001.png --model reports/benchmark/model.json
```

The command returns a screening score, the frozen threshold and a review flag. Color inputs are rejected; grayscale format alone cannot detect whether an image is out of distribution. The model must still be used only on comparable EL crops.

## Dataset and split

Source: [ELPV dataset, ZAE Bayern](https://github.com/zae-bayern/elpv-dataset), pinned to revision `93e82ae507c36b3f2c8227eade8b792fcbefcca6`. The upstream dataset contains 2,624 normalized, 300 × 300 grayscale cell crops from 44 modules, with defect annotations and mono/polycrystalline type.

The loader validates dimensions and annotations, verifies downloaded-file SHA-256 hashes and removes exact decoded-pixel duplicates before splitting. No exact duplicates were found in this snapshot. Near duplicates are not detected.

| Role | Images | Use |
|---|---:|---|
| Training | 1,333 | Fit feature scaling and logistic coefficients |
| Validation | 445 | Select model and operating threshold |
| Test | 445 | Frozen evaluation |
| Intermediate annotations | 401 | Separate exploratory challenge set |

Only annotation endpoints 0 and 1 enter supervised learning and binary metrics. The 295 images labeled 1/3 and 106 labeled 2/3 are neither rounded nor treated as certain ground truth. Their scores are exported separately; 233 were flagged, but no binary accuracy is claimed for them.

The 60/20/20 split uses seed 42, stratified jointly by endpoint label and cell type. **It is an image-level split, not a module-level split.** The distributed annotations do not identify parent modules. Cells from one module may occur in multiple partitions, so these results can be optimistic for unseen modules. The exported split manifest makes this limitation and the exact image assignments auditable.

## Vision pipeline

All images are resized to 96 × 96 with bilinear interpolation. Absolute intensity is retained; no per-image contrast normalization or augmentation is applied.

Two feature families are compared:

- **Intensity baseline, 42 values:** mean, standard deviation, five quantiles, three intensity fractions, and means/standard deviations over a 4 × 4 grid.
- **Texture family, 1,032 values:** the 42 intensity values, 900 histogram-of-oriented-gradient values (9 orientations, 16-pixel cells, 2 × 2-cell blocks, L2-Hys normalization), and 90 uniform local-binary-pattern histogram values over a 3 × 3 grid.

StandardScaler is fitted only on training features. Six logistic-regression candidates combine these families with regularization C ∈ {0.01, 0.1, 1}. Validation average precision selects the texture model with C = 0.01 (validation AP 0.7526 versus 0.7132 for the best intensity candidate). No cell type is supplied as a predictive feature.

The operating threshold maximizes validation precision among observed thresholds achieving at least 90% validation recall; ties prefer the higher threshold. Scores equal to the threshold are flagged. The selected threshold is **0.11813911995517438**, with 90.2% validation recall and 48.3% validation precision. The same validation set selects model and threshold, so those validation metrics are not an independent estimate. The test set is untouched until both choices are fixed. There is no final refit on validation or test data.

Logistic outputs are **uncalibrated screening scores**, not validated defect probabilities. The portable JSON stores the scaler, coefficients, feature version and threshold; inference does not require loading executable pickle objects.

## Held-out results

The 445-image test set contains 143 labeled defects and 302 clear cells.

| Metric | Result |
|---|---:|
| Defect recall | 92.31% |
| Flag precision | 49.62% |
| Average precision | 0.8361 |
| ROC AUC | 0.8904 |
| F1 | 0.6455 |
| Accuracy | 67.42% |
| Review workload | 266 / 445 images (59.78%) |
| True positive / false negative | 132 / 11 |
| True negative / false positive | 168 / 134 |

For context, always predicting clear would achieve 67.87% accuracy while finding zero defects. Accuracy alone therefore misses the purpose of the high-recall operating point. The test defect prevalence is 32.13%, which provides a reference for precision-based ranking metrics.

At the fixed threshold, monocrystalline recall is 95.16% (180 images), and polycrystalline recall is 90.12% (265 images). A 1,000-replicate class-stratified image bootstrap estimates 95% percentile intervals of 87.41–96.50% for recall and 46.37–53.06% for precision. It conditions on class counts and does not account for module clustering, model-selection uncertainty or deployment shift.

Synthetic brightness tests multiply the held-out pixels by 0.8 or 1.2, rounding and clipping to the 8-bit range, without retuning:

| Image condition | Recall | Precision | Review workload |
|---|---:|---:|---:|
| Original | 92.31% | 49.62% | 59.78% |
| 20% darker | 97.90% | 40.11% | 78.43% |
| 20% brighter | 87.41% | 53.88% | 52.13% |

The shift in workload and missed defects shows sensitivity to acquisition brightness. These transformations are not independent external validation. Before any operational use, evaluate new module-level data and acquisition conditions, establish review costs, and validate score calibration if probabilities are needed.

## Visual interpretation and review

The gallery includes four cases per confusion category, selected closest to the original threshold. It is intentionally difficult and is not a representative random sample. Each 6 × 6 sensitivity map replaces one patch at a time with the image's mean intensity and measures the score drop. Orange indicates a positive drop, with opacity capped at 0.7. It is an occlusion sensitivity experiment, not a defect segmentation or causal explanation; artificial replacement patterns can influence the result.

The threshold slider recalculates the test confusion matrix for exploration. The headline metrics, gallery outcomes and saved queue remain tied to the original validation-selected threshold. Using the slider to optimize the test result would consume the holdout and require a new independent evaluation set.

Every test case enters a SQLite review queue as `QUEUED`, with `FLAGGED` or `LOW_SCORE` priority. Low-score cases remain available for audit. After an actual inspection, a reviewer can record `defect`, `clear`, or `uncertain` plus a required note. Duplicate cases and repeated review transitions are rejected; events have UTC timestamps. For example, after inspecting a case in your own run:

```sh
python -m solarvision.cli review --db reports/my-run/inspection.sqlite --id images/cell0001.png --decision uncertain --note "Needs a second EL exposure"
```

Use an ID that is present in your queue; `cell0001.png` is present in this fixed test split. This is a local workflow, without reviewer authentication or multi-user coordination. HTML/CSV files are static benchmark snapshots and do not update when SQLite reviews change. No fabricated human reviews are included.

## Tests and project layout

**22 local tests pass.** They cover disjoint and reproducible splits, intermediate-label exclusion, duplicate detection, checksum failures, feature validity, sensitivity semantics, inclusive score ties, confusion metrics, JSON/sklearn inference equivalence, test-data invariance of training and threshold selection, bootstrap reproducibility, queue persistence, event rollback, invalid transitions and safe script embedding. GitHub Actions runs the suite on Python 3.11, 3.12 and 3.13 without downloading the dataset.

```text
solarvision/data.py       Pinned download, fingerprints, validation and split
solarvision/features.py   Intensity, HOG, LBP and occlusion sensitivity
solarvision/model.py      Candidate selection, threshold, metrics and JSON inference
solarvision/review.py     SQLite review queue and audit events
solarvision/pipeline.py   End-to-end benchmark and exports
solarvision/report.py     Offline interactive report
solarvision/cli.py        Download, run, predict and review commands
tests/                   Offline behavioral tests
reports/benchmark/       Model, predictions, split, metrics, queue and report
```

Raw images, the virtual environment and SQLite files are excluded from Git and the release ZIP. The report embeds only 16 attributed image previews. Re-running creates your local SQLite database. Dependencies are separate from the downloadable project files.

## Attribution and license

Original project code is MIT licensed; see [LICENSE](LICENSE). ELPV images are **CC BY-NC-SA 4.0**, not MIT. This project's image-containing report, previews and published dataset-derived benchmark artifacts under `reports/benchmark/` are shared under CC BY-NC-SA 4.0; see [DATA_LICENSE.txt](DATA_LICENSE.txt) and [THIRD_PARTY.md](THIRD_PARTY.md). This is a noncommercial research portfolio project. Contact the dataset owners about commercial image use.

Please credit the dataset authors and cite their work when using it in research. Dataset provenance and publication links are in [THIRD_PARTY.md](THIRD_PARTY.md).
