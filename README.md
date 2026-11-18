# Rule-based screening of ship operational records

This repository holds the rule set, the reference implementation and the criteria catalogue of the rule-based (manual) outlier screening used in

> Seo, Y., Kim, M., Kim, S.-Y., Kwak, Y.-M., Kim, M.S., Kim, J., Ahn, Y. Screening of operational records and engine-signal-free prediction of main-engine fuel flow for twelve sister container ships (submitted).

## Contents

| Path | Content |
|---|---|
| `rules/screening_rules_v1.4.yaml` | All rules of the reference (version 1.4): static limits, spikes, rates of change, freezing and relations, with thresholds and state conditions |
| `rules/ship_scale_fix.yaml` | Per-ship scale correction applied before the rules |
| `code/rule_based_screening.py` | Reference implementation that applies the rules to a one-minute record and writes a per-cell bit mask (one bit per axis) and a berthing flag |
| `docs/rules_table_en.md` | The rules as a table in English, with the ship-specific calibration values (Supplementary Table S1 of the article) |
| `docs/screening_guideline_ko.md`, `docs/criteria_catalogue_ko.csv` | Working guideline and catalogue of criteria (in Korean) |

Ships are denoted A to L. Ships A to G belong to the first building series and ships H to L to the second.

## Data

The measurement records are not part of this repository. They are proprietary and were provided through Korean Register; the records and the data derived from them can be shared only with the permission of Korean Register. The code expects one CSV file per ship in `dataset/` with the 43 columns named in the rule file, reads the rules from `rules/`, and writes the flags to `dataset_baseline/` and summaries to `results/outlier_baseline/`. Run it from the repository root. It uses PyTorch (CUDA if available), pandas, NumPy and PyYAML.

## Bit mask

static 1, spike 2, rate of change 4, freezing 8, relation 16.

## License

MIT License (see `LICENSE`).
