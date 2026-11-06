# How random_audit.csv was produced

`random_audit.csv` is a random sample of 100 papers (25 per conference) from `1-paper-scraping/cv_papers_2021_2026_with_intros.json`, labeled by hand in `category_1`–`category_3`. A blank row means the paper has no downstream task. The sample was drawn without looking at any model's predictions, so it can be used to estimate each model's recall.

**Running the current `sample_random_audit.py` will not reproduce this file.** It was produced in two steps.

## 1. Original sample (October 17, 2026)

The first version of `sample_random_audit.py` drew 25 papers at random from each conference, with seed 2026:

| Conference | Papers sampled | Papers available |
|---|---|---|
| CVPR 2021 | 25 | 1,651 |
| ICCV 2021 | 25 | 1,592 |
| CVPR 2026 | 25 | 4,039 |
| ECCV 2026 | 25 | 2,830 |

That version did not exclude papers already in `1-paper-scraping/pre_audit.csv`. The papers were then labeled by hand. The script also wrote an `audit_id` column, which is not in the saved file.

## 2. Replacing the paper shared with the pre-audit (November 2, 2026)

One sampled paper was also in `pre_audit.csv`, the 60-paper set used to tune the prompt and taxonomy. Models may do better on those papers than on unseen ones, so the paper was replaced:

- **Removed:** *Jo-SRC: A Contrastive Approach for Combating Noisy Labels* (CVPR 2021, row 8). It had been labeled as having no downstream task.
- **Added:** *Gated Spatio-Temporal Attention-Guided Video Deblurring* (CVPR 2021). It was drawn with `random.Random(2027).choice(...)` from the 1,612 CVPR 2021 papers in neither audit, and then labeled by hand.

No other rows changed. The file now has 100 unique papers and none of them are in `pre_audit.csv`.

One paper, *ReMoGen: Real-time Human Interaction-to-Reaction Generation via Modular Learning from Diverse Data* (CVPR 2026), is also in the older `2-model-comparison/taxonomy_labels_hand_labeled.csv`. It was kept because that file is no longer used for evaluation.

## Why the current script differs

`sample_random_audit.py` now skips papers in `pre_audit.csv` before sampling. This makes each conference's pool of papers smaller, so the same seed draws a different sample. A test run with the defaults shared only 3 of its 100 papers with `random_audit.csv`.

To draw a new audit, pass a different `--output` path. The script will not overwrite an existing file.
