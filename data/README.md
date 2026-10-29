# Data Directory

This directory holds all data artefacts produced by the pipeline.

| Sub-directory | Contents |
|---|---|
| `raw/` | Downloaded GitHub Archive `.json.gz` files (cached). |
| `processed/` | Parsed and cleaned event tables (Parquet / CSV). |
| `splits/` | Train / test edge-split files used for evaluation. |

> **Note:** Raw archive files and processed data files are excluded from
> version control (see `.gitignore`). Re-run the pipeline to regenerate them.

