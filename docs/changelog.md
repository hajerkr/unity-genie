# Changelog

## 2026-05-07
- `pages/2_Cleaning_Outlier_Detection.py`: fixed `TypeError` from `set.union()` when no evaluated sets exist.
- `pages/2_Cleaning_Outlier_Detection.py`: fixed `UnboundLocalError` for `projects_str` when no per-tool outliers are confirmed.
- `pages/2_Cleaning_Outlier_Detection.py`: fixed `OSError` when writing final outliers by creating the `data/` directory before save.
- `pages/2_Cleaning_Outlier_Detection.py`: added schema fallback for unprefixed volumetric headers (for example `mm_*` / `ra_*`) when prefixed `MRR_`/`GAMBAS_` columns are not present.
- `pages/2_Cleaning_Outlier_Detection.py`: made column selection defensive so optional fields like `MRR_acquisition` and `GAMBAS_acquisition` do not raise `KeyError` when absent.
- `pages/2_Cleaning_Outlier_Detection.py`: ensured analysis ID columns are carried forward in outlier CSVs for both prefixed and unprefixed schemas (for example `analysis_id_mm`).
- `pages/3_QC_Segmentation.py`: replaced hardcoded analysis-ID lookup with flexible detection across common variants (`MRR_analysis_id_*`, `GAMBAS_analysis_id_*`, `analysis_id_*`).
- `pages/3_QC_Segmentation.py`: fixed upload-state handling to reload new outlier CSV files reliably (file signature tracking), preventing stale session-state schemas.
- `pages/3_QC_Segmentation.py`: normalized uploaded CSV headers (trim spaces, strip BOM, replace spaces with underscores) before analysis-ID detection.
- `pages/3_QC_Segmentation.py`: improved error message to show analysis-id-like columns found in the uploaded file when expected columns are missing.

## 2026-04-30
- `start.sh`: auto-creates `.env` from `.env_example` if missing and exits with a prompt to fill in credentials
- Fixed README: `API_KEY` -> `FW_CLI_API_KEY`; added `cp .env_example .env` step
- `Home.py`: uncommented env var auto-login; changed `load_dotenv(override=True)` -> `load_dotenv()` so shell env takes precedence
- `start.sh`: skips `.env` check if `FW_CLI_API_KEY` already set in shell; improved first-run instructions
- `.gitignore`: fixed `./data/*` -> `data/` to correctly ignore nested subdirectories; untracked `data/` from git index
- `pages/1_Data_Download.py`: fixed f-string quote nesting on line 392 for Python <3.12 compatibility

## 2026-04-30 (initial)
- Added `start.sh`: creates `.venv` if absent, installs requirements, launches `Home.py`
- Updated README: launch instructions updated to use `start.sh`; fixed `app.py` -> `Home.py` references
