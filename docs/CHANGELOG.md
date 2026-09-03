# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `render.yaml` blueprint for deploying the app to Render as a web service (Python runtime, Streamlit start command, health check).
- Per-session data isolation: `get_session_data_dir()` in `utils/utils.py` gives every browser session a private `data/<session_id>/` folder, so concurrent users on the same instance no longer read/write each other's downloads, QC ratings, or cleaned CSVs.
- Best-effort cleanup of stale session folders (older than 6 hours) so disk usage doesn't grow unbounded between deploys.

### Changed
- `pages/1_Data_Download.py`, `pages/2_Cleaning_Outlier_Detection.py`, `pages/3_QC_Segmentation.py`, `pages/6_Misc.py` now resolve their working data directory via `get_session_data_dir()` instead of a single shared `data/` folder.
- Download buttons now show the plain filename (`os.path.basename(...)`) instead of the full on-disk path, so the session id is never exposed to the user.

### Fixed
- `pages/1_Data_Download.py`: `assemble_csv()` previously wrote its combined CSV to a bare relative filename (i.e. the process's current working directory) instead of the app's data folder — now written under the session's data directory.
- `requirements.txt`: removed the `flywheel` package, which shares an import namespace (`flywheel`) with `flywheel-sdk` and was clobbering it during install, causing `ImportError: cannot import name 'Model' from 'flywheel.models'` on a fresh Render build. `flywheel-sdk` alone provides everything the app imports.
