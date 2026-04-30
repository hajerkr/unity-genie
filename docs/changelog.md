# Changelog

## 2026-04-30
- `start.sh`: auto-creates `.env` from `.env_example` if missing and exits with a prompt to fill in credentials
- Fixed README: `API_KEY` → `FW_CLI_API_KEY`; added `cp .env_example .env` step
- `Home.py`: uncommented env var auto-login; changed `load_dotenv(override=True)` → `load_dotenv()` so shell env takes precedence
- `start.sh`: skips `.env` check if `FW_CLI_API_KEY` already set in shell; improved first-run instructions
- `.gitignore`: fixed `./data/*` → `data/` to correctly ignore nested subdirectories; untracked `data/` from git index
- `pages/1_Data_Download.py`: fixed f-string quote nesting on line 392 for Python <3.12 compatibility

## 2026-04-30 (initial)
- Added `start.sh`: creates `.venv` if absent, installs requirements, launches `Home.py`
- Updated README: launch instructions updated to use `start.sh`; fixed `app.py` → `Home.py` references
