# Changelog

## 2026-04-30
- `start.sh`: auto-creates `.env` from `.env_example` if missing and exits with a prompt to fill in credentials
- Fixed README: `API_KEY` → `FW_CLI_API_KEY`; added `cp .env_example .env` step

## 2026-04-30 (initial)
- Added `start.sh`: creates `.venv` if absent, installs requirements, launches `Home.py`
- Updated README: launch instructions updated to use `start.sh`; fixed `app.py` → `Home.py` references
