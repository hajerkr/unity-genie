## Neuroimaging Derivative Downloader

This repository provides a workflow for downloading and aggregating neuroimaging derivatives (e.g., recon-all or minimorph) across one or multiple projects. The tool wraps existing Python scripts into a Streamlit interface, allowing users to run the workflow without using the command line.

### Features

#### Data Download Module
* Select project(s) to fetch derivatives from
* Choose derivative type (recon-all / minimorph)
* Progress bar and status updates during download
* Combine results into a single CSV
* Preview and download the aggregated results

#### Data Cleaning Module
* Detect outliers within an age-group
* Clean failed segmenations, outliers, duplicate data

#### Segmentation QC
* Visualise 3-plane segmentation
* Assess and record the quality of the segmented regions 
* Produce a csv report on the QC'ed segmentations

#### Batch runs
* Submit batch jobs at the project level
* _(Coming soon) Upload a CSV specifying session to process_

### Requirements

* Python 3.9+
* Dependencies listed in requirements.txt

Install with:
```
pip install -r requirements.txt
```

Or use the provided `start.sh` to set up and launch in one step (see [Usage](#usage)).

### Authentication

This tool requires a Flywheel API key. The recommended setup for new users is a `.env` file:

1. On first run, `start.sh` creates `.env` automatically from `.env_example`
2. Open `.env` and replace the placeholder:
```
FW_CLI_API_KEY=your_api_key_here
```
3. Re-run `bash start.sh` — the app will authenticate silently.

> Advanced: if `FW_CLI_API_KEY` is already exported in your shell (e.g. `.zshrc`), no `.env` is needed.

⚠️ Do not commit `.env` to GitHub — it is already listed in `.gitignore`.

### Usage

To run this app locally:

```bash
bash start.sh
```

This will create a `.venv` if one does not exist, install dependencies, and launch the app. Or run manually:
```
streamlit run Home.py
```

This will open a local web interface in your browser.
Steps in the app: 

1. Enter one or more projects
2. Choose the derivative type (recon-all or minimorph)
3. Click Fetch derivatives.

Monitor progress in real-time.


_Note: This app has been deployed on the Streamlit cloud, however access must be requested from the author of this repository._


### Development Notes

Replace the placeholder functions (download_derivatives, assemble_csv) with your own implementations.

If extending the app, keep API key handling through .env for security.

For debugging, run the app with:
```
streamlit run Home.py --logger.level=debug
```