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
start.sh
```
In start.sh, a virtual environment is created and dependencies are installed:

```
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate 
# Install required packages
pip install -r requirements.txt
```

Authentication

This tool requires an API key to access data. The API key should be stored in a .env file in the root of the repository.

1. Create a .env file
2. Add your API key to the file:
```
FW_CLI_API_KEY=your_api_key_here
```

The app automatically loads this file using python-dotenv
.

⚠️ Do not commit your .env file to GitHub. Make sure .gitignore includes it.

### Usage

To run this app locally:

Run the Streamlit app:
```
venv/bin/python -m streamlit run Home.py
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
streamlit run app.py --logger.level=debug
```