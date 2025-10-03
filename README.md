## Neuroimaging Derivative Downloader

This repository provides a workflow for downloading and aggregating neuroimaging derivatives (e.g., recon-all or minimorph) across one or multiple projects. The tool wraps existing Python scripts into a Streamlit interface, allowing users to run the workflow without using the command line.

### Features

* Select project(s) to fetch derivatives from
* Choose derivative type (recon-all / minimorph)
* Progress bar and status updates during download
* Combine results into a single CSV
* Preview and download the aggregated results

### Requirements

* Python 3.9+
* Dependencies listed in requirements.txt

Install with:
```
pip install -r requirements.txt
```

Authentication

This tool requires an API key to access data. The API key should be stored in a .env file in the root of the repository.

1. Create a .env file
2. Add your API key to the file:
```
API_KEY=your_api_key_here
```

The app automatically loads this file using python-dotenv
.

⚠️ Do not commit your .env file to GitHub. Make sure .gitignore includes it.

### Usage

Run the Streamlit app:
```
streamlit run app.py
```

This will open a local web interface in your browser.
Steps in the app: 

1. Enter one or more projects
2. Choose the derivative type (recon-all or minimorph)
3. Click Fetch derivatives.

Monitor progress in real-time.

Preview the combined CSV and download it.

### Output

Individual derivative files are downloaded locally into a project-based folder structure.

A summary CSV (derivatives_summary.csv) is generated, containing merged results across all subjects/projects.

### Development Notes

Replace the placeholder functions (download_derivatives, assemble_csv) with your own implementations.

If extending the app, keep API key handling through .env for security.

For debugging, run the app with:
```
streamlit run app.py --logger.level=debug
```