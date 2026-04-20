
# Set a virtual environment and run the Streamlit app
python3 -m venv venv
source venv/bin/activate 

# ADD venv to .gitignore
echo "venv/" >> .gitignore

# Install required packages
pip install -r requirements.txt

# Run the Streamlit application (from within the virtual environment)
venv/bin/python -m streamlit run Home.py