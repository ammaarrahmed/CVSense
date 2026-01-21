import streamlit as st
import subprocess
import sys

# Download NLTK data on first run
@st.cache_resource
def download_nltk_data():
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt_tab', quiet=True)

# Run on startup
download_nltk_data()
