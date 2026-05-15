"""
download_dataset.py — Automated Dataset Downloader

Downloads the WELFake dataset from Kaggle if it is not present in the workspace.
Requires a Kaggle API token (kaggle.json) placed in ~/.kaggle/kaggle.json.
"""

import os
import zipfile
import subprocess
import sys

DATASET_NAME = "saurabhshahane/fake-news-classification"
EXPECTED_FILE = "WELFake_Dataset.csv"

def download_welfake():
    if os.path.exists(EXPECTED_FILE):
        print(f"Dataset '{EXPECTED_FILE}' already exists in the current directory.")
        return

    print("Checking for Kaggle authentication...")
    # The kaggle library expects ~/.kaggle/kaggle.json
    kaggle_dir = os.path.join(os.path.expanduser('~'), '.kaggle')
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')

    if not os.path.exists(kaggle_json):
        print("\n[ERROR] Kaggle API token not found!")
        print(f"Please place your kaggle.json file in: {kaggle_json}")
        print("To get this file:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Scroll to the 'API' section and click 'Create New Token'")
        print("3. Move the downloaded kaggle.json to the folder mentioned above.")
        sys.exit(1)

    print(f"Downloading dataset '{DATASET_NAME}'...")
    try:
        # Run kaggle via subprocess to avoid import-time auth crashes if token is missing
        subprocess.run(["kaggle", "datasets", "download", "-d", DATASET_NAME], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to download dataset. Ensure your Kaggle token is valid. Details: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("[ERROR] 'kaggle' command not found. Have you installed the requirements? (pip install -r requirements.txt)")
        sys.exit(1)

    zip_file = "fake-news-classification.zip"
    if not os.path.exists(zip_file):
        print(f"[ERROR] Expected downloaded zip file '{zip_file}' not found.")
        sys.exit(1)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(".")

    # Clean up the zip file
    os.remove(zip_file)
    print(f"Successfully downloaded and extracted '{EXPECTED_FILE}'.")

if __name__ == "__main__":
    download_welfake()
