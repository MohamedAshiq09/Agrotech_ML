#!/usr/bin/env python3
"""
AgroGraphNet Setup Verification Script
Run this script to verify that all files are correctly set up.
"""

import os
import sys
from pathlib import Path
import json

def check_file_exists(file_path, description):
    """Check if a file exists and print status"""
    if Path(file_path).exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} - NOT FOUND")
        return False

def check_json_file(file_path, description):
    """Check if a JSON file exists and is valid"""
    if not Path(file_path).exists():
        print(f"❌ {description}: {file_path} - NOT FOUND")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Additional check for Jupyter notebook structure
        if 'cells' in data and 'metadata' in data:
            print(f"✅ {description}: {file_path}")
            return True
        else:
            print(f"❌ {description}: {file_path} - NOT A VALID NOTEBOOK")
            return False
    except json.JSONDecodeError as e:
        print(f"❌ {description}: {file_path} - INVALID JSON: {str(e)}")
        return False
    except UnicodeDecodeError:
        print(f"❌ {description}: {file_path} - ENCODING ERROR")
        return False
    except Exception as e:
        print(f"❌ {description}: {file_path} - ERROR: {str(e)}")
        return False

def main():
    print("🔍 AgroGraphNet Setup Verification")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("AgroGraphNet").exists() and not Path("notebooks").exists():
        print("❌ Please run this script from the AgroGraphNet directory")
        sys.exit(1)
    
    # If we're in parent directory, change to AgroGraphNet
    if Path("AgroGraphNet").exists():
        os.chdir("AgroGraphNet")
    
    all_good = True
    
    print("\n📁 Checking Directory Structure...")
    directories = [
        "notebooks", "src", "data", "models", "results"
    ]
    
    for directory in directories:
        if Path(directory).exists():
            print(f"✅ Directory: {directory}/")
        else:
            print(f"❌ Directory: {directory}/ - NOT FOUND")
            all_good = False
    
    print("\n📓 Checking Notebook Files...")
    notebooks = [
        ("notebooks/01_data_collection.ipynb", "Data Collection Notebook"),
        ("notebooks/02_data_preprocessing.ipynb", "Data Preprocessing Notebook"),
        ("notebooks/03_graph_construction.ipynb", "Graph Construction Notebook"),
        ("notebooks/04_feature_engineering.ipynb", "Feature Engineering Notebook"),
        ("notebooks/05_model_development.ipynb", "Model Development Notebook"),
        ("notebooks/06_evaluation_visualization.ipynb", "Evaluation & Visualization Notebook"),
        ("notebooks/07_prediction_pipeline.ipynb", "Prediction Pipeline Notebook")
    ]
    
    for notebook_path, description in notebooks:
        if not check_json_file(notebook_path, description):
            all_good = False
    
    print("\n🐍 Checking Source Files...")
    source_files = [
        ("src/config.py", "Configuration File"),
        ("src/data_utils.py", "Data Utilities"),
        ("src/graph_utils.py", "Graph Utilities"),
        ("src/model_utils.py", "Model Utilities"),
        ("src/visualization.py", "Visualization Utilities")
    ]
    
    for file_path, description in source_files:
        if not check_file_exists(file_path, description):
            all_good = False
    
    print("\n📋 Checking Configuration Files...")
    config_files = [
        ("requirements.txt", "Requirements File"),
        ("README.md", "README File"),
        ("SETUP_AND_RUN_GUIDE.md", "Setup Guide")
    ]
    
    for file_path, description in config_files:
        if not check_file_exists(file_path, description):
            all_good = False
    
    print("\n🔍 Checking Python Dependencies...")
    try:
        import numpy
        print("✅ NumPy installed")
    except ImportError:
        print("❌ NumPy not installed")
        all_good = False
    
    try:
        import pandas
        print("✅ Pandas installed")
    except ImportError:
        print("❌ Pandas not installed")
        all_good = False
    
    try:
        import torch
        print("✅ PyTorch installed")
    except ImportError:
        print("❌ PyTorch not installed")
        all_good = False
    
    try:
        import torch_geometric
        print("✅ PyTorch Geometric installed")
    except ImportError:
        print("❌ PyTorch Geometric not installed")
        all_good = False
    
    try:
        import matplotlib
        print("✅ Matplotlib installed")
    except ImportError:
        print("❌ Matplotlib not installed")
        all_good = False
    
    try:
        import folium
        print("✅ Folium installed")
    except ImportError:
        print("❌ Folium not installed")
        all_good = False
    
    print("\n" + "=" * 50)
    
    if all_good:
        print("🎉 ALL CHECKS PASSED!")
        print("\nYou're ready to run the AgroGraphNet project!")
        print("\nNext steps:")
        print("1. Start Jupyter Notebook: jupyter notebook")
        print("2. Open and run notebooks in order (01 through 07)")
        print("3. Check the SETUP_AND_RUN_GUIDE.md for detailed instructions")
    else:
        print("❌ SOME CHECKS FAILED!")
        print("\nPlease fix the issues above before running the project.")
        print("Check the SETUP_AND_RUN_GUIDE.md for troubleshooting help.")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())