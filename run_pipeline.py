#!/usr/bin/env python3
"""
STI Predictor Pro - Complete Pipeline Runner
Runs the entire project pipeline from data generation to web application.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"🚀 {title}")
    print("="*60)

def print_step(step, description):
    """Print a formatted step."""
    print(f"\n📋 Step {step}: {description}")
    print("-" * 40)

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_file_exists(filepath):
    """Check if a file exists."""
    return Path(filepath).exists()

def main():
    """Run the complete STI Predictor Pro pipeline."""
    print_header("STI Predictor Pro - Complete Pipeline")
    print("This script will run the entire project pipeline:")
    print("1. Generate synthetic data")
    print("2. Preprocess and engineer features")
    print("3. Train multiple ML models")
    print("4. Launch the web application")
    print("\n⚠️  Note: This project uses synthetic data for educational purposes only.")
    
    # Check if we're in the right directory
    if not check_file_exists("requirements.txt"):
        print("❌ Error: requirements.txt not found. Please run this script from the project root directory.")
        return False
    
    # Step 1: Install dependencies
    print_step(1, "Installing Dependencies")
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        return False
    
    # Step 2: Generate data
    print_step(2, "Generating Synthetic Data")
    if not check_file_exists("src/data/generate_data.py"):
        print("❌ Error: Data generation script not found.")
        return False
    
    if not run_command("python src/data/generate_data.py", "Generating synthetic STI data"):
        return False
    
    # Step 3: Preprocess data
    print_step(3, "Preprocessing Data")
    if not check_file_exists("src/data/preprocessing.py"):
        print("❌ Error: Preprocessing script not found.")
        return False
    
    if not run_command("python src/data/preprocessing.py", "Preprocessing and feature engineering"):
        return False
    
    # Step 4: Train models
    print_step(4, "Training Machine Learning Models")
    if not check_file_exists("src/models/train_models.py"):
        print("❌ Error: Model training script not found.")
        return False
    
    if not run_command("python src/models/train_models.py", "Training ML models"):
        return False
    
    # Step 5: Run tests
    print_step(5, "Running Tests")
    if check_file_exists("tests/test_models.py"):
        run_command("python -m pytest tests/ -v", "Running unit tests")
    else:
        print("⚠️  Warning: Test files not found, skipping tests.")
    
    # Step 6: Launch web application
    print_step(6, "Launching Web Application")
    if not check_file_exists("webapp/app.py"):
        print("❌ Error: Web application not found.")
        return False
    
    print("🌐 Starting Streamlit web application...")
    print("📱 The application will open in your browser at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the application")
    print("\n" + "="*60)
    
    try:
        # Launch the web application
        subprocess.run("streamlit run webapp/app.py", shell=True, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching web application: {e}")
        return False
    
    return True

def quick_start():
    """Quick start function for experienced users."""
    print_header("STI Predictor Pro - Quick Start")
    print("This will run the minimal setup to get the web app running.")
    
    # Check if models already exist
    if check_file_exists("models/saved_models/best_model.joblib"):
        print("✅ Models found, skipping training...")
        print("🌐 Launching web application...")
        try:
            subprocess.run("streamlit run webapp/app.py", shell=True, check=True)
        except KeyboardInterrupt:
            print("\n🛑 Application stopped by user.")
        return True
    else:
        print("❌ No trained models found. Please run the full pipeline first.")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="STI Predictor Pro Pipeline Runner")
    parser.add_argument("--quick", action="store_true", help="Quick start (skip training if models exist)")
    
    args = parser.parse_args()
    
    if args.quick:
        success = quick_start()
    else:
        success = main()
    
    if success:
        print("\n🎉 Pipeline completed successfully!")
    else:
        print("\n❌ Pipeline failed. Please check the error messages above.")
        sys.exit(1) 