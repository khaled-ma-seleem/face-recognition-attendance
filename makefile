# Makefile for Face Recognition Attendance System

# Variables
PYTHON := python
PIP := pip

# Default target
.DEFAULT_GOAL := all

# -----------------------------------------------------------------------------
# Targets
# -----------------------------------------------------------------------------

all: setup run

help:
	@echo "Usage:"
	@echo "  make            Run setup + start Streamlit app"
	@echo "  make setup      Install dependencies (Python + system libs)"
	@echo "  make run        Run Streamlit app"
	@echo "  make clean      Remove all data files (WARNING!)"

setup:
	@echo "📦 Upgrading pip..."
	$(PIP) install --upgrade pip -q
	@echo "📦 Installing Python dependencies from requirements.txt..."
	$(PIP) install -r requirements.txt -q
	@echo "📦 Installing system dependencies..."
	sudo apt-get update
	sudo apt-get install -y libgl1

run:
	@echo "🚀 Starting Streamlit app..."
	$(PYTHON) -m streamlit run app.py

clean:
	@echo "⚠️  WARNING: This will delete all data files in ./data/"
	@echo "⚠️  Press Ctrl+C now to cancel, or wait 5 seconds..."
	sleep 5
	rm -rf data/
	@echo "🗑️  Data folder removed!"