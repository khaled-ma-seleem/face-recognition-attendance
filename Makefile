# Makefile for Face Recognition Attendance System

# Variables
PYTHON := python
PIP := pip
LOG_DIR := logs
SETUP_LOG := $(LOG_DIR)/setup.log
STREAMLIT_LOG := $(LOG_DIR)/streamlit.log
TIMESTAMP := $(shell date +'%Y%m%d_%H%M%S')

# Default target
.DEFAULT_GOAL := all

# -----------------------------------------------------------------------------
# Targets
# -----------------------------------------------------------------------------

all: setup run-background

help:
	@echo "Usage:"
	@echo "  make all               Run setup + start Streamlit app (foreground)"
	@echo "  make help              Show this help message"
	@echo "  make setup             Install dependencies (Python + system libs)"
	@echo "  make run               Run Streamlit app (foreground)"
	@echo "  make run-background    Run Streamlit app in background with PID tracking"
	@echo "  make run-timestamped   Run with timestamped log files"
	@echo "  make stop-app          Stop the background Streamlit app"
	@echo "  make check-app         Check if background app is running"
	@echo "  make clean             Remove all data files and logs (WARNING!)"
	@echo "  make logs              Create logs directory"
	@echo "  make tail-setup        View last 20 lines of setup log"
	@echo "  make tail-app          View last 20 lines of Streamlit log"
	@echo "  make view-logs         List all available log files"

setup: logs
	@echo "📦 Upgrading pip..."
	$(PIP) install --upgrade pip >> $(SETUP_LOG) 2>&1
	@echo "📦 Installing Python dependencies from requirements.txt..."
	$(PIP) install -r requirements.txt >> $(SETUP_LOG) 2>&1
	@echo "📦 Installing system dependencies..."
	sudo apt-get update >> $(SETUP_LOG) 2>&1
	sudo apt-get install -y libgl1 >> $(SETUP_LOG) 2>&1
	@echo "✅ Setup completed! Logs saved to $(SETUP_LOG)"

run: logs
	@echo "🚀 Starting Streamlit app..."
	@echo "📝 Logs will be saved to $(STREAMLIT_LOG)"
	$(PYTHON) -m streamlit run app.py >> $(STREAMLIT_LOG) 2>&1

run-background: logs
	@echo "🚀 Starting Streamlit app in background..."
	@echo "📝 Logs will be saved to $(STREAMLIT_LOG)"
	nohup $(PYTHON) -m streamlit run app.py >> $(STREAMLIT_LOG) 2>&1 & \
	echo $$! > $(LOG_DIR)/streamlit.pid
	@echo "✅ App started with PID: $$(cat $(LOG_DIR)/streamlit.pid)"
	@echo "ℹ️  Use 'make tail-app' to view logs or 'make stop-app' to stop"

run-timestamped: logs
	@echo "🚀 Starting Streamlit app with timestamped logs..."
	@echo "📝 Logs will be saved to $(LOG_DIR)/streamlit_$(TIMESTAMP).log"
	nohup $(PYTHON) -m streamlit run app.py >> $(LOG_DIR)/streamlit_$(TIMESTAMP).log 2>&1 & \
	echo $$! > $(LOG_DIR)/streamlit.pid
	@echo "✅ App started with PID: $$(cat $(LOG_DIR)/streamlit.pid)"
	@echo "ℹ️  Use 'make stop-app' to stop"

stop-app:
	@if [ -f $(LOG_DIR)/streamlit.pid ]; then \
		PID=$$(cat $(LOG_DIR)/streamlit.pid); \
		if ps -p $$PID > /dev/null 2>&1; then \
			kill -TERM $$PID && \
			echo "✅ Stopped Streamlit app (PID: $$PID)"; \
		else \
			echo "⚠️  Process $$PID not found"; \
		fi; \
		rm -f $(LOG_DIR)/streamlit.pid; \
	else \
		echo "⚠️  No PID file found. Is the app running?"; \
	fi

check-app:
	@if [ -f $(LOG_DIR)/streamlit.pid ]; then \
		PID=$$(cat $(LOG_DIR)/streamlit.pid); \
		if ps -p $$PID > /dev/null 2>&1; then \
			echo "✅ Streamlit app is running (PID: $$PID)"; \
		else \
			echo "⚠️  PID file exists but process $$PID is not running"; \
			rm -f $(LOG_DIR)/streamlit.pid; \
		fi; \
	else \
		echo "⚠️  No Streamlit app running (no PID file)"; \
	fi

clean:
	@echo "⚠️  WARNING: This will delete all data files in ./data/ and logs in ./logs/"
	@echo "⚠️  Press Ctrl+C now to cancel, or wait 5 seconds..."
	sleep 5
	rm -rf data/
	rm -rf $(LOG_DIR)/
	@echo "🗑️  Data folder and logs removed!"

logs:
	@echo "📁 Creating logs directory..."
	mkdir -p $(LOG_DIR)
	@echo "📁 Logs directory created at $(LOG_DIR)/"

tail-setup:
	@if [ -f $(SETUP_LOG) ]; then \
		echo "📋 Last 20 lines of setup log:"; \
		tail -20 $(SETUP_LOG); \
	else \
		echo "⚠️  Setup log file not found: $(SETUP_LOG)"; \
	fi

tail-app:
	@if [ -f $(STREAMLIT_LOG) ]; then \
		echo "📋 Last 20 lines of Streamlit log:"; \
		tail -20 $(STREAMLIT_LOG); \
	else \
		echo "⚠️  Streamlit log file not found: $(STREAMLIT_LOG)"; \
	fi

view-logs:
	@if [ -d $(LOG_DIR) ]; then \
		echo "📊 Available log files in $(LOG_DIR)/:"; \
		ls -lA $(LOG_DIR)/ | grep -v '^total'; \
		if [ -f $(LOG_DIR)/streamlit.pid ]; then \
			echo ""; \
			echo "📋 PID file contents:"; \
			cat $(LOG_DIR)/streamlit.pid; \
		fi; \
	else \
		echo "⚠️  No log directory found: $(LOG_DIR)"; \
	fi
