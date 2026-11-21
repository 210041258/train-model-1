# --- Variables ---
PYTHON = python
TRAIN_SCRIPT = main.py
EVAL_SCRIPT = multi_seed_eval.py
ANALYSIS_SCRIPT = convergence_analysis.py

# Default target executed when you just type 'make'
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make setup      : Install dependencies from requirements.txt"
	@echo "  make train      : Run the main training script (main.py)"
	@echo "  make eval       : Run multi-seed evaluation (multi_seed_eval.py)"
	@echo "  make analyze    : Run convergence analysis and generate plots"
	@echo "  make pipeline   : Run train, eval, and analyze in sequence"
	@echo "  make clean      : Remove Python cache files (__pycache__)"
	@echo "  make clean_out  : Remove generated logs, csvs, and plots"

# --- Environment Setup ---
.PHONY: setup
setup:
	$(PYTHON) -m pip install --upgrade pip
	@if [ -f requirements.txt ]; then $(PYTHON) -m pip install -r requirements.txt; else echo "No requirements.txt found. Please create one."; fi

# --- Training & Evaluation ---
.PHONY: train
train:
	@echo "Starting training..."
	$(PYTHON) $(TRAIN_SCRIPT)

.PHONY: eval
eval:
	@echo "Starting multi-seed evaluation..."
	$(PYTHON) $(EVAL_SCRIPT)

# --- Analysis ---
.PHONY: analyze
analyze:
	@echo "Running convergence analysis..."
	$(PYTHON) $(ANALYSIS_SCRIPT)

# Run the whole pipeline
.PHONY: pipeline
pipeline: train eval analyze
	@echo "Full pipeline complete!"

# --- Cleanup ---
# Removes Python compiled files and caches
.PHONY: clean
clean:
	@echo "Cleaning up Python cache..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Removes generated output files (be careful with this!)
.PHONY: clean_out
clean_out:
	@echo "Removing generated outputs..."
	rm -f analysis_*.png analysis_*.json analysis_*.npz analysis_*.csv
	rm -f final_*.png final_*.pt train_log*.csv train_log*.jsonl