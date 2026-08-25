.PHONY: help install lint test train serve clean

help:
	@echo "Available commands:"
	@echo "  make install  - Install all dependencies"
	@echo "  make lint     - Run flake8 code quality check"
	@echo "  make format   - Auto-format code with black and isort"
	@echo "  make test     - Run all unit tests"
	@echo "  make train    - Run full training pipeline"
	@echo "  make serve    - Start FastAPI server"
	@echo "  make clean    - Remove temporary files"

install:
	pip install -r requirements.txt

lint:
	flake8 src/ tests/

format:
	isort src/ tests/
	black src/ tests/

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

train:
	python -m src.train_pipeline.rf_pipeline

serve:
	python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
