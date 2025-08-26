.PHONY: prepare train report infer lint test clean all

all: prepare train report

prepare:
	@echo "📦 Preparing data..."
	@python src/data.py

train: prepare
	@echo "⚙️ Training baseline model..."
	@python src/train_sklearn.py

report: train
	@echo "📊 Generating reports and visuals..."
	@python src/report_baseline.py

infer:
	@echo "🔮 Running quick inference demo..."
	@python -m src.infer --text "great product, loved it!"

lint:
	@echo "🔍 Linting..."
	@ruff check .
	@black --check .
	@isort --check-only .

test:
	@echo "🧪 Running tests..."
	@pytest -q

clean:
	@echo "🧹 Cleaning generated files..."
	@rm -rf __pycache__ .pytest_cache .ruff_cache
	@rm -rf data/*.parquet models/baseline/* metrics/* reports/*


⸻