.PHONY: install run train test lint format typecheck clean

PYTHON := python
PIP := pip

install:
	$(PIP) install -r requirements.txt

run:
	$(PYTHON) main.py --config config/settings.yaml

train:
	$(PYTHON) -m classifier.trainer --gesture neutral --samples 60
	$(PYTHON) -m classifier.trainer --gesture head_turn_left --samples 60
	$(PYTHON) -m classifier.trainer --gesture head_turn_right --samples 60
	$(PYTHON) -m classifier.trainer --gesture head_nod_down --samples 60
	$(PYTHON) -m classifier.trainer --gesture head_tilt_up --samples 60
	$(PYTHON) -m classifier.trainer --gesture open_palm --samples 60
	$(PYTHON) -m classifier.trainer --gesture fist --samples 60
	$(PYTHON) -m classifier.trainer --gesture peace_sign --samples 60

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

lint:
	ruff check .
	$(PYTHON) -m mypy --strict .

format:
	black .
	ruff check --fix .

typecheck:
	$(PYTHON) -m mypy --strict .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .mypy_cache .pytest_cache .ruff_cache
