CHECK_DIR := ./

quality:
		black --check $(CHECK_DIR)
		isort --check-only  $(CHECK_DIR)
		flake8  $(CHECK_DIR)

format:
		black --fast $(CHECK_DIR)
		isort  $(CHECK_DIR)

clean:
		rm -rf .pytest_cache/
		rm -rf dcc.egg-info/
		rm -rf dist/
		rm -rf build/
		find . | grep -E '(\.mypy_cache|__pycache__|\.pyc|\.pyo$$)' | xargs rm -rf

.PHONY: run_ui
run_ui:
	export PYTHONPATH=$(PWD) && \
	chainlit run chainlit_chat/ui/main.py --host 0.0.0.0 --port 3000