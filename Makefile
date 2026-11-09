.PHONY: install run test figures clean

install:            ## install dependencies
	pip install -r requirements-dev.txt

run:                ## generate data -> validate -> load warehouse -> report
	python -m dq_pipeline

test:               ## run the test suite
	python -m pytest -q

figures:            ## regenerate the charts in docs/
	python scripts/make_figures.py

clean:              ## remove generated files
	rm -rf data/raw warehouse/*.db reports/*.html outputs .pytest_cache
