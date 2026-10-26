SHELL := /bin/bash

.PHONY: validate source-data figures quality-check checksums controlled-preflight controlled-preflight-required

validate:
	python tools/validate_repository.py

source-data:
	rm -rf reproduced/reporting_source_data
	mkdir -p reproduced/reporting_source_data
	python code/04_reporting/build_manuscript_source_data.py --results-root results --out-root reproduced/reporting_source_data
	python tools/verify_reporting_source_data.py --generated reproduced/reporting_source_data --reference results/reporting

figures:
	rm -rf reproduced/figures
	mkdir -p reproduced/figures
	python code/04_reporting/make_publication_figures.py --source-data results/reporting/figure_data --out reproduced/figures
	python tools/verify_reproduced_outputs.py --generated reproduced/figures --reference results/reporting/figures

quality-check: validate source-data figures controlled-preflight

checksums:
	python tools/validate_repository.py --checksums-only

controlled-preflight:
	python tools/preflight_controlled_inputs.py

controlled-preflight-required:
	python tools/preflight_controlled_inputs.py --require-present
