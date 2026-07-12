.PHONY: setup install test mechanism smoke paper link all figures status package clean

setup:
	bash setup.sh

install:
	python -m pip install -e ".[dev]"

test:
	PYTHONPATH=src pytest -q

mechanism:
	bash run_mechanism.sh

smoke:
	bash run_smoke.sh

paper:
	bash run_paper.sh

link:
	bash run_link.sh

all:
	bash run_everything.sh

figures:
	bash run_visualize.sh outputs/results/complete.csv outputs/reports/complete

status:
	PYTHONPATH=src python scripts/status.py

package:
	python scripts/package_project.py

clean:
	rm -rf outputs/runs outputs/results outputs/reports outputs/status outputs/manifests outputs/cache dist
