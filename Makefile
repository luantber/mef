example: 
	python example/run.py

example_stric:
	mypy example/run.py --namespace-packages

env:
	conda env export > environment.yml

.PHONY: example env