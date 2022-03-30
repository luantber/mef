example: 
	python example/run.py

env:
	conda env export > environment.yml

.PHONY: example env