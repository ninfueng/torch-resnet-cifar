.PHONY: run
run:
	python main.py

.PHONY: fmt
fmt:
	isort .
	black --skip-string-normalization .

.PHONY: clean
clean:
	find __pycache__ | xargs rm -rf
