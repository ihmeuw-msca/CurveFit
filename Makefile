# makefile for easy manage package
.PHONY: clean, tests

build: setup.py
	python setup.py build

install: setup.py
	python setup.py install

sdist: setup.py
	python setup.py sdist

tests:
	pytest tests

clean:
	find . -name "*.so*" | xargs rm -rf
	find . -name "*.pyc" | xargs rm -rf
	find . -name "__pycache__" | xargs rm -rf
	find . -name "build" | xargs rm -rf
	find . -name "dist" | xargs rm -rf
	find . -name "MANIFEST" | xargs rm -rf
	find . -name "*.egg-info" | xargs rm -rf
	find . -name ".pytest_cache" | xargs rm -rf

uninstall:
	find $(CONDA_PREFIX)/lib/ -name "*curvefit*" | xargs rm -rf