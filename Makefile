# makefile for easy manage package
.PHONY: clean, tests

phony:

gh-pages: phony
	bin/extract_md.py
	mkdocs build
	git checkout mkdocs.yml
	git checkout gh-pages
	rm -r extract_md
	cp -r site/* .

build: setup.py
	python setup.py build

# make install with optional prefix=directory on command line
install: setup.py
ifdef prefix
	python setup.py install --prefix=$(prefix)
else
	python setup.py install
endif

sdist: setup.py
	python setup.py sdist

tests:
	pytest tests

examples:
	python example/get_started.py
	python example/covariate.py
	python example/random_effect.py
	python example/sizes_to_indices.py
	python example/param_time_fun.py

cppad_py: phony
	pytest cppad_py

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
