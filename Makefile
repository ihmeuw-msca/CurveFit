# makefile for easy manage package
.PHONY: clean, tests

phony:

build: setup.py
	python3 setup.py build

# make install with optional prefix=directory on command line
install: setup.py
ifdef prefix
	python3 setup.py install --prefix=$(prefix)
else
	python3 setup.py install
endif
	conda install swig
	pip install -i https://test.pypi.org/simple/ cppad_py --verbose

sdist: setup.py
	python3 setup.py sdist

tests:
	pytest tests

examples:
	python example/get_started.py
	python example/model_runner.py
	python example/covariate.py
	python example/random_effects.py
	python example/sizes_to_indices.py
	python example/param_time_fun.py
	python example/unzip_x.py
	python example/effects2params.py
	python example/objective_fun.py
	python example/prior_initializer.py

# Use mkdocs gh-deploy to make changes to the gh-pages branch.
# This is for running extract_md.py and checking the differences before
# deploying.
gh-pages: phony
	bin/extract_md.py
	mkdocs build
	git checkout gh-pages
	rm -r extract_md
	cp -r site/* .
	git show master:.gitignore > .gitignore
	@echo 'Use the following command to return to master branch:'
	@echo '    rm .gitignore; git reset --hard; git checkout master'
	@echo 'files of the form extract_md/*.md have not yet been deployed'

gh-deploy: phony
	bin/extract_md.py
	mkdocs gh-deploy

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
