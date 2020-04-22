from setuptools import setup
from setuptools import find_packages

setup(
    name='curvefit',
    version='0.0.0',
    description='Curve Fitting Tool',
    url='https://github.com/ihmeuw-msca/CurveFit',
    license='MIT',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'pytest',
        'matplotlib',
        'xspline',
        'scikit-learn',
        'pyspellchecker',
        'mkdocs',
        'mkdocs-material',
        'mkautodoc',
        'pymdown-extensions',
        'python-markdown-math'
    ],
    zip_safe=False,
)
