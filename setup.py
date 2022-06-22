from setuptools import setup
from setuptools import find_packages

setup(
    name='curvefit',
    version='1.0.0',
    description='Curve Fitting Tool',
    url='https://github.com/ihmeuw-msca/CurveFit',
    license='MIT',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy==1.22.0',
        'scipy',
        'pandas',
        'pytest',
        'matplotlib',
        'xspline',
        'scikit-learn',
        'pydantic',
        'tqdm'
    ],
    zip_safe=False,
)
