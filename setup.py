from setuptools import setup, PEP420PackageFinder

setup(name='curvefit',
      version='0.0.0',
      description='Curve Fitting Tool',
      url='https://github.com/ihmeuw-msca/CurveFit',
      license='MIT',
      packages=PEP420PackageFinder.find("src"),
      package_dir={"": "src"},
      install_requires=['numpy',
                        'scipy',
                        'pandas',
                        'pytest',
                        'matplotlib'],
      zip_safe=False)
