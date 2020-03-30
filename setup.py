from setuptools import setup

setup(name='curvefit',
      version='0.0.0',
      description='Curve Fitting Tool',
      url='https://github.com/ihmeuw-msca/CurveFit',
      license='MIT',
      packages=['curvefit'],
      package_dir={'': 'src'},
      install_requires=['numpy',
                        'scipy',
                        'pandas',
                        'pytest',
                        'matplotlib'],
      zip_safe=False)