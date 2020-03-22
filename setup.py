from setuptools import setup

setup(name='curvefit',
      version='0.0.0',
      description='Curve Fitting Tool',
      url='https://github.com/zhengp0/covid-19',
      author='Peng Zheng',
      author_email='zhengp@uw.edu',
      license='MIT',
      packages=['curvefit'],
      package_dir={'': 'src'},
      install_requires=['numpy',
                        'scipy',
                        'pandas',
                        'pytest'],
      zip_safe=False)