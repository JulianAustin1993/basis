from setuptools import setup, find_packages


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='basis',
      version='0.1',
      description='Univariate basis systems in Python.',
      long_description=readme(),
      classifiers=[
          'Development Status :: 1 - Planning',
          'Programming Language :: Python :: 3.8',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Natural Language :: English',
      ],
      keywords='basis',
      url='http://github.com/julianaustin1993/basis',
      author='Julian Austin',
      author_email='J.Austin3@newcastle.ac.uk',
      packages=find_packages(include=['basis', 'basis.*']),
      install_requires=[
          'numpy',
          'scipy'
      ],
      tests_require=['pytest',
                     'numpy',
                     'scipy'],
      include_package_data=True,
      zip_safe=False)
