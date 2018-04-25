from setuptools import find_packages, setup


setup(name='scikit-kda',
      packages=find_packages(),
      version='0.1.0',
      description='Scikit-learn-compatible Kernel Discriminant Analysis',
      author='David Diaz Vico',
      author_email='david.diaz.vico@outlook.com',
      url='https://github.com/daviddiazvico/scikit-kda',
      download_url='https://github.com/daviddiazvico/scikit-kda/archive/v0.1.0.tar.gz',
      keywords=['scikit-learn', 'kda'],
      classifiers=['Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'Topic :: Software Development',
                   'Topic :: Scientific/Engineering',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6'],
      install_requires=['scikit-learn'])
