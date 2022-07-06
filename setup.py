# coding=utf-8
import os
from setuptools import setup

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

with open('README.rst') as readme_file:
    long_description = readme_file.read()

setup(
    name='ann2pmml',
    version='1.0.2',
    packages=['ann2pmml'],
    include_package_data=True,
    license='MIT',
    description='Auto PMML Exporter of Nerual Network Models.',
    long_description=long_description,
    url='https://github.com/lampda/ANN2PMML',
    author='lampda',
    author_email='saintree@gmail.com',
    install_requires=[
        'numpy>=1.6.1',
        'SciPy>=0.9',
        'tensorflow>=1.12.0',
        'keras>=1.0.6,<=2.3.1',
        'scikit-learn>=0.22'
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Development Status :: 3 - Alpha',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
