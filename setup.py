# encoding: utf-8
from setuptools import setup

import benderopt
with open("README.md", "r") as fh:
    long_description = fh.read()


setup(

    name='benderopt',

    # la version du code
    version=benderopt.__version__,

    packages=["benderopt"],

    author="Valentin Thorey",

    author_email="v.thorey@gmail.com",

    description="Black Box optimization library.",

    include_package_data=True,

    url='https://github.com/Dreem-Organization/benderopt',

    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],

    long_description=long_description,

    long_description_content_type="text/markdown",

    install_requires=[
        'numpy>=1.15.4',
        'scipy>=1.1.0',
    ],

)
