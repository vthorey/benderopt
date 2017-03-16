# encoding: utf-8
from setuptools import setup, find_packages
import sys

import benderopt


def inject_custom_repository(repository_name):
    blacklist = ['register', 'upload']
    inject_arg = '--repository=%s' % (repository_name)
    for command in blacklist:
        try:
            index = sys.argv.index(command)
        except ValueError:
            continue
        sys.argv.insert(index + 1, inject_arg)


inject_custom_repository('internal')


setup(

    name='benderopt',

    # la version du code
    version=benderopt.__version__,

    packages=find_packages(),

    author="valentin",

    author_email="v.thorey@gmail.com",

    description="General optimization library.",

    include_package_data=True,

    url='http://github.com/Dreem-Devices/benderopt',

    classifiers=[
        "Programming Language :: Python",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],

)
