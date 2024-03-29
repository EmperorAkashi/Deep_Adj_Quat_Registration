#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from setuptools import find_namespace_packages, setup

setup(
    name='Deep_AdjQuat_Registration',
    version='0.1.0',
    license='MIT',
    description='Learning based LiDAR registration on different Dataset',
    author='Chen Lin, ...',
    author_email='clin@flatironinstitute.org',
    url='',
    packages=find_namespace_packages(where='src'),
    package_dir={'': 'src'},
    package_data={'': ['*.yaml']},
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.8',
    install_requires=[
        'hydra-core==1.3.1',
        'pytorch-lightning==1.9.4',
        'matplotlib',
        'tensorboard',
        'tensorflow',
    ],
)