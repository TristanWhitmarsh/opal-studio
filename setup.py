"""Setup script for opal-studio.

All package metadata lives in pyproject.toml; this file only lets the classic
commands work. To publish a release on PyPI:

    rm -rf opal_studio.egg-info build dist
    python setup.py sdist bdist_wheel
    twine check dist/*
    twine upload dist/*

Bump __version__ in opal_studio/__init__.py first — PyPI never accepts the same
version twice.
"""
from setuptools import setup

setup()
