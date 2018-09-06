from __future__ import absolute_import, division, print_function

import pytest

import warnings

# Avoid benign scipy-numpy version warnings following
# https://stackoverflow.com/questions/40845304
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


def pytest_collection_modifyitems(items):
    for item in items:
        if item.nodeid.startswith("tests/contrib"):
            if "stage" not in item.keywords:
                item.add_marker(pytest.mark.stage("unit"))
            if "init" not in item.keywords:
                item.add_marker(pytest.mark.init(rng_seed=123))
