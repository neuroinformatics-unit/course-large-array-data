import importlib
from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def mock_get_ipython(monkeypatch):
    mock_ipython = MagicMock()
    monkeypatch.setattr("IPython.get_ipython", lambda: mock_ipython)
    yield mock_ipython


def test_package_can_be_imported(mock_get_ipython):
    module = importlib.import_module("course_large_array_data")
    assert module is not None
