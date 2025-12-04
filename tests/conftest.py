"""
Shared pytest configuration and fixtures for all tests.
"""

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_path():
    """Return the project root directory path."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def data_storage_path(tmp_path_factory):
    """Create a temporary data storage directory for tests."""
    return tmp_path_factory.mktemp("data_storage")


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests to avoid state leakage."""
    yield
