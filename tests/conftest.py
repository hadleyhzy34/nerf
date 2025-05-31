# tests/conftest.py
import pytest

def pytest_addoption(parser):
    """Adds command-line options to pytest.

    Allows running tests conditionally based on these options.
    Currently adds:
        --run_cuda_tests: Run tests that require a CUDA-enabled GPU.
    """
    parser.addoption(
        "--run_cuda_tests",
        action="store_true",
        default=False,
        help="run tests that require CUDA and a GPU"
    )
