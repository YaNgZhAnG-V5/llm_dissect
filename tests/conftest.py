import pytest


def pytest_addoption(parser):
    parser.addoption("--gpu-id", type=int, default=0, help="GPU ID to use.")


@pytest.fixture
def gpu_id(request):
    return request.config.getoption("--gpu-id")
