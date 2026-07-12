import torch


def pytest_sessionstart(session):  # noqa: ARG001
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
