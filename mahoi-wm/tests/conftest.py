"""느린 회귀 테스트는 기본으로 건너뛴다.

    pytest                 # 계약 + 안전 (수십 초)
    pytest --runslow       # 5종 회귀까지 (약 10분)
"""
import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="5종 시나리오 회귀 테스트까지 실행한다")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="--runslow 를 붙이면 실행된다")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
