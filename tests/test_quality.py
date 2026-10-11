import pytest

from pipeline.quality import run_checks


@pytest.mark.parametrize("name, ok, detail", run_checks())
def test_data_quality(name, ok, detail):
    assert ok, f"{name} : {detail}"
