from pathlib import Path

import pytest

from tests.make_fixtures import make_fixtures
from uni_rag.config import Config, Source


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    fixtures = tmp_path / "fixtures"
    make_fixtures(fixtures)
    return Config(
        (Source("fixtures", fixtures),),
        data_dir=tmp_path / "data",
        power_log=tmp_path / "data/power.log",
        root=tmp_path,
        energy_backend="none",
    )
