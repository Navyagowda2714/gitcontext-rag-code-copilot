from __future__ import annotations

from pathlib import Path

import pytest

from gitcontext.config import Settings
from gitcontext.engine import GitContext
from gitcontext.llm import OfflineProvider
from gitcontext.samples import build_sample_repo


@pytest.fixture(scope="session")
def sample_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_sample_repo(tmp_path_factory.mktemp("repo") / "shopfront-api")


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    return Settings(index_dir=tmp_path_factory.mktemp("index"))


@pytest.fixture(scope="session")
def engine(sample_repo: Path, settings: Settings) -> GitContext:
    return GitContext.build(sample_repo, settings=settings, llm=OfflineProvider())
