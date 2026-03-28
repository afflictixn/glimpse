from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings
from src.storage.database import DatabaseManager
from src.api.server import create_app


@pytest.fixture
def tmp_settings(tmp_path):
    s = Settings(data_dir=tmp_path, port=0)
    s.ensure_dirs()
    return s


@pytest_asyncio.fixture
async def db(tmp_settings):
    manager = DatabaseManager(tmp_settings)
    await manager.initialize()
    yield manager
    await manager.close()


@pytest.fixture
def app(db):
    return create_app(db)
