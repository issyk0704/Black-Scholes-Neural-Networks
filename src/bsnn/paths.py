"""Where cached market data and trained models live on disk.

Running from a source checkout keeps everything inside the repository
(``data/`` and ``models/``). An installed copy uses ``~/.bsnn`` instead, and the
``BSNN_HOME`` environment variable overrides both.
"""

import os
from pathlib import Path


def _default_home() -> Path:
    env = os.environ.get("BSNN_HOME")
    if env:
        return Path(env).expanduser()
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "pyproject.toml").exists():
        return repo_root
    return Path.home() / ".bsnn"


HOME = _default_home()
DATA_DIR = HOME / "data"
STOCK_DIR = DATA_DIR / "stock"
OPTIONS_DIR = DATA_DIR / "options"
MODELS_DIR = HOME / "models"
