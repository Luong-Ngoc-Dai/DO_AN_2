"""Compatibility entrypoint; new code lives in src/model.py."""
from src.model import *  # noqa: F403

if __name__ == "__main__":
    from src.common import run_cli
    run_cli(main)
