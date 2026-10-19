"""Backward-compatible entry point for the single active 15-input trainer.

Use ``python -m src.train_uci_sem1_15`` for explicitness. This legacy command
delegates to the same trainer and can no longer write an incompatible schema.
"""
from src.train_uci_sem1_15 import *  # noqa: F401,F403
from src.train_uci_sem1_15 import main


if __name__ == "__main__":
    main()
