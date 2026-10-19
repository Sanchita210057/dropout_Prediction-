"""Deprecated 11-input command retained as a safe compatibility shim.

To avoid overwriting the active model with an incompatible schema, this
historical entry point now delegates to the sole supported 15-input trainer.
"""
from src.train_uci_sem1_15 import *  # noqa: F401,F403
from src.train_uci_sem1_15 import main


if __name__ == "__main__":
    main()
