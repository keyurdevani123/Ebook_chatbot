"""Re-export shim — allows `from training.state_handler import ...`
to work while the actual implementation lives in training/ebooks/state_handler.py.
"""
from training.ebooks.state_handler import check_state, read_state, write_state  # noqa: F401
