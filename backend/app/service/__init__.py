"""Phase 6: the FastAPI layer — one process, one model, HTTP in front of it.

See `app/viewer.py`'s module docstring for why this did not exist earlier:
no job queue, no model in the process, no `/trace` or `/steer` until now.
"""
