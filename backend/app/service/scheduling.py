"""Serialize model selection and job submission, including model switches."""
from functools import wraps
from inspect import signature
from threading import RLock

_lock = RLock()


def serialized(fn):
    @wraps(fn)
    def guarded(*args, **kwargs):
        with _lock:
            return fn(*args, **kwargs)
    # FastAPI must resolve forward annotations in the endpoint's own module.
    guarded.__signature__ = signature(fn, eval_str=True)
    return guarded
