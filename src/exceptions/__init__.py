"""Exceptions package for AutoML Platform."""

from .internalServerError import InternalServerError
from .notFoundError import DataSetNotFoundError

__all__ = ["InternalServerError", "DataSetNotFoundError"]
