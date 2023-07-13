"""
Contains ForestSentinel exception classes.
"""

import requests


class pyeoException(Exception):
    pass


class StackImagesException(pyeoException):
    pass


class CreateNewStacksException(pyeoException):
    pass


class StackImageException(pyeoException):
    pass


class BadS2Exception(pyeoException):
    pass


class BadGoogleURLExceeption(pyeoException):
    pass


class BadDataSourceExpection(pyeoException):
    pass


class NoL2DataAvailableException(pyeoException):
    pass


class FMaskException(pyeoException):
    pass


class InvalidGeometryFormatException(pyeoException):
    pass


class NonSquarePixelException(pyeoException):
    pass


class InvalidDateFormatException(pyeoException):
    pass


class TooManyRequests(requests.RequestException):
    """Too many requests; do exponential backoff"""
