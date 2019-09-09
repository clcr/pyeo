"""
Contains ForestSentinel exception classes.
"""

import requests


class PyeoException(Exception):
    pass


class StackImagesException(PyeoException):
    pass


class CreateNewStacksException(PyeoException):
    pass


class StackImageException(PyeoException):
    pass


class BadS2Exception(PyeoException):
    pass


class BadGoogleURLExceeption(PyeoException):
    pass


class BadDataSourceExpection(PyeoException):
    pass


class NoL2DataAvailableException(PyeoException):
    pass


class FMaskException(PyeoException):
    pass

class NonSquarePixelException(PyeoException):
    pass

class TooManyRequests(requests.RequestException):
    """Too many requests; do exponential backoff"""