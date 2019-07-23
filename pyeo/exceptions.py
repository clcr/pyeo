"""
Contains ForestSentinel exception classes.
"""

import requests


class ForestSentinelException(Exception):
    pass


class StackImagesException(ForestSentinelException):
    pass


class CreateNewStacksException(ForestSentinelException):
    pass


class StackImageException(ForestSentinelException):
    pass


class BadS2Exception(ForestSentinelException):
    pass


class BadGoogleURLExceeption(ForestSentinelException):
    pass


class BadDataSourceExpection(ForestSentinelException):
    pass


class NoL2DataAvailableException(ForestSentinelException):
    pass


class FMaskException(ForestSentinelException):
    pass


class TooManyRequests(requests.RequestException):
    """Too many requests; do exponential backoff"""