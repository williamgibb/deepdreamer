# XXX Fill out docstring!
"""
utility.py from deepdreamer
Created: 8/23/15

Purpose:

Examples:

Usage:

"""
# Stdlib
from __future__ import print_function
import logging
import os
import sys
# Third Party code
# Custom Code
log = logging.getLogger(__name__)
__author__ = 'wgibb'
__version__ = '0.0.1'

def safe_makedirs(fp):
    if os.path.isdir(fp):
        return
    if os.path.isfile(fp):
        raise ValueError('Path exists and it is a file {}'.format(fp))
    try:
        os.makedirs(fp)
    except OSError as e:
        log.exception('Failed to make directory {}'.format(fp))
        raise e

