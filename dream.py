# XXX Fill out docstring!
"""
dream.py from deepdreamer
Created: 8/2/15

Purpose:

Examples:

Usage:

"""
from __future__ import print_function
import argparse
import datetime
import logging
import json
import os
import sys
# Custom code
import proc
import utility

log = logging.getLogger(__name__)
__author__ = 'wgibb'
__version__ = '0.0.1'

def make_output_fp(config):
    input_fn = os.path.basename(config.get('input'))
    utility.safe_makedirs(config.get('output'))
    bn, extension = input_fn.rsplit('.', 1)
    now = datetime.datetime.utcnow()
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    output_fn = '{bn}_{ts}.{ext}'.format(bn=bn,
                                         ts=timestamp,
                                         ext=extension)
    output_fp = os.path.join(config.get('output'), output_fn)
    return output_fp


def run_dream(config):
    output_fp = make_output_fp(config)
    dreamp = proc.Proc(model_path=config.get('model_path'),
                       model_name=config.get('model_name'))

    dreamp.process_image(input_fp=config.get('input'),
                         output_fp=output_fp,
                         **config.get('deepdream_params'))
    return True

def dump_layers(config):
    pass