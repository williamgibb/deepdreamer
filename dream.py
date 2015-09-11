# XXX Fill out docstring!
"""
dream.py from deepdreamer
Created: 8/2/15

Purpose:

Examples:

Usage:

"""
from __future__ import print_function
import datetime
import logging
import json
import os
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
    fn_fmt = '{bn}_{ts}.{ext}'
    output_fn = fn_fmt.format(bn=bn,
                              ts=timestamp,
                              ext=extension)
    config_output_fn = fn_fmt.format(bn=bn,
                                     ts=timestamp,
                                     ext='.json')
    output_fp = os.path.join(config.get('output'), output_fn)
    config_output_fp = os.path.join(config.get('output'), config_output_fn)
    return output_fp, config_output_fp


def run_dream(config, dreamp=None):
    output_fp, config_output_fp = make_output_fp(config)
    if not dreamp:
        dreamp = proc.Proc(model_path=config.get('model_path'),
                           model_name=config.get('model_name'))

    dreamp.process_image(input_fp=config.get('input'),
                         output_fp=output_fp,
                         **config.get('deepdream_params'))

    with open(config_output_fp, 'wb') as f:
        f.write(json.dumps(config))

    return True


def dream_output_layers(config):
    dreamp = proc.Proc(model_path=config.get('model_path'),
                       model_name=config.get('model_name'))

    layers = [layer for layer in dreamp.net_layers() if layer.endswith('output')]
    for layer in layers:
        log.info('Running layer [{}]'.format(layer))
        config['deepdream_params']['end'] = layer
        run_dream(config, dreamp=dreamp)


def march(config, param, start, end, increment):
    dreamp = proc.Proc(model_path=config.get('model_path'),
                       model_name=config.get('model_name'))
    for i in range(start, end, increment):
        config['deepdream_params'][param] = i
        run_dream(config, dreamp=dreamp)


def dump_layers(config):
    dreamp = proc.Proc(model_path=config.get('model_path'),
                       model_name=config.get('model_name'))
    layers = dreamp.net_layers()
    layers = [layer for layer in layers if layer not in ['prob', 'data']]
    s = json.dumps(layers, indent=2, sort_keys=True)
    print(s)
