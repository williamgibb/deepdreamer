# XXX Fill out docstring!
"""
dream.py from deepdreamer
Created: 8/2/15

Purpose:

Examples:

Usage:

"""
from __future__ import print_function
import hashlib
import json
import logging
import os
# Custom code
import proc
import utility

log = logging.getLogger(__name__)
__author__ = 'wgibb'
__version__ = '0.0.1'


DREAMED_FN_SEP = '_DD_'


def make_output_fp(config):
    utility.safe_makedirs(config.get('output'))
    input_fn = os.path.basename(config.get('input'))
    if DREAMED_FN_SEP in input_fn:
        bn, suffix = input_fn.split(DREAMED_FN_SEP)
        _, extension = suffix.rsplit('.', 1)
    else:
        bn, extension = input_fn.rsplit('.', 1)
    h = hashlib.md5(json.dumps(config, sort_keys=True)).hexdigest()
    fn_fmt = '{bn}{const}{h}.{ext}'
    output_fn = fn_fmt.format(bn=bn,
                              const=DREAMED_FN_SEP,
                              h=h,
                              ext=extension)
    json_fn_fmt = '{bn}_{h}.{ext}'
    config_output_fn = json_fn_fmt.format(bn=bn,
                                          h=h,
                                          ext='json')
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

    return output_fp


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


def continual_dream(config, n, zoom=False, intensify=0):
    # XXX Add the zoom code from google here?
    dreamp = proc.Proc(model_path=config.get('model_path'),
                       model_name=config.get('model_name'))
    log.info('Dreaming the same image {} times'.format(n))
    for i in range(n):
        if i and i % 10 == 0:
            if intensify:
                iterations = config['deepdream_params']['iter_n']
                ni = iterations + intensify
                log.info('Setting the iterations value to {}'.format(ni))
                config['deepdream_params']['iter_n'] = ni
        log.info('Dream iteration {}'.format(i+1))
        output_fp = run_dream(config, dreamp=dreamp)
        config['input'] = output_fp


def dump_layers(config):
    dreamp = proc.Proc(model_path=config.get('model_path'),
                       model_name=config.get('model_name'))
    layers = dreamp.net_layers()
    layers = [layer for layer in layers if layer not in ['prob', 'data']]
    s = json.dumps(layers, indent=2, sort_keys=True)
    print(s)
