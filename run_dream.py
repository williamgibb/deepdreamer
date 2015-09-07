# XXX Fill out docstring!
"""
run_dream.py from deepdreamer
Created: 8/23/15

Purpose:

Examples:

Usage:

"""
# Stdlib
from __future__ import print_function
import argparse
import logging
import json
import os
import sys
# Custom code
import dream

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s [%(filename)s:%(funcName)s]')
log = logging.getLogger(__name__)

__author__ = 'wgibb'
__version__ = '0.0.1'


def main(options):
    if not options.verbose:
        logging.disable(logging.DEBUG)
    log.debug('Reading configuration')
    with open(options.config, 'rb') as f:
        conf = json.loads(f.read().decode('utf-8'))
    if 'input' not in conf:
        log.debug('Setting input to [{}]'.format(options.input))
        conf['input'] = options.input
    if 'output' not in conf:
        log.debug('Setting output to [{}]'.format(options.output))
        conf['output'] = options.output

    if options.dump_layers:
        dream.dump_layers(conf)
        sys.exit(0)

    if options.dream_layers:
        dream.dream_output_layers(conf)
        sys.exit(0)
    dream.run_dream(conf)


def makeargpaser():
    # XXX Fill in description!
    parser = argparse.ArgumentParser(description="Runner script for dream/proc code.")
    parser.add_argument('-i', '--input', dest='input', default=None, action='store',
                        help='Input file.')
    parser.add_argument('-o', '--output', dest='output', default=None, action='store',
                        help='Output directory.  It will be created if it does not exist.')
    parser.add_argument('-c', '--config', dest='config', required=True, action='store',
                        help='Configuration file')
    parser.add_argument('--dump-layers', dest='dump_layers', default=False, action='store_true',
                        help='Dump the layers for the configured model and exit')
    parser.add_argument('--dream-layers', dest='dream_layers', default=False, action='store_true',
                        help='Dream all the output layers?')
    parser.add_argument('-v', '--verbose', dest='verbose', default=False, action='store_true',
                        help='Enable verbose output')
    return parser


if __name__ == '__main__':
    p = makeargpaser()
    opts = p.parse_args()
    main(opts)
