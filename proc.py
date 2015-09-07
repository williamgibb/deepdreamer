# XXX Fill out docstring!
"""
proc.py from deepdreamer
Created: 8/2/15

Purpose:

Examples:

Usage:

"""
# Stdlib
from __future__ import print_function
import logging
import os
# Third Party code
# imports and basic notebook setup
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from google.protobuf import text_format
# noinspection PyUnresolvedReferences
import caffe
# Custom Code
log = logging.getLogger(__name__)
__author__ = 'wgibb'
__version__ = '0.0.1'


# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']


def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])


class Proc(object):
    def __init__(self, model_path, model_name, net_fn='deploy.prototxt'):
        self.objective_data = {}
        self.model_path = str(model_path)
        self.model_name = str(model_name)
        self.net_fn = os.path.join(self.model_path, net_fn)
        self.param_fn = os.path.join(self.model_path, '{}.caffemodel'.format(self.model_name))
        # Patching model to be able to compute gradients.
        # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
        model = caffe.io.caffe_pb2.NetParameter()
        text_format.Merge(open(self.net_fn).read(), model)
        model.force_backward = True
        self.tmp_prototxt = os.path.join(self.model_path, 'tmp.prototxt')
        with open(self.tmp_prototxt, 'w') as f:
            f.write(str(model))
        # XXX mean and channel_swap should come from **kwargs
        # And default to the values for bvlc_googlenet?
        self.net = caffe.Classifier(self.tmp_prototxt,
                                    self.param_fn,
                                    mean=np.float32([104.0, 116.0, 122.0]),  # ImageNet mean, training set dependent
                                    channel_swap=(2, 1, 0))  # the reference model has channels in BGR order instead of RGB

    def reset(self):
        if self.objective_data:
            log.info('Resetting guide_data')
            self.objective_data = {}

    def make_step(self,
                  objective,
                  step_size=1.5,
                  end='inception_4c/output',
                  jitter=32,
                  clip=True,
                  **kwargs
                  ):
        """
        Basic gradient ascent step

        :param objective:
        :param step_size:
        :param end:
        :param jitter:
        :param clip:
        :param kwargs:
        :return:
        """

        src = self.net.blobs['data']  # input image is stored in Net's 'data' blob
        dst = self.net.blobs[end]
        # XXX Make jitter optional ? Will need to experiment to see what that does/does not do.
        ox, oy = np.random.randint(-jitter, jitter + 1, 2)
        src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2)  # apply jitter shift

        self.net.forward(end=end)
        # Do a lookup of the objective function!
        objective_func = getattr(self, 'objective_{}'.format(objective), None)
        if not objective_func:
            raise ValueError('Could not find objective function')
        objective_func(dst, **kwargs.get('objective_args', {}))  # specify the optimization objective
        self.net.backward(start=end)
        g = src.diff[0]
        # apply normalized ascent step to the input image
        src.data[:] += step_size / np.abs(g).mean() * g

        src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2)  # unshift image

        if clip:
            bias = self.net.transformer.mean['data']
            src.data[:] = np.clip(src.data, -bias, 255 - bias)

    def deepdream(self,
                  base_img,
                  iter_n=10,
                  octave_n=4,
                  octave_scale=1.4,
                  end='inception_4c/output',
                  clip=True,
                  **kwargs):
        """

        :param base_img:
        :param iter_n:
        :param octave_n:
        :param octave_scale:
        :param end:
        :param clip:
        :param kwargs:
        :return:
        """
        # prepare base images for all octaves
        octaves = [preprocess(self.net, base_img)]
        for i in xrange(octave_n - 1):
            octaves.append(nd.zoom(octaves[-1], (1, 1.0 / octave_scale, 1.0 / octave_scale), order=1))

        src = self.net.blobs['data']
        detail = np.zeros_like(octaves[-1])  # allocate image for network-produced details
        for octave, octave_base in enumerate(octaves[::-1]):
            h, w = octave_base.shape[-2:]
            if octave > 0:
                # upscale details from the previous octave
                h1, w1 = detail.shape[-2:]
                detail = nd.zoom(detail, (1, 1.0 * h / h1, 1.0 * w / w1), order=1)

            src.reshape(1, 3, h, w)  # resize the network's input image size
            src.data[0] = octave_base + detail
            for i in xrange(iter_n):
                self.make_step(end=end, clip=clip, **kwargs.get('step_params', {}))

                # visualization
                vis = deprocess(self.net, src.data[0])
                if not clip:  # adjust image contrast if clipping is disabled
                    vis *= 255.0 / np.percentile(vis, 99.98)
                #                showarray(vis)
                log.info('{}, {}, {}'.format(octave, i, end, vis.shape))
            #                clear_output(wait=True)

            # extract details produced on the current octave
            detail = src.data[0] - octave_base
        # returning the resulting image
        return deprocess(self.net, src.data[0])

    def process_image(self, input_fp, output_fp, reset=True, **deepdream_params):
        """

        :param input_fp:
        :param output_fp:
        :param reset:
        :param deepdream_params:
        :return:
        """
        if reset:
            self.reset()
        frame = np.float32(PIL.Image.open(input_fp))
        frame = self.deepdream(base_img=frame, **deepdream_params)
        PIL.Image.fromarray(np.uint8(frame)).save(output_fp)
        return frame

    @staticmethod
    def objective_l2(dst, **kwargs):
        dst.diff[:] = dst.data

    def objective_guide(self, dst, **kwargs):
        x = dst.data[0].copy()
        y = self.objective_data.get('guide_features', None)
        if not y:
            # We make a new net since self.net is already loaded
            # with data by the time we are inside of the guide function.
            # In short, the first time we enter into the objective_guide is that
            # we need to reload the current net locally, (which we can do with the
            # self. params.
            net = caffe.Classifier(self.tmp_prototxt,
                                   self.param_fn,
                                   # XXX See note in __init__ about seeting mean / channel_swap
                                   mean=np.float32([104.0, 116.0, 122.0]),  # ImageNet mean, training set dependent
                                   channel_swap=(2, 1, 0))
            guide_image = kwargs.get('guide_image')
            log.info('Building guide from [{}]'.format(guide_image))
            guide = np.float32(PIL.Image.open(guide_image))
            end = kwargs.get('guide_end_layer', 'inception_3b/output')
            h, w = guide.shape[:2]
            src, dst = net.blobs['data'], net.blobs[end]
            src.reshape(1,3,h,w)
            src.data[0] = preprocess(self.net, guide)
            net.forward(end=end)
            y = dst.data[0].copy()
            self.objective_data['guide_features'] = y
        ch = x.shape[0]
        x = x.reshape(ch, -1)
        y = y.reshape(ch, -1)
        a = x.T.dot(y)  # compute the matrix of dot-products with guide features
        dst.diff[0].reshape(ch, -1)[:] = y[:, a.argmax(1)]  # select ones that match best

    def net_layers(self):
        """

        :return:
        """
        return self.net.blobs.keys()
