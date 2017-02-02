from __future__ import print_function

import os.path as osp

import caffe
import chainer

from lib.models.faster_rcnn import FasterRCNN


this_dir = osp.dirname(__file__)


def main():
    caffemodel_url = 'http://www.cs.berkeley.edu/~rbg/faster-rcnn-data/coco_vgg16_faster_rcnn_final.caffemodel'  # NOQA
    caffemodel_path = chainer.dataset.cached_download(caffemodel_url)
    caffe_prototxt = osp.join(
        osp.dirname(caffe.__file__),
        '../../../models/coco/VGG16/fast_rcnn/test.prototxt')

    chainermodel_path = osp.join(
        this_dir, 'coco_vgg16_faster_rcnn_final.chainermodel.h5')

    caffemodel = caffe.Net(caffe_prototxt, caffemodel_path, caffe.TEST)

    chainermodel = FasterRCNN(num_classes=81)

    for name, param in caffemodel.params.iteritems():
        if name.startswith('conv'):
            layer = getattr(chainermodel.trunk, name)
        else:
            layer = getattr(chainermodel, name)

        print('{0}:'.format(name))
        # weight
        print('  - W:', param[0].data.shape, layer.W.data.shape)
        assert param[0].data.shape == layer.W.data.shape
        layer.W.data = param[0].data
        # bias
        has_bias = False if len(param) == 1 else True
        if has_bias:
            print('  - b:', param[1].data.shape, layer.b.data.shape)
            assert param[1].data.shape == layer.b.data.shape
            layer.b.data = param[1].data

    chainer.serializers.save_hdf5(chainermodel_path, chainermodel)


if __name__ == '__main__':
    main()
