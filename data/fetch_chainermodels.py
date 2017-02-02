#!/usr/bin/env python
from __future__ import print_function

import os.path as osp
import sys

try:
    import gdown
except ImportError:
    print('Please install gdown: pip install gdown', file=sys.stderr)
    sys.exit(1)


this_dir = osp.dirname(__file__)


def main():
    path = osp.join(this_dir, 'VGG16_faster_rcnn_final.model')
    if not osp.exists(path):
        gdown.download(
            url='https://dl.dropboxusercontent.com/u/2498135/faster-rcnn/VGG16_faster_rcnn_final.model',  # NOQA
            output=path,
            quiet=False,
        )

    path = osp.join(this_dir, 'coco_vgg16_faster_rcnn_final.chainermodel.h5')
    if not osp.exists(path):
        gdown.download(
            url='http://drive.google.com/uc?id=0B19w-32ZKSg6MkxXTEhzTWRUNDg',
            output=path,
            quiet=False,
        )


if __name__ == '__main__':
    main()
