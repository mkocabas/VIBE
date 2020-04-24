import sys
sys.path.append('.')

import torch
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from lib.dataset import *
from lib.utils.vis import batch_draw_skeleton, batch_visualize_preds


def debug_2d_data(dataset, DEBUG=True):
    is_train = True
    seqlen = 32
    batch_size = 1
    db = eval(dataset)(seqlen=seqlen, debug=DEBUG)

    dataloader = DataLoader(
        dataset=db,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )

    for i, target in enumerate(dataloader):
        for k, v in target.items():
            print(k, v.shape)

        if DEBUG:
            if dataset is 'Insta':
                input = torch.ones(batch_size, seqlen, 3, 224, 224)[0]
            else:
                input = target['video'][0]
            single_target = {k: v[0] for k, v in target.items()}

            dataset_name = 'spin'
            plt.figure(figsize=(19.2,10.8))
            images = batch_draw_skeleton(input, single_target, dataset=dataset_name, max_images=4)
            plt.imshow(images)
            plt.show()

        if i == 20:
            break


if __name__ == '__main__':
    debug_2d_data('Insta', DEBUG=True)
