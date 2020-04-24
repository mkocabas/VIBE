import sys
sys.path.append('.')
import time
from lib.dataset import *
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from lib.models.smpl import SMPL, SMPL_MODEL_DIR
from lib.utils.vis import batch_draw_skeleton, batch_visualize_preds

dataset = 'MPII3D'
seqlen = 16
DEBUG = True

db = eval(dataset)(set='val', seqlen=seqlen, debug=DEBUG)

dataloader = DataLoader(
    dataset=db,
    batch_size=4,
    shuffle=True,
    num_workers=1,
)

smpl = SMPL(SMPL_MODEL_DIR)

start = time.time()
for i, target in enumerate(dataloader):
    data_time = time.time() - start
    start = time.time()
    print(f'Data loading time {data_time:.4f}')

    for k, v in target.items():
        print(k, v.shape)

    if DEBUG:
        input = target['video'][0]
        single_target = {k: v[0] for k, v in target.items()}

        if dataset == 'MPII3D':
            images = batch_draw_skeleton(input, single_target, dataset='spin', max_images=4)
            plt.imshow(images)
            plt.show()
        else:
            theta = single_target['theta']
            pose, shape = theta[:, 3:75], theta[:, 75:]

            # verts, j3d, smpl_j3d = smpl(pose, shape)

            pred_output = smpl(betas=shape, body_pose=pose[:, 3:], global_orient=pose[:, :3], pose2rot=True)

            single_target['verts'] = pred_output.vertices

            images = batch_visualize_preds(input, single_target, single_target, max_images=4, dataset='spin')
            # images = batch_draw_skeleton(input, single_target, dataset='common', max_images=10)
            plt.imshow(images)
            plt.show()

    if i == 100:
        break