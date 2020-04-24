import os
import torch

from lib.dataset import ThreeDPW
from lib.models import VIBE
from lib.core.evaluate import Evaluator
from lib.core.config import parse_args
from torch.utils.data import DataLoader


def main(cfg):
    print('...Evaluating on 3DPW test set...')

    model = VIBE(
        n_layers=cfg.MODEL.TGRU.NUM_LAYERS,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        seqlen=cfg.DATASET.SEQLEN,
        hidden_size=cfg.MODEL.TGRU.HIDDEN_SIZE,
        pretrained=cfg.TRAIN.PRETRAINED_REGRESSOR,
        add_linear=cfg.MODEL.TGRU.ADD_LINEAR,
        bidirectional=cfg.MODEL.TGRU.BIDIRECTIONAL,
        use_residual=cfg.MODEL.TGRU.RESIDUAL,
    ).to(cfg.DEVICE)

    if cfg.TRAIN.PRETRAINED != '' and os.path.isfile(cfg.TRAIN.PRETRAINED):
        checkpoint = torch.load(cfg.TRAIN.PRETRAINED)
        best_performance = checkpoint['performance']
        model.load_state_dict(checkpoint['gen_state_dict'])
        print(f'==> Loaded pretrained model from {cfg.TRAIN.PRETRAINED}...')
        print(f'Performance on 3DPW test set {best_performance}')
    else:
        print(f'{cfg.TRAIN.PRETRAINED} is not a pretrained model!!!!')
        exit()

    test_db = ThreeDPW(set='test', seqlen=cfg.DATASET.SEQLEN, debug=cfg.DEBUG)

    test_loader = DataLoader(
        dataset=test_db,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    Evaluator(
        model=model,
        device=cfg.DEVICE,
        test_loader=test_loader,
    ).run()


if __name__ == '__main__':
    cfg, cfg_file = parse_args()

    main(cfg)
