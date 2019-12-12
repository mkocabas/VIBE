# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.spin import Regressor, hmr
from lib.models.attention import SelfAttention


class TemporalEncoder(nn.Module):
    def __init__(
            self,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True
    ):
        super(TemporalEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=2048,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers
        )

        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(hidden_size*2, 2048)
        elif add_linear:
            self.linear = nn.Linear(hidden_size, 2048)
        self.use_residual = use_residual

    def forward(self, x):
        n,t,f = x.shape
        x = x.permute(1,0,2) # NTF -> TNF
        y, _ = self.gru(x)
        if self.linear:
            y = F.relu(y)
            y = self.linear(y.view(-1, y.size(-1)))
            y = y.view(t,n,f)
        if self.use_residual and y.shape[-1] == 2048:
            y = y + x
        y = y.permute(1,0,2) # TNF -> NTF
        return y


class TemporalEncoderWAttention(nn.Module):
    def __init__(
            self,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            attention_size=1024,
            attention_layers=1,
            attention_dropout=0.5,
            use_residual=True,
    ):
        super(TemporalEncoderWAttention, self).__init__()

        self.gru = nn.GRU(input_size=2048, hidden_size=hidden_size, bidirectional=bidirectional, num_layers=n_layers)
        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(hidden_size*2, 2048)
            self.attention = SelfAttention(attention_size*2,
                                           layers=attention_layers,
                                           dropout=attention_dropout)

        elif add_linear:
            self.linear = nn.Linear(hidden_size, 2048)
        self.use_residual = use_residual
        self.attention = SelfAttention(attention_size,
                                       layers=attention_layers,
                                       dropout=attention_dropout)


    def forward(self, x):
        n,t,f = x.shape
        x = x.permute(1,0,2) # NTF -> TNF
        outputs, _ = self.gru(x)
        outputs = outputs.permute(1, 0, 2)
        y, attentions = self.attention(outputs)
        y = y.permute(1, 0, 2)
        if self.linear:

            y = self.linear(y.reshape(-1, y.size(-1)))
            y = y.view(t,n,f)
        if self.use_residual and y.shape[-1] == 2048:
            y = y + x
        y = y.permute(1,0,2) # TNF -> NTF
        return y


class VIBE(nn.Module):
    def __init__(
            self,
            seqlen,
            batch_size=64,
            n_layers=1,
            hidden_size=2048,
            pretrained='data/vibe_data/spin_model_checkpoint.pth.tar',
            add_linear=False,
            bidirectional=False,
            attention=False,
            attention_cfg=None,
            use_residual=True,
            use_6d=True,
            disable_temporal=False
    ):

        super(VIBE, self).__init__()

        self.seqlen = seqlen
        self.batch_size = batch_size
        self.disable_temporal = disable_temporal

        if attention:
            cfg = attention_cfg
            self.encoder = TemporalEncoderWAttention(
                hidden_size=hidden_size,
                bidirectional=bidirectional,
                add_linear=add_linear,
                attention_size=cfg.SIZE,
                attention_layers=cfg.LAYERS,
                attention_dropout=cfg.DROPOUT,
                use_residual=use_residual,
            )
        else:
            self.encoder = TemporalEncoder(
                n_layers=n_layers,
                hidden_size=hidden_size,
                bidirectional=bidirectional,
                add_linear=add_linear,
                use_residual=use_residual,
            )

        # regressor can predict cam, pose and shape params in an iterative way
        self.regressor = Regressor(use_6d=use_6d)

        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            if not use_6d:
                del pretrained_dict['decpose.weight']
                del pretrained_dict['decpose.bias']
                del pretrained_dict['fc1.weight']
                del pretrained_dict['fc1.bias']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')


    def forward(self, input, J_regressor=None):
        # input size NTF
        batch_size, seqlen = input.shape[:2]

        if self.disable_temporal:
            feature = input.reshape(-1, input.size(-1))
        else:
            feature = self.encoder(input)
            feature = feature.reshape(-1, feature.size(-1))


        smpl_output = self.regressor(feature, J_regressor=J_regressor)
        for s in smpl_output:
            s['theta'] = s['theta'].reshape(batch_size, seqlen, -1)
            s['verts'] = s['verts'].reshape(batch_size, seqlen, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(batch_size, seqlen, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(batch_size, seqlen, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(batch_size, seqlen, -1, 3, 3)

        return smpl_output

class VIBE_Demo(nn.Module):
    def __init__(
            self,
            seqlen,
            batch_size=64,
            n_layers=1,
            hidden_size=2048,
            pretrained='data/vibe_data/spin_model_checkpoint.pth.tar',
            add_linear=False,
            bidirectional=False,
            attention=False,
            attention_cfg=None,
            use_residual=True,
            use_6d=True,
            disable_temporal=False
    ):

        super(VIBE_Demo, self).__init__()

        self.seqlen = seqlen
        self.batch_size = batch_size
        self.disable_temporal = disable_temporal

        if attention:
            cfg = attention_cfg
            self.encoder = TemporalEncoderWAttention(
                hidden_size=hidden_size,
                bidirectional=bidirectional,
                add_linear=add_linear,
                attention_size=cfg.SIZE,
                attention_layers=cfg.LAYERS,
                attention_dropout=cfg.DROPOUT,
                use_residual=use_residual,
            )
        else:
            self.encoder = TemporalEncoder(
                n_layers=n_layers,
                hidden_size=hidden_size,
                bidirectional=bidirectional,
                add_linear=add_linear,
                use_residual=use_residual,
            )

        self.hmr = hmr()
        checkpoint = torch.load(pretrained)
        self.hmr.load_state_dict(checkpoint['model'], strict=False)

        # regressor can predict cam, pose and shape params in an iterative way
        self.regressor = Regressor(use_6d=use_6d)

        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            if not use_6d:
                del pretrained_dict['decpose.weight']
                del pretrained_dict['decpose.bias']
                del pretrained_dict['fc1.weight']
                del pretrained_dict['fc1.bias']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')


    def forward(self, input, J_regressor=None):
        # input size NTF
        batch_size, seqlen, nc, h, w = input.shape

        feature = self.hmr.feature_extractor(input.reshape(-1, nc, h, w))

        if not self.disable_temporal:
            feature = feature.reshape(batch_size, seqlen, -1)
            feature = self.encoder(feature)
            feature = feature.reshape(-1, feature.size(-1))


        smpl_output = self.regressor(feature, J_regressor=J_regressor)

        for s in smpl_output:
            s['theta'] = s['theta'].reshape(batch_size, seqlen, -1)
            s['verts'] = s['verts'].reshape(batch_size, seqlen, -1, 3)
            s['kp_2d'] = s['kp_2d'].reshape(batch_size, seqlen, -1, 2)
            s['kp_3d'] = s['kp_3d'].reshape(batch_size, seqlen, -1, 3)
            s['rotmat'] = s['rotmat'].reshape(batch_size, seqlen, -1, 3, 3)

        return smpl_output
