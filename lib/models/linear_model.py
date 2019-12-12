# This script is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/LinearModel.py
# Adhere to their licence to use this script.

import torch.nn as nn
import numpy as np
import sys
import torch
from torch.nn.utils import spectral_norm

class LinearModel(nn.Module):
    '''
        input param:
            fc_layers: a list of neuron count, such as [2133, 1024, 1024, 85]
            use_dropout: a list of bool define use dropout or not for each layer, such as [True, True, False]
            drop_prob: a list of float defined the drop prob, such as [0.5, 0.5, 0]
            use_ac_func: a list of bool define use active function or not, such as [True, True, False]
    '''

    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func, use_spectral_norm=False):
        super(LinearModel, self).__init__()
        self.fc_layers = fc_layers
        self.use_dropout = use_dropout
        self.drop_prob = drop_prob
        self.use_ac_func = use_ac_func
        self.use_spectral_norm = use_spectral_norm

        if not self._check():
            msg = 'wrong LinearModel parameters!'
            print(msg)
            sys.exit(msg)

        self.create_layers()

    def _check(self):
        while True:
            if not isinstance(self.fc_layers, list):
                print('fc_layers require list, get {}'.format(type(self.fc_layers)))
                break

            if not isinstance(self.use_dropout, list):
                print('use_dropout require list, get {}'.format(type(self.use_dropout)))
                break

            if not isinstance(self.drop_prob, list):
                print('drop_prob require list, get {}'.format(type(self.drop_prob)))
                break

            if not isinstance(self.use_ac_func, list):
                print('use_ac_func require list, get {}'.format(type(self.use_ac_func)))
                break

            l_fc_layer = len(self.fc_layers)
            l_use_drop = len(self.use_dropout)
            l_drop_porb = len(self.drop_prob)
            l_use_ac_func = len(self.use_ac_func)

            return l_fc_layer >= 2 and l_use_drop < l_fc_layer and l_drop_porb < l_fc_layer and l_use_ac_func < l_fc_layer and l_drop_porb == l_use_drop

        return False

    def create_layers(self):
        l_fc_layer = len(self.fc_layers)
        l_use_drop = len(self.use_dropout)
        l_drop_porb = len(self.drop_prob)
        l_use_ac_func = len(self.use_ac_func)

        self.fc_blocks = nn.Sequential()

        for _ in range(l_fc_layer - 1):
            if self.use_spectral_norm:
                self.fc_blocks.add_module(
                    name='regressor_fc_{}'.format(_),
                    module=spectral_norm(nn.Linear(in_features=self.fc_layers[_], out_features=self.fc_layers[_ + 1]))
                )
            else:
                self.fc_blocks.add_module(
                    name='regressor_fc_{}'.format(_),
                    module=nn.Linear(in_features=self.fc_layers[_], out_features=self.fc_layers[_ + 1])
                )

            if _ < l_use_ac_func and self.use_ac_func[_]:
                self.fc_blocks.add_module(
                    name='regressor_af_{}'.format(_),
                    module=nn.ReLU()
                )

            if _ < l_use_drop and self.use_dropout[_]:
                self.fc_blocks.add_module(
                    name='regressor_fc_dropout_{}'.format(_),
                    module=nn.Dropout(p=self.drop_prob[_])
                )

    def forward(self, inputs):
        msg = 'the base class [LinearModel] is not callable!'
        sys.exit(msg)


if __name__ == '__main__':
    fc_layers = [2133, 1024, 1024, 85]
    iterations = 3
    use_dropout = [True, True, False]
    drop_prob = [0.5, 0.5, 0]
    use_ac_func = [True, True, False]
    device = torch.device('cuda')
    net = LinearModel(fc_layers, use_dropout, drop_prob, use_ac_func).to(device)
    print(net)
    nx = np.zeros([2, 2048])
    vx = torch.from_numpy(nx).to(device)
