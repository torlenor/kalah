import numpy as np

import torch.nn as nn
import torch.nn.functional as F

eps = np.finfo(np.float32).eps.item()

class ReinforceModel(nn.Module):
    """
    implements REINFORCE model
    """

    def __init__(self, inputs, outputs, neurons, drop_out):
        super(ReinforceModel, self).__init__()
        self._drop_out = drop_out

        self.affine1 = nn.Linear(inputs, neurons)
        if self._drop_out > eps:
            self.dropout = nn.Dropout(p=self._drop_out)
        self.affine2 = nn.Linear(neurons, outputs)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        if self._drop_out > eps:
            x = self.dropout(x)
        x = F.relu(x)

        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=-1)
