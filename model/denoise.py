import torch
from torch import nn

#from walkjump.data import AbBatch
#from walkjump.utils import isotropic_gaussian_noise_like

from base import TrainableScoreModel

#  denoise model extends trainable score model from base
class DenoiseModel(TrainableScoreModel):
# score, comp_loss

