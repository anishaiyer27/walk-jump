import hydra
import torch
from omegaconf import DictConfig

#from walkjump.data import AbBatch
#from walkjump.sampling import walk
#from walkjump.utils import random_discrete_seeds

from base import TrainableScoreModel



# class null denoiser



class NoiseEnergyModel(TrainableScoreModel):
# approx noise, param by EBM with pretrained score-based Bayes estimator
# init, score, apply_noise, conif_optimizers, xhat, comp_loss
