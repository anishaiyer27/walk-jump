import hydra
import torch
from lightning.pytorch import LightningModule
from omegaconf import DictConfig

#from walkjump.data import AbBatch

# from original
_DEFAULT_TRAINING_PARAMS = {
	"sigma": 1.0
	"lr": 1e-4
	"weight_decay": 0.01
	"warmup_batches": 0.01
	"beta1":0.09
}

class TrainableScoreModel(LightningModule):
# init/constructor, forward, config_optimizers, training_step, val_step, sample_noise, apply_noise, xhat, score, comp_loss
# some of these are not implemented in original code
