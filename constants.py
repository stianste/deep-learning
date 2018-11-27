import torch
import torch.nn as nn

"""General"""
BATCH_SIZE = 1
NUM_EPOCHS = 100
FS = 5

INPUT_SIZE = 128
HIDDEN_SIZE = 128
NUM_HIDDEN_LAYERS = 2
LEARNING_RATE = 0.005

LOSS_FUNCTION = nn.MSELoss()
OPTIMIZER = torch.optim.Adam

SEQ_LEN = 200

PRETRAINED_MODELS_PATH = "./models/pretrained_models/"
