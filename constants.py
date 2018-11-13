import torch
import torch.nn as nn

"""General"""
# Currently only works with batchsize 1, since the songs are of unequal length
BATCH_SIZE = 1
NUM_EPOCHS = 100

HIDDEN_SIZE = 128
LEARNING_RATE = 0.001

LOSS_FUNCTION = nn.MSELoss()
OPTIMIZER = torch.optim.Adam

STEP = 40
