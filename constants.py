import torch.nn as nn
"""General"""
# Currently only works with batchsize 1, since the songs are of unequal length
BATCH_SIZE = 1
NUM_EPOCHS = 10

HIDDEN_SIZE = 128
LEARNING_RATE = 0.005

LOSS_FUNCTION = nn.MSELoss()
