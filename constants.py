import torch
import torch.nn as nn

"""General"""
# Currently only works with batchsize 1, since the songs are of unequal length
BATCH_SIZE = 1
NUM_EPOCHS = 0
FS = 5

INPUT_SIZE = 128
HIDDEN_SIZE = 128
NUM_HIDDEN_LAYERS = 2
LEARNING_RATE = 0.005

LOSS_FUNCTION = nn.MSELoss()
OPTIMIZER = torch.optim.Adam

SEQ_LEN = 200
