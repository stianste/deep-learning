import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=self.num_layers)

        self.hidden2out = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.num_layers, 1, self.hidden_size),
                torch.zeros(self.num_layers, 1, self.hidden_size))

    def forward(self, inp, hidden, tag=None):
        lstm_out, hidden = self.lstm(inp, hidden)
        output = self.hidden2out(lstm_out.view(len(inp), -1))
        output = torch.sigmoid(output)
        return output, hidden


class LSTMSpecialist(LSTM):
    def __init__(self, input_size, hidden_size,
                 output_size, num_layers=1, num_composers=4):
        super(LSTMSpecialist, self).__init__(
              input_size, hidden_size, output_size, num_layers=1
              )
        self.h_embed = nn.Embedding(num_composers, hidden_size)
        self.c_embed = nn.Embedding(num_composers, hidden_size)

    def forward(self, inp, hidden, tag=0):
        if hidden is None:
            h_t = self.h_embed(tag).view(self.num_layers, 1, self.input_size)
            c_t = self.c_embed(tag).view(self.num_layers, 1, self.input_size)
            hidden = (h_t, c_t)

        return super(LSTMSpecialist, self).forward(inp, hidden, tag)
