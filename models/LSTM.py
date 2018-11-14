import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2out = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_size),
                torch.zeros(1, 1, self.hidden_size))

    def forward(self, inp):
        lstm_out, self.hidden = self.lstm(inp, self.hidden)
        output = self.hidden2out(lstm_out.view(len(inp), -1))
        output = torch.sigmoid(output)
        return output

    def reset_hidden(self):
        self.hidden = self.init_hidden()
