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
        super(LSTMSpecialist, self).__init__()
        self.num_composers = num_composers
        self.composer_embeddings = nn.Embedding(num_composers, hidden_size)
        self.notes_encoder = nn.Linear(in_features=input_size,
                                       out_features=hidden_size)

    def forward(self, inp, tag, hidden=None):
        if not hidden:
            self.hidden = self.composer_embeddings()
        super.forward(inp, tag)
