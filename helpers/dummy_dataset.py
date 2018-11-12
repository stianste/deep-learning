import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):

    def __init__(self):
        self.length = 32
        self.input_size = self.length // 2

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_tensor = torch.zeros(self.length - 1, 1, self.input_size)
        output_tensor = torch.zeros(self.length - 1, 1, self.input_size)
        tags = torch.zeros(self.length - 1, 1, self.input_size)

        for i in range(self.input_size):
            input_tensor[i][0][i % self.input_size] = 1
            output_tensor[i][0][(i + 1) % self.input_size] = 1

        for i in range(self.input_size):
            input_tensor[i + self.input_size - 1][0][
                self.input_size - i - 1] = 1
            output_tensor[i + self.input_size - 1][0][
                self.input_size - i - 2] = 1

        return input_tensor, tags, output_tensor
