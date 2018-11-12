import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):

    def __init__(self):
        self.length = 32
        self.repetitions = 16
        self.input_size = self.length // 2

    def __len__(self):
        return self.length * self.repetitions

    def __getitem__(self, idx):
        input_tensor, tags, output_tensor = self.generate_pattern()
        for i in range(self.repetitions - 1):
            new_input_tensor, new_tags, new_output_tensor = \
                self.generate_pattern()
            input_tensor = torch.cat((input_tensor[:-1], new_input_tensor), 0)
            tags = torch.cat((tags[:-1], new_tags), 0)
            output_tensor = torch.cat(
                (output_tensor[:-1], new_output_tensor), 0
                )

        return input_tensor, tags, output_tensor

    def generate_pattern(self):
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
