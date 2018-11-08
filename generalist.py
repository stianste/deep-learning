# import torch
import torch.nn as nn
import constants as const

from torch.utils.data import Dataset
from helpers.dataset import pianoroll_dataset_batch
from models.RNN import RNN

fs1_rolls = 'datasets/training/piano_roll_fs1/'
fs2_rolls = 'datasets/training/piano_roll_fs2/'
fs5_rolls = 'datasets/training/piano_roll_fs5/'
csv_dir = fs1_rolls


def get_training_data_loader(directory=csv_dir):
    return pianoroll_dataset_batch(fs1_rolls)


def train_model(model: nn, dataset: Dataset) -> nn:
    hidden = model.init_hidden()
    criterion = const.LOSS_FUNCTION

    for epoch in range(const.NUM_EPOCHS):
        for i, song in enumerate(dataset):
            input_tensors, tags, output_tensors = song

            model.zero_grad()

            for t in range(input_tensors.size(0)):
                x_t = input_tensors[t]
                y_t = output_tensors[t]

                output, hidden = model(x_t, hidden)
            loss = criterion(output, y_t.float())

            print("Loss:", loss)
            loss.backward(retain_graph=True)

            for p in model.parameters():
                p.data.add_(-const.LEARNING_RATE, p.grad.data)


def compose(model: nn) -> None:
    pass


if __name__ == "__main__":
    dataset = get_training_data_loader()
    # Index first song tuple, then input, then the final input dim
    input_size = output_size = dataset[0][0][0][0].size(0)

    model = RNN(input_size, const.HIDDEN_SIZE, output_size)
    model = train_model(model, dataset)
