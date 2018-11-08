import torch
import torch.nn as nn
import constants as const

from helpers.dataset import pianoroll_dataset_batch

fs1_rolls = 'datasets/training/piano_roll_fs1/'
fs2_rolls = 'datasets/training/piano_roll_fs2/'
fs5_rolls = 'datasets/training/piano_roll_fs5/'
csv_dir = fs1_rolls


def get_training_data_loader(directory=csv_dir):
    return pianoroll_dataset_batch(fs1_rolls)


def train_model() -> nn:
    dataset = get_training_data_loader()

    for epoch in range(const.NUM_EPOCHS):
        for i, song in enumerate(dataset):
            input_tensors, tags, output_tensors = song


def compose(model: nn) -> None:
    pass


if __name__ == "__main__":
    # TODO: Setup

    model = train_model()
