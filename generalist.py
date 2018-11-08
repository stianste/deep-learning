import constants as const
import torch.nn as nn

from helpers.dataset import pianoroll_dataset_batch
from torch.utils.data import DataLoader

fs1_rolls = 'datasets/training/piano_roll_fs1/'
fs2_rolls = 'datasets/training/piano_roll_fs2/'
fs5_rolls = 'datasets/training/piano_roll_fs5/'
csv_dir = fs1_rolls


def get_training_data_loader(directory=csv_dir):
    dataset_batch = pianoroll_dataset_batch(fs1_rolls) 
    train_loader = DataLoader(
        dataset=dataset_batch,
        batch_size=const.BATCH_SIZE,
        shuffle=False
    )

    return train_loader


def train_model() -> nn:
    data_loader = get_training_data_loader()

    for epoch in range(const.NUM_EPOCHS):
        for i, sample_batch in enumerate(data_loader):
            print(sample_batch)


def compose(model: nn) -> None:
    pass


if __name__ == "__main__":
    # TODO: Setup

    model = train_model()
