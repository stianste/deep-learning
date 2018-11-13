import torch
import torch.nn as nn
import constants as const
import numpy as np
import datetime

from torch.utils.data import Dataset
from helpers.dataset import pianoroll_dataset_batch
from helpers.datapreparation import gen_music_pianoroll, piano_roll_to_mid_file
from helpers.dummy_dataset import DummyDataset
from models.RNN import RNN

fs1_rolls = 'datasets/training/piano_roll_fs1/'
fs2_rolls = 'datasets/training/piano_roll_fs2/'
fs5_rolls = 'datasets/training/piano_roll_fs5/'
csv_dir = fs5_rolls


def get_training_data_loader(directory: str = csv_dir):
    return pianoroll_dataset_batch(fs1_rolls)


def set_max_value_to_1(tensor: torch.Tensor) -> torch.Tensor:
    max_index = torch.argmax(tensor, 2).item()
    # TODO: This is very ad-hoc. Do properly later
    tensor = torch.zeros(1, 1, 128)
    tensor[0][0][max_index] = 1
    return tensor


def output_to_piano_keys(
        output: torch.Tensor,
        threshold: float = 0.6) -> torch.Tensor:

    above_threshold = output > threshold
    if above_threshold.sum() > 1:
        return above_threshold.float().requires_grad_()
    else:
        return set_max_value_to_1(output).float().requires_grad_()


def train_model(model: nn, dataset: Dataset, num_epochs: int = 10) -> nn:
    criterion = const.LOSS_FUNCTION
    optimizer = const.OPTIMIZER(model.parameters(), lr=const.LEARNING_RATE)

    for epoch in range(num_epochs):
        for i, song in enumerate(dataset):
            hidden = model.init_hidden()

            input_tensors, tags, output_tensors = song
            song_length = song[0].size(0)

            song_losses = []
            for step_start in range(0, song_length - 1, const.STEP):
                h = min(step_start + const.STEP, len(song))
                for t in range(step_start, h):
                    x_seq = input_tensors[t]
                    y_t = output_tensors[t]

                    output, hidden = model(x_seq, None, hidden)

                loss = criterion(output, y_t)
                # print("Input:", x_seq)
                # print("Target:", y_t)
                # print("Output:", output)
                # print("Loss:", loss)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                song_losses.append(loss.item())

            for name, param in model.named_parameters():
                if param.requires_grad:
                    if np.isnan(param.data[0][0][0]):
                        print("Parameter = nan. Returning model")
                        return model
        print("Epoch", epoch)
        print("Avg loss", sum(song_losses)/len(song_losses))

    return model


def compose(model: nn.Module) -> None:
    with torch.no_grad():
        composer = 0
        timestamp = datetime.datetime.utcnow()
        filename = f"compositions/{composer}_{timestamp}.mid"
        piano_roll = gen_music_pianoroll(model, composer=composer)
        print(piano_roll.shape)
        print(piano_roll * 100)

        full_path = piano_roll_to_mid_file(piano_roll * 100, filename)
        print(f"Saved file to {full_path}")


if __name__ == "__main__":
    dataset = DummyDataset()
    dataset = get_training_data_loader()
    # Index first song tuple, then input, then the final input dim
    input_size = output_size = dataset[0][0][0][0].size(0)

    model = RNN(input_size, const.HIDDEN_SIZE, output_size)
    model = train_model(model, dataset, num_epochs=const.NUM_EPOCHS)

    compose(model)
