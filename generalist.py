import torch
import torch.nn as nn
import numpy as np
import datetime
import constants as const

from torch.utils.data import Dataset
from helpers.dataset import pianoroll_dataset_batch
from helpers.datapreparation import piano_roll_to_mid_file, gen_music_pianoroll
from models.LSTM import LSTM

fs_rolls = f'datasets/training/piano_roll_fs{const.FS}/'
csv_dir = fs_rolls


def get_training_data_loader(directory: str = csv_dir):
    return pianoroll_dataset_batch(directory)


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
    loss_function = const.LOSS_FUNCTION
    optimizer = const.OPTIMIZER(model.parameters(), lr=const.LEARNING_RATE)

    for epoch in range(num_epochs):
        for i, song in enumerate(dataset):
            input_tensors, tags, output_tensors = song
            model.reset_hidden()

            song_length = len(input_tensors)
            song_losses = []
            for t in range(1, song_length - 1, const.SEQ_LEN):
                model.zero_grad()

                roof = min(t + const.SEQ_LEN, song_length - 1)
                x_seq = input_tensors[t:roof]
                y_t = output_tensors[t:roof]

                output = model(x_seq, None)

                loss = loss_function(output, y_t.view(roof-t, -1))
                song_losses.append(loss.item())
                loss.backward(retain_graph=True)
                optimizer.step()

            print("Epoch", epoch+1, "Song", i+1, "/", len(dataset))
            print("Avg loss for this song", sum(song_losses)/len(song_losses))

    return model


def generate_from_song(
        model: nn.Module,
        song: torch.Tensor,
        init_length: int = 100,
        gen_length: int = 1000) -> np.array:

    init = song[1:init_length]  # [1000, 1, 128] -> [100, 1, 128]
    generated_song = init.squeeze().numpy()

    with torch.no_grad():
        model.reset_hidden()
        output = model(init, None)
        generated_song = np.concatenate((generated_song, output))

        for t in range(gen_length):
            output = model(output.view(-1, 1, const.INPUT_SIZE), None)
            generated_song = np.concatenate((generated_song, output))

    return np.round((generated_song/np.max(generated_song))).T


def compose(model: nn.Module, dataset: Dataset) -> None:
    for song_nr in range(len(dataset)):
        timestamp = datetime.datetime.utcnow()
        filename = (f"v5_c{song_nr}_l{const.NUM_HIDDEN_LAYERS}"
                    f"_e{const.NUM_EPOCHS}"
                    f"_s{const.SEQ_LEN}_{timestamp}.mid")
        filename = "compositions/" + filename

        piano_roll = gen_music_pianoroll(model, init=dataset[song_nr][0][1:50])
        print(piano_roll)
        print(piano_roll.shape)

        full_path = piano_roll_to_mid_file(piano_roll * 100,
                                           filename, fs=const.FS)
        print(f"Saved file to {full_path}")


if __name__ == "__main__":
    # dataset = DummyDataset()
    dataset = get_training_data_loader()
    # Index first song tuple, then input, then the final input dim
    input_size = output_size = dataset[0][0][0][0].size(0)

    # model = RNN(input_size, const.HIDDEN_SIZE, output_size)
    model = LSTM(input_size, const.HIDDEN_SIZE,
                 output_size, num_layers=const.NUM_HIDDEN_LAYERS)
    model = train_model(model, dataset, num_epochs=const.NUM_EPOCHS)

    compose(model, dataset)
