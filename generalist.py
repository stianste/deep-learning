import torch
import torch.nn as nn
import datetime
import constants as const

from torch.utils.data import Dataset
from helpers.dataset import pianoroll_dataset_batch
from helpers.datapreparation import (
    piano_roll_to_mid_file,
    gen_music_pianoroll,
)
from models.LSTM import LSTM

fs_rolls = f'datasets/training/piano_roll_fs{const.FS}/'
csv_dir = fs_rolls


def get_training_data_loader(directory: str = csv_dir):
    return pianoroll_dataset_batch(directory)


def _get_filename(version: str, song_nr: int, extension: str = "mid") -> str:
    timestamp = datetime.datetime.utcnow()
    filename = (f"{timestamp}_{version}_c{song_nr}_"
                f"l{const.NUM_HIDDEN_LAYERS}"
                f"_e{const.NUM_EPOCHS}"
                f"_s{const.SEQ_LEN}.{extension}")

    return filename


def train_model(model: nn, dataset: Dataset,
                num_epochs: int = 10, specialize: bool = False) -> nn:

    loss_function = const.LOSS_FUNCTION
    optimizer = const.OPTIMIZER(model.parameters(), lr=const.LEARNING_RATE)

    for epoch in range(num_epochs):
        for i, song in enumerate(dataset):
            input_tensors, tag, output_tensors = song
            hidden = None if specialize else model.init_hidden()

            song_length = len(input_tensors)
            song_losses = []
            for t in range(1, song_length - 1, const.SEQ_LEN):
                roof = min(t + const.SEQ_LEN, song_length - 1)
                x_seq = input_tensors[t:roof]
                y_t = output_tensors[t:roof]

                model.zero_grad()
                output, hidden = model(x_seq, hidden, tag)

                loss = loss_function(output, y_t.view(roof-t, -1))
                song_losses.append(loss.item())
                loss.backward(retain_graph=True)
                optimizer.step()

            print("Epoch", epoch+1, "Song", i+1, "/", len(dataset))
            print("Avg loss for this song", sum(song_losses)/len(song_losses))

    return model


def single_composition(model: nn.Module, 
                       dataset: Dataset,
                       version: str,
                       prefix: str,
                       song_nr: int = 0,
                       specialize: bool = False,
                       composer: int = -1) -> None:

    filename = _get_filename(version, song_nr)
    filename = prefix + filename

    if composer == -1:
        composer = dataset[song_nr][1]

    init = dataset[song_nr][0][1:50]

    piano_roll = gen_music_pianoroll(model,
                                     init=init,
                                     composer=composer,
                                     specialize=specialize)

    full_path = piano_roll_to_mid_file(piano_roll * 100,
                                       filename, fs=const.FS)
    print(f"Saved file to {full_path}")


def compose(model: nn.Module, dataset: Dataset,
            version: str, prefix: str,
            specialize: bool = False) -> None:

    for song_nr in range(len(dataset)):
        single_composition(model, dataset, version,
                           prefix, song_nr, specialize)


def save_model(model: nn.Module, filename: str = None, specialized: bool = False) -> None:
    if not filename:
        if specialized:
            filename = _get_filename("specialized", "", "pth")
        else:
            filename = _get_filename("", "", "pth")

    torch.save(model, f'{const.PRETRAINED_MODELS_PATH}{filename}')


def main(model_type: object) -> nn.Module:
    dataset = get_training_data_loader()
    # Index first song tuple, then input, then the final input dim
    input_size = output_size = dataset[0][0][0][0].size(0)
    model = model_type(input_size, const.HIDDEN_SIZE,
                       output_size, num_layers=const.NUM_HIDDEN_LAYERS,
                       dropout=const.DROPOUT)
    model = train_model(model, dataset, num_epochs=const.NUM_EPOCHS)

    return model, dataset


if __name__ == "__main__":
    model, dataset = main(LSTM)
    compose(model, dataset, "v7", "compositions/generalist/")
