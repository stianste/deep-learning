import torch
import torch.nn as nn
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


def compose(model: nn.Module, dataset: Dataset,
            version: str, prefix: str, specialize: bool = False) -> None:
    for song_nr in range(len(dataset)):
        timestamp = datetime.datetime.utcnow()
        filename = (f"{version}_c{song_nr}_l{const.NUM_HIDDEN_LAYERS}"
                    f"_e{const.NUM_EPOCHS}"
                    f"_s{const.SEQ_LEN}_{timestamp}.mid")
        filename = prefix + filename

        composer = dataset[song_nr][1].item()
        piano_roll = gen_music_pianoroll(model, init=dataset[song_nr][0][1:50],
                                         composer=composer,
                                         specialize=specialize)
        print(piano_roll)
        print(piano_roll.shape)

        full_path = piano_roll_to_mid_file(piano_roll * 100,
                                           filename, fs=const.FS)
        print(f"Saved file to {full_path}")


def main(model_type: object) -> nn.Module:
    dataset = get_training_data_loader()
    # Index first song tuple, then input, then the final input dim
    input_size = output_size = dataset[0][0][0][0].size(0)
    model = model_type(input_size, const.HIDDEN_SIZE,
                       output_size, num_layers=const.NUM_HIDDEN_LAYERS)
    model = train_model(model, dataset, num_epochs=const.NUM_EPOCHS)

    return model, dataset


if __name__ == "__main__":
    model, dataset = main(LSTM)
    compose(model, dataset, "v6", "compositions/")
