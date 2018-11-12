import torch
import torch.nn as nn
import constants as const

from torch.utils.data import Dataset
from helpers.dataset import pianoroll_dataset_batch
from helpers.dummy_dataset import DummyDataset
from models.RNN import RNN

fs1_rolls = 'datasets/training/piano_roll_fs1/'
fs2_rolls = 'datasets/training/piano_roll_fs2/'
fs5_rolls = 'datasets/training/piano_roll_fs5/'
csv_dir = fs1_rolls


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

            for step_start in range(0, song_length - 1, const.STEP):
                for t in range(step_start, step_start + const.STEP):
                    x_seq = input_tensors[t]
                    y_t = output_tensors[t]

                    output, hidden = model(x_seq.view(1, -1), hidden)

                loss = criterion(output, y_t)
                # print(t, "/", song_length, f"{i}/{len(dataset)}")
                # print("Input:", x_seq)
                # print("Target:", y_t)
                # print("Output", output)
                print("Loss:", loss)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            print("Params", i, epoch)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(name, param.data)


def compose(model: nn) -> None:
    with torch.no_grad():
        pass


if __name__ == "__main__":
    dataset = DummyDataset()
    # dataset = get_training_data_loader()
    # Index first song tuple, then input, then the final input dim
    input_size = output_size = dataset[0][0][0][0].size(0)

    model = RNN(input_size, const.HIDDEN_SIZE, output_size)
    model = train_model(model, dataset, num_epochs=1)
