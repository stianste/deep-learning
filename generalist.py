import torch
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


def train_model(model: nn, dataset: Dataset, num_epocs: int = 10) -> nn:
    hidden = model.init_hidden()
    criterion = const.LOSS_FUNCTION
    optimizer = const.OPTIMIZER(model.parameters(), lr=const.LEARNING_RATE)

    for epoch in range(const.NUM_EPOCHS):
        for i, song in enumerate(dataset):
            input_tensors, tags, output_tensors = song
            song_length = song[0].size(0)

            for t in range(0, song_length - 1):
                # print(t, "/", song_length, f"{i}/{len(dataset)}")
                x_seq = input_tensors[t:min(t + const.STEP, song_length - 1)]
                y_t = output_tensors[t]

                output, hidden = model(x_seq.view(1, 1, -1), hidden)
                print(output)

                loss = criterion(output, y_t.unsqueeze(0))
                # print("Loss:", loss)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)

                optimizer.step()
            print("Params", i)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(name, param.data)


def compose(model: nn) -> None:
    with torch.no_grad():
        pass


if __name__ == "__main__":
    dataset = get_training_data_loader()
    # Index first song tuple, then input, then the final input dim
    input_size = output_size = dataset[0][0][0][0].size(0)

    model = RNN(input_size, const.HIDDEN_SIZE, output_size)
    model = train_model(model, dataset, num_epocs=1)
