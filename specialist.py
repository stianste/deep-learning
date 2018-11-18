import torch.nn as nn
import constants as const
from torch.utils.data import Dataset
from generalist import main, train_model, compose
from models.LSTM import LSTMSpecialist


def specialize_model(model: nn, dataset: Dataset,
                     num_epochs: int = 10) -> nn.Module:
    return train_model(model, dataset, num_epochs, True)


if __name__ == "__main__":
    model, dataset = main(LSTMSpecialist)
    print("Now specializing")
    model = specialize_model(model, dataset,
                             num_epochs=const.NUM_EPOCHS // 4)
    compose(model, dataset, "comp", "specialized_compositions/", specialize=True)
