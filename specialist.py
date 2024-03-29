import torch.nn as nn
import constants as const
from torch.utils.data import Dataset
from generalist import main, train_model, compose, save_model
from models.LSTM import LSTMSpecialist


def specialize_model(model: nn, dataset: Dataset,
                     num_epochs: int = 10) -> nn.Module:
    return train_model(model, dataset, num_epochs, True)


if __name__ == "__main__":
    model, dataset = main(LSTMSpecialist)
    compose(
        model,
        dataset,
        "v7",
        "compositions/generalist/__temp",
        specialize=False
    )
    save_model(model)

    print("Now specializing")
    model = specialize_model(model, dataset,
                             num_epochs=const.NUM_EPOCHS // 4)
    compose(
        model,
        dataset,
        "v2",
        "compositions/specialized/__temp",
        specialize=True
    )
    save_model(model, filename=None, specialized=True)
