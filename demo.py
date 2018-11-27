import torch
import torch.nn as nn
import pretty_midi
import constants as const
from helpers.datapreparation import (
    gen_music_pianoroll,
    visualize_piano_roll,
    midfile_to_piano_roll
)
from generalist import single_composition, get_training_data_loader


def load_model(model_name: str) -> nn.Module:
    return torch.load(f'{const.PRETRAINED_MODELS_PATH}{model_name}')


def visualize_song(song_filepath: str):
    pm = pretty_midi.PrettyMIDI(song_filepath)
    piano_roll = pm.get_piano_roll(const.FS)
    visualize_piano_roll(piano_roll, const.FS)


if __name__ == "__main__":
    # visualize_song("./compositions/demos/[short]v6_c12_l1_e50_s200.mid")
    visualize_song("./compositions/demos/v8_c12_l2_e100_s200.mid")
    model = load_model("2018-11-26 23:30:42.765568_specialized__c_l2_e100_s200.pth")
    dataset = get_training_data_loader()
    folder_prefix = "compositions/live_demos/"
    for i in range(4):
        single_composition(model, dataset, "", folder_prefix, 12, True, i)
