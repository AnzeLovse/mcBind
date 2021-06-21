import sys

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import h5py
import numpy as np
from keras import activations
from keras.models import model_from_json
from vis.utils import utils
from vis.visualization import visualize_saliency

from constants import experiment_set
from utils import get_positive_samples, reset_seeds

reset_seeds()


def append_to_h5py(protein, example, grads_sum):
    with h5py.File("results/set_0/{}/saliency.h5".format(protein), "a") as f:
        # Append sum of saliency over all layers
        dset = f[protein]
        dset.resize((example + 1, 101,))
        dset[example] = grads_sum


def swap_activation(protein):
    with open("results/set_0/{}/model.json".format(protein), "r") as json_file:
        json = json_file.read()
        model = model_from_json(json)
        model.load_weights("results/set_0/{}/weights.h5".format(protein))

        for layer_index, layer in enumerate(model.layers):
            # Swap softmax with linear
            model.layers[layer_index].activation = activations.linear
            model = utils.apply_modifications(model)

    return model


def save_saliency(protein, example, seed_input, model):
    layer_index = -1

    # Calculate saliency
    grads = visualize_saliency(
        model, layer_index, filter_indices=1, seed_input=seed_input
    )
    append_to_h5py(protein, example, grads)


protein = sys.argv[2]

positive_samples = get_positive_samples(protein, "test", 0, *experiment_set)

model = swap_activation(protein)

slice_n = int(float(sys.argv[1]))

for example in range(slice_n, 50 + slice_n):
    seed_input = np.expand_dims(positive_samples[example,], axis=0)
    save_saliency(protein, example, seed_input, model)

# for example, seed_input in enumerate(positive_samples):
#     time1 = time.time()
#     save_saliency(protein, example, seed_input, model)
#     time2 = time.time()
#     print('{} function took {} s'.format(example, (time2-time1)))
