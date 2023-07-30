import pathlib
import pickle

import numpy as np
import tree
from gym_sts.data.state_log_loader import StateLogLoader


def pickle_observations(input_folder_name: str, output_file_name: str, cutoff: int = 20_000):
    """
    Transforms gym-sts state logs into a column-major format that is useful for RL training.
    """

    loader = StateLogLoader()

    state_folder = pathlib.Path(input_folder_name)
    files = [f for f in state_folder.iterdir()]

    for f in files:
        loader.load_file(f.open())

        if len(loader.steps) >= cutoff:
            break

    steps = loader.steps[:cutoff]
    col_major = tree.map_structure(lambda *xs: np.stack(xs), *[s.dict() for s in steps])

    output_file = pathlib.Path(output_file_name)
    pickle.dump(col_major, output_file.open(mode="wb"))
