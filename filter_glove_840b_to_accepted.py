"""
take raw 840b glove vector file
filter such that only lines for accepted words remain
"""

import csv

import pandas as pd
import numpy as np
import functools
import logging

PDF = pd.DataFrame
NDARR = np.ndarray

# see here https://stackoverflow.com/a/38537983
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
info = logging.info


def main() -> None:
    accepted_hints_path: str = "./accepted_hints.txt"
    with open(accepted_hints_path, "r") as f:
        accepted_hints: set = set(f.read().splitlines())

    glove_path: str = "./embeddings/glove/840b/glove.840B.300d.txt"
    filtered_path: str = glove_path + ".python_filtered"
    with open(glove_path, "r") as fin, open(filtered_path, "w") as fout:
        for i, line in enumerate(fin):
            if (i + 1) % 1e4 == 0:
                info(f"processing line {i+1}")
            word, _, _ = line.partition(" ")
            if word.lower() in accepted_hints:
                fout.write(line)

    info("all done")


if __name__ == "__main__":
    main()
