import sys
from pathlib import Path
from typing import List

import pandas as pd
from pandas import DataFrame
import csv
import numpy as np

def is_interactive_py_session() -> bool:
    """
    Tests if we are running in an interactive session or not.

    Based on https://stackoverflow.com/a/64523765
    Returns
    -------
        a boolean indicator
    """
    return hasattr(sys, 'ps1')


def load_glove(glove_path: Path) -> DataFrame:
    glove_raw: DataFrame = pd.read_csv(
        filepath_or_buffer=glove_path,
        delim_whitespace=True,
        engine="c",
        header=None,
        index_col=0,
        quoting=csv.QUOTE_NONE,
    )
    # can't do this while reading, ignored by csv parser, see here:
    # https://stackoverflow.com/questions/24761122/pandas-read-csv-ignoring-column-dtypes-when-i-pass-skip-footer-arg
    dtypes: dict = {i: np.float16 for i in range(1, len(glove_raw.columns) + 1)}
    glove_raw = glove_raw.astype(dtypes)
    return glove_raw


def main():
    data_dir = Path('../data')
    glove_vectors = load_glove(data_dir/'glove.6B.50d.txt')
    codenames_words_path = data_dir / 'codenames-wordlist-eng.txt'
    with codenames_words_path.open('r') as f:
        codenames_words_raw: List[str] = f.readlines()
    codenames_words: List[str] = [w.strip().lower() for w in codenames_words_raw]
    words_in_common: List[str] = sorted(set(codenames_words).intersection(set(glove_vectors.index)))
    # the words and their embeddings which we actually want to use
    words_n_vecs = glove_vectors.loc[words_in_common]
    # words and their vectors for cards placed on the board
    words_n_vecs_board = words_n_vecs
    # words and their vectors used for hints
    # in the future, likely want to have different list of words used for giving hints
    # than words that can be on the board. giving a different name now to keep concepts
    # apart
    words_n_vecs_hints = words_n_vecs

    # let's try actually playing a little
    # setup: set the board
    rgen = np.random.default_rng(42)
    n_cards = 25
    n_agents = 15
    board_words = rgen.choice(words_n_vecs_board.index.values, n_cards)
    special_positions = rgen.choice(np.arange(len(board_words)), n_agents + 1)
    agent_positions = special_positions[:n_agents]
    assassin_position = special_positions[-1]
    unplayed_positions = np.arange(n_cards)
    rounds_to_play = 9
    rounds_left = rounds_to_play
    valid_hints = set(words_n_vecs_hints.index.values) - set(board_words)
    terminated = False

    # round 1
    # spymaster gives hint
    # probably want to assume fixed value here for now
    connect_n_words = 2
    hint = rgen.choice(list(valid_hints), 1)[0]

    # guesser guesses
    # note: guess should probably not an index on the board
    # but instead an index of a word instead, so agent can learn which words to guess
    # regardless of where they are on the board
    guess_position = rgen.choice(unplayed_positions, 1)[0]

    # now update the board state according to guess
    if guess_position == assassin_position:
        print('you lose :(')
        terminated = True
    elif guess_position in agent_positions:
        print('hello :) great guess!')
        # how do i handle agent being allowed another guess here? hmmm
    else:
        print('this was a neutral card, nothing happens but can not guess again')

    # don't want to overwrite for now
    new_unplayed_possitions = np.delete(unplayed_positions, unplayed_positions == guess_position)
    new_unplayed_possitions


if __name__ == '__main__' and not is_interactive_py_session():
    main()
