"""
just playing around a bit
"""
import csv

import pandas as pd
import numpy as np
import functools

PDF = pd.DataFrame
NDARR = np.ndarray


def main() -> None:
    glove_path: str = "./embeddings/glove/6b/glove.6B.300d.txt"
    # glove_path: str = "./embeddings/glove/840b/glove.840B.300d.txt"
    glove: PDF = pd.read_csv(
        filepath_or_buffer=glove_path,
        delim_whitespace=True,
        engine="c",
        header=None,
        index_col=0,
        # nrows=100000,
        quoting=csv.QUOTE_NONE,
    )

    # can't do this while reading, ignored by csv parser, see here:
    # https://stackoverflow.com/questions/24761122/pandas-read-csv-ignoring-column-dtypes-when-i-pass-skip-footer-arg
    dtypes: dict = {i: np.float16 for i in range(1, len(glove.columns) + 1)}
    glove = glove.astype(dtypes)

    # nouns in the english language
    en_noun_path: str = './english_noun_lists/great_noun_list/nounlist.txt'
    with open(en_noun_path, 'r') as f:
        # lowercase to match glove
        # read().splitlines() instead of readlines() to get rid of
        # trailing newlines
        en_nouns: list = [word.lower() for word in f.read().splitlines()]

    nouns_with_glove = set(en_nouns).intersection(set(glove.index))
    glove = glove.loc[nouns_with_glove]

    wordlist_ids = ["ceadmilefailte", "default", "duet", "thegamegal", "thegamegal"]

    codewords: set = set()
    for wordlist_id in wordlist_ids:
        wordlist_path_template: str = "./wordlists/sagelga/wordlist/en-EN/{}/wordlist.txt"
        wl_path: str = wordlist_path_template.format(wordlist_id)
        with open(wl_path, "r") as f:
            # lowercase to match glove
            # read().splitlines() instead of readlines() to get rid of
            # trailing newlines
            words: set = set([word.lower() for word in f.read().splitlines()])

        codewords = codewords.union(words)

    codewords_l: list = list(codewords)
    rand_state = np.random.RandomState(0)

    for i in range(5):
        # make some riddles
        riddle: NDARR = rand_state.choice(codewords_l, 2, replace=False)

        # can we find some vectors for this?
        try:
            vecs: PDF = glove.loc[riddle]
        except KeyError as e:
            print("not all words of riddle contained in glove vecs")
            print(f"riddle: {riddle}")
            continue

        # average of those
        vec_avg = vecs.mean()

        # don't want a riddle word to be a possible hint
        glove_no_riddle = glove[~glove.index.isin(riddle)]

        # now to back-translate that into a word, find closest glove vec to this
        # euclidean distances
        dists: NDARR = np.linalg.norm(
            glove_no_riddle.values.astype(np.float16) - vec_avg.values.astype(np.float16), axis=1
        )
        hint_ix: int = dists.argmin()
        hint: str = glove_no_riddle.iloc[hint_ix].name

        print(f"riddle: {riddle}")
        print(f"hint: {hint}")


if __name__ == "__main__":
    main()
