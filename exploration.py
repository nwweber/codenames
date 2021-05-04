"""
just playing around a bit
"""

import pandas as pd
import numpy as np
import functools

PDF = pd.DataFrame
NDARR = np.ndarray


def main() -> None:
    glove_path: str = './embeddings/glove/6b/glove.6B.50d.txt'
    glove: PDF = pd.read_csv(
        filepath_or_buffer=glove_path,
        delim_whitespace=True,
        engine='python',
        header=None,
        index_col=0
    )

    wordlist_ids = [
        'ceadmilefailte',
        'default',
        'duet',
        'thegamegal',
        'thegamegal'
    ]

    codewords: set = set()
    for wordlist_id in wordlist_ids:
        wordlist_path_template: str = \
            './wordlists/sagelga/wordlist/en-EN/{}/wordlist.txt'
        wl_path: str = wordlist_path_template.format(wordlist_id)
        with open(wl_path, 'r') as f:
            # lowercase to match glove
            # read().splitlines() instead of readlines() to get rid of
            # trailing newlines
            words: set = set([word.lower() for word in f.read(
            ).splitlines()])

        codewords = codewords.union(words)

    codewords_l: list = list(codewords)
    rand_state = np.random.RandomState(0)

    for i in range(10):
        # make some riddles
        riddle: NDARR = rand_state.choice(codewords_l, 3, replace=False)

        # can we find some vectors for this?
        try:
            vecs: PDF = glove.loc[riddle]
        except KeyError as e:
            print("not all words of riddle contained in glove vecs")
            print(f"riddle: {riddle}")
            print(e)
            continue

        # average of those
        vec_avg = vecs.mean()

        # now to back-translate that into a word, find closest glove vec to this
        # euclidean distances
        dists: NDARR = np.linalg.norm(glove.values - vec_avg.values, axis=1)
        hint_ix: int = dists.argmin()
        hint: str = glove.iloc[hint_ix].name

        print(f"riddle: {riddle}")
        print(f"hint: {hint}")


if __name__ == '__main__':
    main()