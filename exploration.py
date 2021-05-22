"""
just playing around a bit
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

    # words (mostly nouns though) in the english language
    # great noun list
    # http://www.desiquintans.com/nounlist
    gr_n_l_path: str = "./english_noun_lists/great_noun_list/nounlist.txt"
    with open(gr_n_l_path, "r") as f:
        # lowercase to match glove
        # read().splitlines() instead of readlines() to get rid of
        # trailing newlines
        gr_n_l_nouns: set = set([word.lower() for word in f.read().splitlines()])

    # webster's unabridged
    # https://www.gutenberg.org/ebooks/author/139
    wbstr_pth: str = "./english_noun_lists/webster_unabridged/extracted_nouns.txt"
    with open(wbstr_pth, "r") as f:
        wbstr_nouns: set = set([word.lower() for word in f.read().splitlines()])

    # dwyl english word list
    # https://github.com/dwyl/english-words/
    dywl_pth: str = "./english_noun_lists/dwyl_english_words/words_alpha.txt"
    with open(dywl_pth, "r") as f:
        dwyl_words: set = set([word.lower() for word in f.read().splitlines()])

    # overlap between various word collections and glove
    glove_wrds = set(glove.index)
    len(glove_wrds)
    len(wbstr_nouns)
    len(wbstr_nouns.intersection(glove_wrds))
    len(gr_n_l_nouns)
    len(gr_n_l_nouns.intersection(glove_wrds))
    len(gr_n_l_nouns.intersection(wbstr_nouns))
    len(dwyl_words)
    # only ~100k glove words left after filtering by dwyl. seems like a reasonable reduction
    len(dwyl_words.intersection(glove_wrds))
    len(dwyl_words.intersection(gr_n_l_nouns))
    len(dwyl_words.intersection(wbstr_nouns))
    # conclusion: dwyl contains basically words from great noun list and webster
    # question: does it contain additional useful words though?


    # glv_wrds_accept_set: set = set(glove.index).intersection(dwyl_words.union(wbstr_nouns).union(gr_n_l_nouns))
    glv_wrds_accept_set: set = set(glove.index).intersection(wbstr_nouns.union(gr_n_l_nouns))
    glove_filtered = glove.loc[glv_wrds_accept_set]

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
            vecs: PDF = glove_filtered.loc[riddle]
        except KeyError as e:
            print("not all words of riddle contained in glove vecs")
            print(f"riddle: {riddle}")
            continue

        # average of those
        vec_avg = vecs.mean()

        # don't want a riddle word to be a possible hint
        glove_no_riddle = glove_filtered[~glove_filtered.index.isin(riddle)]

        # now to back-translate that into a word, find closest glove vec to this
        # euclidean distances
        dists: NDARR = np.linalg.norm(
            glove_no_riddle.values.astype(np.float16)
            - vec_avg.values.astype(np.float16),
            axis=1,
        )
        hint_ix: int = dists.argmin()
        hint: str = glove_no_riddle.iloc[hint_ix].name

        print(f"riddle: {riddle}")
        print(f"hint: {hint}")


if __name__ == "__main__":
    main()
