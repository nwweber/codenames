"""
just playing around a bit
"""
import csv
import logging
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

PDF = pd.DataFrame
NDARR = np.ndarray

# see here https://stackoverflow.com/a/38537983
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
info = logging.info


TEST_CASES_2W: list = [
    ["table", "chair"],  # furniture
    ["pilot", "airplane"],  # vehicle?
    ["apple", "banana"],  # fruit, food
    ["window", "door"],  # house, building
    ['train', 'car'],  # vehicle?
    ['cake', 'table'],  # meal
    ['cake', 'cookie'],  # dessert? pastry?
    ['cookie', 'croissant'],  # pastry?
    ['croissant', 'brie'],  # france?
    ['brie', 'gouda'],  # cheese
    ['gouda', 'tulip'],  # netherlands
    ['tulip', 'rose'],  # flower
]


TEST_CASES_3W: list = [
    ["table", "chair", 'couch'],  # furniture
    ["pilot", "airplane", 'helicopter'],  # vehicle?
    ["apple", "banana", 'strawberry'],  # fruit, food
    ["window", "door", 'roof'],  # house, building
    ['train', 'car', 'bus'],  # vehicle?
    ['cake', 'table', 'tea'],  # meal
    ['cake', 'cookie', 'pudding'],  # dessert? pastry?
    ['cookie', 'croissant', 'brownie'],  # pastry?
    ['croissant', 'brie', 'wine'],  # france?
    ['brie', 'gouda', 'roquefort'],  # cheese
    ['gouda', 'tulip', 'canal'],  # netherlands
    ['tulip', 'rose', 'orchid'],  # flower
]

# sort to make sure related cases are next to each other
ALL_TEST_CASES = sorted(TEST_CASES_2W + TEST_CASES_3W)


def generate_hints(riddle: list, glove: PDF, k: int = 5) -> list:
    """
    for a riddle, i.e. a bunch of words, generate best hints we can come up, ordered by goodness
    :param k: int > 0 nr of best hints to return
    :param glove: pandas DF of glove vectors, indexed by word
    :param riddle:
    :return:
    """
    # can we find some vectors for this?
    vecs: PDF = glove.loc[riddle]

    # average of those
    vec_avg = vecs.mean()

    # filter glove words to only contain legal hints
    # hint can not be:
    # * one of the riddle words
    # * a substring of one of the riddle words, i.e. hint cannot be 'cake' for riddle 'cheesecake'
    # * a superstring of one of the riddle words, i.e. hint cannot be 'cheesecake' for riddle 'cake'

    # initially, all hints are viable
    hint_mask = np.repeat(True, len(glove))
    # for each word contained in riddle: only leave hints (glove words) that are neither super- nor sub-strings of
    # the riddle word (including full match)
    for riddle_word in riddle:
        # riddle word is not substring of hints, includes case of hint == riddle
        not_substring: NDARR = ~glove.index.str.contains(riddle_word)
        # hints are not substring of riddle word, e.g. illegal: hint: 'cake' for riddle: 'cheesecake'
        bool_ix = []
        for hint_candidate in glove.index:
            bool_ix.append(hint_candidate not in riddle_word)
        not_superstring = np.asarray(bool_ix)
        # combine with existing mask
        hint_mask = hint_mask & not_superstring & not_substring

    glove_legal_hints = glove[hint_mask]

    # now to back-translate that into a word, find closest glove vec to this
    # euclidean distances
    dists: NDARR = np.linalg.norm(
        glove_legal_hints.values.astype(np.float16) - vec_avg.values.astype(np.float16),
        axis=1,
    )

    # noinspection PyUnresolvedReferences
    hint_ix: NDARR = np.argsort(dists)[:k]
    hint: list = list(glove_legal_hints.iloc[hint_ix].index)

    return hint


def main() -> None:
    # glove_path: str = "./embeddings/glove/6b/glove.6B.300d.txt"
    # glove_path: str = "./embeddings/glove/840b/glove.840B.300d.txt"
    glove_path: str = "./embeddings/glove/840b/glove.840B.300d.txt.python_filtered"
    glove_raw: PDF = pd.read_csv(
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

    # write this out for other uses
    accepted_hints: set = wbstr_nouns.union(gr_n_l_nouns).union(codewords)
    # artifact from webster parsing upstream
    accepted_hints.remove("")
    with open("./accepted_hints.txt", "w") as f:
        f.writelines("\n".join(accepted_hints))

    glv_wrds_accept_set: set = set(glove_raw.index).intersection(accepted_hints)
    # TODO words in Glove are Case Sensitive, but my 'accepted hints' are not, cutting out too many words
    glove_filtered = glove_raw.loc[glv_wrds_accept_set]

    hints = []
    for riddle in ALL_TEST_CASES:
        info(f'solving riddle {riddle}')
        try:
            hint = generate_hints(riddle, glove_filtered)
        except KeyError:
            hint = []
            info("not all words of riddle contained in glove vecs, can't generate hint")
        hints.append(hint)

    overview: PDF = pd.DataFrame(data=zip(ALL_TEST_CASES, hints), columns=['riddle', 'hints'])
    info(overview)


if __name__ == "__main__":
    main()
