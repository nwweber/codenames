"""
take html version of gutenberg webster's unabridged dictionary, parse out
nouns and write to stand-alone file
"""
import re
import logging
import glob
import bs4

Tag = bs4.element.Tag

# see here https://stackoverflow.com/a/38537983
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
info = logging.info


def is_noun_entry(p_tag: Tag) -> bool:
    """
    <p> tag is a noun dictionary entry if 2nd child is "<i>n.</i>"
    :param p_tag:
    :return:
    """
    try:
        sec_child = list(p_tag.children)[1]
    # sometimes there are fewer than 2 children
    except IndexError:
        return False

    return (sec_child.name == "i") and (sec_child.contents[0] == "n.")


def noun_tag_to_clean_noun(noun_tag: Tag) -> str:
    """
    get contents of tag and remove a smattering of noise to get the actual
    word out
    :param noun_tag:
    :return:
    """
    cleaned = noun_tag.contents[0].strip().lower()
    # remove (parenthetical explanations of words)
    explanation_pattern = re.compile(r"(.+) \(.*\)")
    match = explanation_pattern.match(cleaned)
    if match is not None:
        cleaned = match.group(1)
    # remove ', alternative spelling of word'
    cleaned = cleaned.split(",")[0]
    # encode spaces as underscore so they survive next steps
    space_pattern = re.compile(r" ")
    cleaned = space_pattern.sub("_", cleaned)
    # remove all non-letter characters
    non_letter_pattern = re.compile(r"\W")
    cleaned = non_letter_pattern.sub("", cleaned)
    # re-instate previous whitespaces
    underscore_pattern = re.compile(r"_")
    cleaned = underscore_pattern.sub(" ", cleaned)
    cleaned = cleaned.strip()
    return cleaned


def wbstr_fl_pth_to_noun_set(wbstr_fl_pth: str) -> set:
    """
    take path to webster dictionary .htm file, return set of nouns found
    in that file
    :param wbstr_fl_pth:
    :return:
    """
    with open(wbstr_fl_pth, "r") as f:
        webster_soup = bs4.BeautifulSoup(f, "html.parser")

    p_tags: list = webster_soup.find_all("p")
    noun_tags: list = [pt for pt in p_tags if is_noun_entry(pt)]
    # multiple meanings of same-spelled word give duplicates, fix here
    webster_nouns = set([noun_tag_to_clean_noun(nt) for nt in noun_tags])
    return webster_nouns


def main() -> None:
    webster_path: str = "./english_noun_lists/webster_unabridged/*.htm"
    wbstr_fl_pths: list = glob.glob(webster_path)

    wbstr_nouns: set = set()
    for i, wbstr_fl_pth in enumerate(wbstr_fl_pths):
        info(f"processing file ({i + 1}/{len(wbstr_fl_pths)}) {wbstr_fl_pth}")
        noun_set = wbstr_fl_pth_to_noun_set(wbstr_fl_pth)
        wbstr_nouns = wbstr_nouns.union(noun_set)
        info(f"done, found {len(noun_set)} nouns")

    info(f"found {len(wbstr_nouns)} nouns in total")

    out_fname = "./english_noun_lists/webster_unabridged/extracted_nouns.txt"
    with open(out_fname, "w") as f:
        f.writelines("\n".join(sorted(wbstr_nouns)))

    info(f"written nouns to {out_fname}")
    info("all done")


if __name__ == "__main__":
    main()
