"""
Let's play some Codenames :)
"""
import csv
import dataclasses
import enum
import sys
import typing
from pathlib import Path
from typing import List, TypeAlias

import numpy as np
import pandas as pd
import tabulate
from pandas import DataFrame

rng = np.random.default_rng(42)


def is_interactive_py_session() -> bool:
    """
    Tests if we are running in an interactive session or not.

    Based on https://stackoverflow.com/a/64523765
    Returns
    -------
        a boolean indicator
    """
    return hasattr(sys, "ps1")


def load_glove(glove_path: Path) -> DataFrame:
    glove: DataFrame = pd.read_csv(
        filepath_or_buffer=glove_path,
        delim_whitespace=True,
        engine="c",
        header=None,
        index_col=0,
        quoting=csv.QUOTE_NONE,
    )
    # can't do this while reading, ignored by csv parser, see here:
    # https://stackoverflow.com/questions/24761122/pandas-read-csv-ignoring-column-dtypes-when-i-pass-skip-footer-arg
    dtypes: dict[int, type] = {i: np.float16 for i in range(1, len(glove.columns) + 1)}
    # mypy is happy and it works at runtime, disable pycharm's type checker
    # noinspection PyTypeChecker
    glove = glove.astype(dtypes)
    return glove


def make_embeddings(glove_path: Path, codenames_words_path: Path) -> DataFrame:
    glove_vectors = load_glove(glove_path)
    codenames_words = load_codenames_words(codenames_words_path)
    words_in_common: List[str] = sorted(
        set(codenames_words).intersection(set(glove_vectors.index))
    )
    # the words and their embeddings which we actually want to use
    embeddings: DataFrame = glove_vectors.loc[words_in_common]
    embeddings.index.name = "word"
    return embeddings


def load_codenames_words(codenames_words_path: Path) -> List[str]:
    """
    Parses wordlist indicated by path into lowercase, stripped list of words.
    """
    with codenames_words_path.open("r") as f:
        codenames_words_raw: List[str] = f.readlines()
    codenames_words: List[str] = [w.strip().lower() for w in codenames_words_raw]
    return codenames_words


def strikethrough(text: str) -> str:
    """
    source:
    https://stackoverflow.com/a/25244576
    """
    result = ""
    for c in text:
        result = result + c + "\u0336"
    return result


class CardType(enum.Enum):
    AGENT = "agent"
    NEUTRAL = "neutral"
    ASSASSIN = "assassin"


@dataclasses.dataclass
class Card:
    word: str
    type: CardType
    is_played: bool


@dataclasses.dataclass
class Hint:
    word: str
    number: int


# None indicates that guesser wants to stop guessing
# integer indicates an index of a card on the playing board
Guess: TypeAlias = typing.Optional[int]


class Spymaster:
    def generate_hint(self) -> Hint:
        return Hint(word="bob", number=2)


class Board:
    """
    This class is responsible for maintaining the current board state and methods
    related to that.
    """
    def __init__(self, cards: list[Card]):
        self.cards = cards

    def display(self) -> None:
        """Pretty prints the contents of the board. No information is hidden."""
        def card_to_str(card: Card) -> str:
            """
            Formats a Card to be displayed in a grid cell of a text-based table.
            """
            card_type_str: str
            match card.type:
                case CardType.AGENT:
                    card_type_str = 'A'
                case CardType.NEUTRAL:
                    card_type_str = 'N'
                case CardType.ASSASSIN:
                    # unicode 'skull and crossbones'
                    card_type_str = '\u2620'
                case _:
                    card_type_str = '?'
            display_str = f"""{'* ' if card.is_played else ''}{card.word} [{card_type_str}]"""
            return display_str

        str_matrix: list[list[str]] = []
        str_per_row = 5
        for card_nr, card in enumerate(self.cards, start=1):
            if card_nr % str_per_row == 1:
                str_matrix.append([])
            str_matrix[-1].append(f"{card_to_str(card)} ({card_nr})")

        print(tabulate.tabulate(str_matrix, tablefmt='grid'))
        print('* = already played')

    def all_agents_found(self) -> bool:
        """
        Returns True iff all cards with type Agent have been played/found.
        """
        return all([card.is_played for card in self.cards if card.type == CardType.AGENT])


class Guesser:
    def generate_guess(self, board: Board, hint: Hint, is_yield_allowed: bool) -> Guess:
        """
        Prompts the user for input. If 'is_yield_allowed'. Output is an integer referring
        to an index of a card on the board or None if user wishes not to guess.

        None is only allowed if 'is_yield_allowed' is True.
        """
        print("This is what the board currently looks like:")
        board.display()
        print(f"This is your hint from the spymaster: '{hint.word}', {hint.number}")
        guess = input(
            f"what is your guess? type the number next to the word you wish to guess.{' Leave blank and hit Enter if you would like to not guess' if is_yield_allowed else ''}: "
        )
        if guess == "":
            return None
        # user sees cards numbered starting with 1, internally we start counting at 0
        card_index = int(guess) - 1
        return card_index


class CodeNamesGame:
    def __init__(self, spymaster: Spymaster, guesser: Guesser, board: Board):
        self.spymaster: Spymaster = spymaster
        self.guesser: Guesser = guesser
        self.board = board

    def handle_guess(self, guess: Guess) -> bool:
        """
        Deals with everything that might happen after a guess is made, which includes
        updating the board, winning or losing, or just basically nothing.

        Returns True if guesser is done guessing (by choice or due to wrong guess),
        False if we want to loop back to Guesser for another guess.
        """
        # None stands for 'guesser is allowed to guess more, but decides not to'
        if guess is None:
            return True
        assert isinstance(guess, int)
        card_index = guess
        guessed_card = self.board.cards[card_index]
        guessed_card.is_played = True
        print(f"You guessed '{guessed_card.word}'. It is a(n) {guessed_card.type.value} card.")
        match guessed_card.type:
            case CardType.ASSASSIN:
                self.lose()
                # return mostly here to satisfy type checking
                # in current implementation lose() ends the program
                return True
            case CardType.AGENT:
                if self.board.all_agents_found():
                    print('Well done! You have found the last agent and win!')
                    self.win()
                print('Well done! You can now choose to take another guess, or to '
                      'yield your turn and get new hint from the Spymaster.')
                return False
            case CardType.NEUTRAL:
                print('The Spymaster can now give another hint if there are turns left.')
                return True
        typing.assert_never(guessed_card.type)

    def lose(self) -> None:
        print('oh no, you lost :((((((')
        sys.exit()

    def play(self) -> None:
        turn_limit = 9
        turn_count = 1
        while turn_count <= turn_limit:
            print(f'\n==== Turn {turn_count} ===\n')
            hint: Hint = self.spymaster.generate_hint()
            guess_count = 1
            while True:
                guess: Guess = self.guesser.generate_guess(
                    board=self.board,
                    hint=hint,
                    is_yield_allowed=guess_count > 1
                )
                guess_count += 1
                stop_guess = self.handle_guess(guess)
                if stop_guess:
                    break
            turn_count += 1
        # we have reached the turn limit
        print('Turn limit reached')
        self.lose()

    def win(self) -> None:
        print('Sweet, sweet victory')
        sys.exit()


def make_random_board(words: list[str]) -> Board:
    """
    Generates a random board to play on.
    """
    n_cards = 25
    n_agents = 13
    n_assassins = 1
    card_words = rng.choice(words, n_cards)
    index_permutation = rng.permutation(np.arange(n_cards))
    agent_ix = set(index_permutation[:n_agents])
    assassin_ix = set(index_permutation[n_agents:(n_agents+n_assassins)])
    cards: list[Card] = []
    for i, word in enumerate(card_words):
        card_type: CardType
        if i in agent_ix:
            card_type = CardType.AGENT
        elif i in assassin_ix:
            card_type = CardType.ASSASSIN
        else:
            card_type = CardType.NEUTRAL
        cards.append(Card(
            word=word,
            type=card_type,
            is_played=False,
        ))
    return Board(cards)


def main() -> None:
    codenames_wordlist_loc = Path(__file__).parent.parent.parent.joinpath('data').joinpath('codenames-wordlist-eng.txt')
    with codenames_wordlist_loc.open('r') as f:
        codenames_words = [word.strip().lower() for word in f.readlines()]
    board = make_random_board(codenames_words)

    game = CodeNamesGame(spymaster=Spymaster(), guesser=Guesser(), board=board)
    game.play()

    # data_dir = Path("../data")
    # embeddings_path = data_dir / 'codenames-en-words-glove-6b-50d-vectors.feather'
    # embeddings: DataFrame
    # if embeddings_path.exists():
    #     print('word list and embedding cache exists, not recomputing it')
    #     embeddings = pd.read_feather(embeddings_path)
    # else:
    #     glove_path = data_dir / "glove.6B.50d.txt"
    #     codenames_words_path = data_dir / "codenames-wordlist-eng.txt"
    #     print('creating word list and embedding cache')
    #     embeddings = make_embeddings(glove_path, codenames_words_path)
    #     embeddings.to_feather(embeddings_path)


if __name__ == "__main__" and not is_interactive_py_session():
    main()
