"""
Let's play some Codenames :)
"""
import stable_baselines3
import csv
import dataclasses
import enum
import itertools
import sys
import typing
from pathlib import Path
from typing import List, TypeAlias

import gymnasium
import numpy as np
import numpy.typing as npt
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
    AGENT = 1, "agent"
    NEUTRAL = 2, "neutral"
    ASSASSIN = 3, "assassin"

    def to_str(self) -> str:
        return self.value[1]

    def to_int(self) -> int:
        return self.value[0]


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
                    card_type_str = "A"
                case CardType.NEUTRAL:
                    card_type_str = "N"
                case CardType.ASSASSIN:
                    # unicode 'skull and crossbones'
                    card_type_str = "\u2620"
                case _:
                    card_type_str = "?"
            display_str = (
                f"""{'* ' if card.is_played else ''}{card.word} [{card_type_str}]"""
            )
            return display_str

        str_matrix: list[list[str]] = []
        str_per_row = 5
        for card_nr, card in enumerate(self.cards, start=1):
            if card_nr % str_per_row == 1:
                str_matrix.append([])
            str_matrix[-1].append(f"{card_to_str(card)} ({card_nr})")

        print(tabulate.tabulate(str_matrix, tablefmt="grid"))
        print("* = already played")

    def get_playable_indices(self) -> list[int]:
        """
        Returns the list of indices of unplayed cards on the board.
        """
        return [i for i, card in enumerate(self.cards) if not card.is_played]

    def all_agents_found(self) -> bool:
        """
        Returns True iff all cards with type Agent have been played/found.
        """
        return all(
            [card.is_played for card in self.cards if card.type == CardType.AGENT]
        )


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
        print(
            f"You guessed '{guessed_card.word}'. It is a(n) {guessed_card.type.value} card."
        )
        match guessed_card.type:
            case CardType.ASSASSIN:
                self.lose()
                # return mostly here to satisfy type checking
                # in current implementation lose() ends the program
                return True
            case CardType.AGENT:
                if self.board.all_agents_found():
                    print("Well done! You have found the last agent and win!")
                    self.win()
                print(
                    "Well done! You can now choose to take another guess, or to "
                    "yield your turn and get new hint from the Spymaster."
                )
                return False
            case CardType.NEUTRAL:
                print(
                    "The Spymaster can now give another hint if there are turns left."
                )
                return True
        typing.assert_never(guessed_card.type)

    def lose(self) -> None:
        print("oh no, you lost :((((((")
        sys.exit()

    def play(self) -> None:
        turn_limit = 9
        turn_count = 1
        while turn_count <= turn_limit:
            print(f"\n==== Turn {turn_count} ===\n")
            hint: Hint = self.spymaster.generate_hint()
            guess_count = 1
            while True:
                guess: Guess = self.guesser.generate_guess(
                    board=self.board, hint=hint, is_yield_allowed=guess_count > 1
                )
                guess_count += 1
                stop_guess = self.handle_guess(guess)
                if stop_guess:
                    break
            turn_count += 1
        # we have reached the turn limit
        print("Turn limit reached")
        self.lose()

    def win(self) -> None:
        print("Sweet, sweet victory")
        sys.exit()


def make_random_board(words: list[str], rng: np.random.Generator) -> Board:
    """
    Generates a random board to play on.
    """
    n_cards = 25
    n_agents = 13
    n_assassins = 1
    card_words = rng.choice(words, n_cards)
    index_permutation = rng.permutation(np.arange(n_cards))
    agent_ix = set(index_permutation[:n_agents])
    assassin_ix = set(index_permutation[n_agents : (n_agents + n_assassins)])
    cards: list[Card] = []
    for i, word in enumerate(card_words):
        card_type: CardType
        if i in agent_ix:
            card_type = CardType.AGENT
        elif i in assassin_ix:
            card_type = CardType.ASSASSIN
        else:
            card_type = CardType.NEUTRAL
        cards.append(
            Card(
                word=word,
                type=card_type,
                is_played=False,
            )
        )
    return Board(cards)


@dataclasses.dataclass
class GameState:
    """
    Encapsulating everything related to game state. Just a simple object holding values.
    """

    board: Board
    current_turn_count: int = 1
    max_turn_count: int = 9
    current_hint: Hint | None = None
    consecutive_guesses: int = 1


BoardIndex: typing.TypeAlias = int


ObsType = npt.NDArray[np.int64]
ActionType = np.int64


class CodenamesGuesserEnv(gymnasium.Env[ObsType, ActionType]):
    """
    A Gymnasium Env in which the Guesser is an RL agent.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, card_words: list[str], render_mode: str = None) -> None:
        # 25 cards on the board, 1 action for 'yield'
        # -1 = yield, 0 .. 24 = index of card on board
        self.action_space = gymnasium.spaces.Discrete(n=25, start=0)
        # TODO change me to actual observation space
        # this will likely be tricky and need iteration
        # obs space = [
        #   (future: likely make these embeddings not indices)
        #   vocab indices of board words (25 integers, low=0, high=len(card_words)-1,
        n_cards = 25
        word_indices_low: list[int] = [0] * n_cards
        max_vocab_index = len(card_words) - 1
        word_indices_high: list[int] = [max_vocab_index] * n_cards
        #   card types for revealed cards, 0 otherwise (25 integers, low=0, high=max(integer values of CardType enum)),
        card_types_low: list[int] = [0] * n_cards
        card_types_high: list[int] = [
            max([card_type.to_int() for card_type in CardType])
        ] * n_cards
        #   is_observed for cards (25 integers, low=0, high=1)
        is_observed_low: list[int] = [0] * n_cards
        is_observed_high: list[int] = [1] * n_cards
        #   current turn count (1 integer, low=1, high=max turn count)
        current_turn_count_low: list[int] = [1]
        current_turn_count_high: list[int] = [9]  # max turn count
        #   max turn count (1 integer, low=max turn count, high=max turn count -> static -> remove from obs?)
        max_turn_count_low: list[int] = [9]
        max_turn_count_high = max_turn_count_low
        #   consecutive guesses (1 integer, low=0, high=???)
        consecutive_guesses_low: list[int] = [0]
        consecutive_guesses_high: list[int] = [25]
        #   current hint, number (1 integer, low=1, high=25?)
        current_hint_nr_low: list[int] = [1]
        current_hint_nr_high: list[int] = [25]
        #   current hint, word index (1 integer, same bounds as other vocab indices)
        current_hint_word_low: list[int] = [0]
        current_hint_word_high: list[int] = [max_vocab_index]

        all_low: npt.NDArray[np.int64] = np.asarray(
            list(itertools.chain(
                word_indices_low,
                card_types_low,
                is_observed_low,
                current_turn_count_low,
                max_turn_count_low,
                consecutive_guesses_low,
                current_hint_nr_low,
                current_hint_word_low,
            )),
            dtype=np.int64,
        )

        all_high: npt.NDArray[np.int64] = np.asarray(
            list(itertools.chain(
                word_indices_high,
                card_types_high,
                is_observed_high,
                current_turn_count_high,
                max_turn_count_high,
                consecutive_guesses_high,
                current_hint_nr_high,
                current_hint_word_high,
            )),
            dtype=np.int64,
        )

        self.observation_space = gymnasium.spaces.Box(
            low=all_low,
            high=all_high,
            dtype=np.int64,
        )
        self._words = card_words
        self._card_word_to_int: dict[str, int] = {
            word: i for i, word in enumerate(card_words)
        }
        self._gamestate: GameState | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, typing.Any] | None = None,
    ) -> tuple[typing.Any, dict[str, typing.Any]]:
        # required as per docs
        super().reset(seed=seed)
        self._gamestate = GameState(
            board=make_random_board(self._words, self.np_random)
        )
        hint: Hint = self._get_hint()
        self._gamestate.current_hint = hint
        return self._state_to_obs(self._gamestate), {}

    def step(
        self, action: np.int64
    ) -> tuple[typing.Any, float, bool, bool, dict[str, typing.Any]]:
        """
        Given a board and a hint coming from a Spymaster, get a guess from the Guesser.
        Evaluate what happens based on that guess, with possibilities being:
        - win
        - lose
        - get to guess again (do another step())
        - get a new hint from the spymaster, then guess again


        Winning or losing are indicated by the `terminated` flag. `truncated` is
        not currently used because an episode is inherently limited in length due to
        a turn limit that is part of Codenames.
        """
        # how should this go?
        # action -> Guesser action -> guess
        # submit guess, if game not done: also generate new hint from Spymaster
        # determine reward
        # return

        # this is true after having called reset()
        assert isinstance(self._gamestate, GameState)
        # bit annoying to do this though
        assert isinstance(self._gamestate.current_hint, Hint)

        terminated = False
        reward = 0
        truncated = False
        info: dict[str, typing.Any] = {}

        if self._gamestate.current_turn_count > self._gamestate.max_turn_count:
            reward -= 15
            terminated = True
            return (
                self._state_to_obs(self._gamestate),
                reward,
                terminated,
                truncated,
                info,
            )

        guessed_card = self._gamestate.board.cards[int(action)]

        match guessed_card.type:
            case CardType.ASSASSIN:
                terminated = True
                reward -= 15
            case CardType.AGENT:
                if self._gamestate.board.all_agents_found():
                    terminated = True
                    reward += 15
                reward += 5
                self._gamestate.consecutive_guesses += 1
            case CardType.NEUTRAL:
                reward -= 2
                self._gamestate.consecutive_guesses = 1
                # neutral -> guesser has to stop guess, get new hint from spymaster
                self._gamestate.current_hint = self._get_hint()
            case _:
                typing.assert_never(guessed_card.type)

        # generic state updates
        guessed_card.is_played = True
        self._gamestate.current_turn_count += 1
        # state updates end

        new_observation = self._state_to_obs(self._gamestate)

        return new_observation, reward, terminated, truncated, info

    def _state_to_obs(self, gamestate: GameState) -> typing.Any:
        """
        Construct an observation from current state.
        """
        # this is a dummy implementation, first need to understand how to structure
        # input well for RL agent

        # output could look like:
        # [int(word1), int(word2), ..., int(card_type1), int(card_type2 if card is played, sentinel otherwise), ..., int(is_played_n), ..., turn_count, max_turn_count, consecutive_guesses

        # True if reset() has been called
        assert isinstance(gamestate.current_hint, Hint)

        n_cards = len(gamestate.board.cards)
        word_encodings = np.zeros(n_cards, dtype=np.int64)
        revealed_card_types = np.zeros(n_cards, dtype=np.int64)
        cards_played_indicator = np.zeros(n_cards, dtype=np.int64)
        for i, card in enumerate(gamestate.board.cards):
            word_encodings[i] = self._card_word_to_int[card.word]
            revealed_card_types[i] = card.type.to_int() if card.is_played else 0
            cards_played_indicator[i] = int(card.is_played)

        other_features = np.zeros(5, dtype=np.int64)
        other_features[0] = gamestate.current_turn_count
        other_features[1] = gamestate.max_turn_count
        other_features[2] = gamestate.consecutive_guesses
        other_features[3] = gamestate.current_hint.number
        other_features[4] = self._card_word_to_int[gamestate.current_hint.word]

        obs_vector: npt.NDArray[np.int64] = np.concatenate(
            [
                word_encodings,
                revealed_card_types,
                cards_played_indicator,
                other_features,
            ],
            dtype=np.int64,
        )

        return obs_vector

    @staticmethod
    def _get_hint() -> Hint:
        """
        Elicit hint from spymaster.
        """
        # bootstrapping: hard-coded to see if rest of code runs
        return Hint("space", 2)

    def _get_guess(self, gamestate: GameState) -> BoardIndex:
        """
        Elicit guess from guesser. Returns guessed BoardIndex, i.e.
        index of Card on Board that Guesser wants to play.

        As a simplification we currently don't given the Guesser the option to
        yield their turn.
        """
        playable_indices: list[BoardIndex] = gamestate.board.get_playable_indices()
        return int(self.np_random.choice(playable_indices, size=1)[0])

    from gymnasium.core import RenderFrame

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        self._gamestate.board.display()
        return


def main() -> None:
    import sys
    print('path:', sys.path)
    codenames_wordlist_path = (
        Path(__file__)
        .parent.parent.parent.joinpath("data")
        .joinpath("codenames-wordlist-eng.txt")
    )
    with codenames_wordlist_path.open("r") as f:
        codenames_words = [word.strip().lower() for word in f.readlines()]

    # board = make_random_board(codenames_words, np.random.default_rng(42))
    # game = CodeNamesGame(spymaster=Spymaster(), guesser=Guesser(), board=board)
    # game.play()

    # env = CodenamesGuesserEnv(card_words=codenames_words)
    # vals = env.reset()
    # print(vals)
    from gymnasium.utils import env_checker

    gymnasium.register(
        'codenames_guesser',
        entry_point="codenames.main:CodenamesGuesserEnv",
        max_episode_steps=50,
    )
    env = gymnasium.make(
        'codenames_guesser',
        card_words=codenames_words,
        render_mode='ansi',
    )
    # env_checker.check_env(env.unwrapped)
    # # env._state_to_obs(env._gamestate)
    # env._get_guess(env._gamestate)
    # env.step(0)

    dqn = stable_baselines3.DQN(
        policy='MlpPolicy',
        env=env,
        verbose=1,
    )

    dqn.learn(total_timesteps=100, log_interval=10)


if __name__ == "__main__" and not is_interactive_py_session():
    main()
