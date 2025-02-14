import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle

from PIL.GimpGradientFile import EPSILON
from matplotlib import style
import time

SIZE = 4  # grid dimension
HM_EPISODES = 40000  # number of episodes
TURN_PENALTY = 50  # penalty for each turn
HIT_REWARD = 100  # reward for a hit
CONSECUTIVEHIT_REWARD = 50  # reward for consecutive hits
CONSECUTIVEMISS_PENALTY = 20  # penalty for misses after a hit
SUNK_REWARD = 30  # reward for sinking a ship
MISS_PENALTY = 25  # penalty for a miss
ALREADY_HIT_PENALTY = 200  # penalty for hitting a cell already hit
WIN_REWARD = 1030  # reward for winning
ZEROCELLS_REWARD = 20  # reward for each remaining zero cell in the grid
epsilon = 0.5  # exploration rate
EPSILON_DECAY = 0.99999  # exploration rate decay
SHOW_EVERY = 1000  # how often to show the game
LEARNING_RATE = 0.1  # learning rate
DISCOUNT = 0.9  # discount rate

# start_q_table = 'qtable-1738664528.pickle'    # None or Filename (qtable-1738664833.pickle)

# start_q_table = 'qtable-1738665202.pickle'    # TURN_PENALTY = 0.5 *i
# start_q_table = 'qtable-1738665606.pickle'    # TURN_PENALTY = 0.5, 1 after 16 turns

# start_q_table = 'qtable-1738925166.pickle'    # Higher penalties
# start_q_table = 'qtable-1738925776.pickle'    # New check_ship
# start_q_table = 'qtable-1739108506.pickle'    # Higher ALREADY_HIT_PENALTY, different miss representation
# start_q_table = 'qtable-1739118229.pickle'    # Higher ALREADY_HIT_PENALTY, new Ship.hit() method
# start_q_table = 'qtable-1739120706.pickle'    # Higher ALREADY_HIT_PENALTY

# start_q_table = 'qtable-1739121441.pickle'    # More rounds
# start_q_table = 'qtable-1739129671.pickle'    # 1000 rounds, no TURN_PENALTY when win
# start_q_table = 'qtable-1739207286.pickle'    # More epochs (40k vs 25k), higher ALREADY_HIT_PENALTY, introduced SUNK_REWARD
# start_q_table = 'qtable-1739374459.pickle'    # Last try

# start_q_table = 'qtable-1739390516.pickle'    # Higher TURN_PENALTY, WIN_REWARD and MISS_PENALTY
# start_q_table = 'qtable-1739390906.pickle'    # Higher TURN_PENALTY, WIN_REWARD and MISS_PENALTY
# start_q_table = 'qtable-1739436541.pickle'    # Introduced ZEROCELLS_REWARD
# start_q_table = 'qtable-1739438178.pickle'    # Introduced CONSECUTIVEHIT_REWARD
# start_q_table = 'qtable-1739444524.pickle'    # Introduced CONSECUTIVEMISS_PENALTY
# start_q_table = 'qtable-1739444650.pickle'    # Retrained on the previous qtable

# start_q_table = 'qtable-1739445130.pickle'    # Higher CONSECUTIVEMISS_PENALTY (20 vs 15)
# start_q_table = 'qtable-1739445255.pickle'    # Retrained on the previous qtable
# start_q_table = 'qtable-1739445374.pickle'    # Retrained on the previous qtable
# start_q_table = 'qtable-1739446745.pickle'    # Retrained on the previous qtable

# start_q_table = 'qtable-1739529293.pickle'    # 7x7 grid

# start_q_table = 'qtable-1739532004.pickle'    # Back to 4x4 grid, no diagonals contact
# start_q_table = 'qtable-1739532145.pickle'    # Retrained on the previous qtable
# start_q_table = 'qtable-1739532673.pickle'    # Retrained on the previous qtable
# start_q_table = None

SHIP_N = 1
SEA_N = 2
UNKNOWN_N = 3

d = {1: (0, 0, 255),
     2: (255, 175, 0),
     3: (128, 128, 128)}


class Ship:
    """
    A class to represent a ship in the Battleship game.

    Attributes:
    -----------
    size : int
        The size of the ship.
    hits : list
        A list to keep track of the coordinates where the ship has been hit.
    x1, x2, y1, y2 : int
        The coordinates of the ship's position on the grid.
    orientation : str
        The orientation of the ship, either 'horizontal' or 'vertical'.
    """

    def __init__(self, size):
        """
        Constructs all the necessary attributes for the ship object.

        Parameters:
        -----------
        size : int
            The size of the ship.
        """
        self.size = size
        self.hits = []
        self.x1 = self.x2 = self.y1 = self.y2 = 0
        self.orientation = None

    def place(self, x1, y1, orientation):
        """
        Saves the position and orientation of the ship on the grid.

        Parameters:
        -----------
        x1 : int
            The starting x-coordinate of the ship.
        y1 : int
            The starting y-coordinate of the ship.
        orientation : str
            The orientation of the ship, either 'horizontal' or 'vertical'.
        """
        self.x1 = x1
        self.y1 = y1
        self.orientation = orientation
        if orientation == "horizontal":
            self.x2 = x1
            self.y2 = y1 + self.size - 1
        else:
            self.x2 = x1 + self.size - 1
            self.y2 = y1

    def hit(self, x, y, show=False):
        """
        Registers a hit on the ship at the given coordinates.

        Parameters:
        -----------
        x : int
            The x-coordinate of the hit.
        y : int
            The y-coordinate of the hit.
        show : bool, optional
            If True, prints a message indicating the hit (default is False).

        Returns:
        --------
        bool
            True if the ship is completely hit (sunk), False otherwise.
        """
        if (x, y) not in self.hits:
            self.hits.append((x, y))
            if show:
                print(f"Ship hit at ({x}, {y})! {len(self.hits)}/{self.size}")
                # print(self.hits)
        return len(self.hits) == self.size


class Battleship:
    """
    A class to represent the Battleship game environment.

    Attributes:
    -----------
    ships : list
        A list of Ship objects representing the ships in the game.
    opponent_grid : numpy.ndarray
        A 2D array representing the opponent's grid with ship positions.
        -1 means sea, 1, 2, 3, ... are the ships
    player_grid : numpy.ndarray
        A 2D array representing the player's grid with hits and misses.
        0 means unknown, -3 means sea, -2 is a hit
    sunken_ships : list
        A list to keep track of the indices of sunken ships.
    """

    def __init__(self):
        """
        Constructs all the necessary attributes for the Battleship object.
        """
        # print("Battleship")
        self.ships = [Ship(3), Ship(2)]
        self.opponent_grid = self.create_grid(opponent=True)
        self.build_ships()
        self.player_grid = self.create_grid(opponent=False)
        self.sunken_ships = []

    def create_grid(self, opponent: bool):
        """
        Creates a grid for the game.

        Parameters:
        -----------
        opponent : bool
            If True, creates the opponent's grid. Otherwise, creates the player's grid.

        Returns:
        --------
        numpy.ndarray
            A 2D array representing the grid.
        """
        if opponent:
            return np.full((SIZE, SIZE), -1)
        else:
            return np.zeros((SIZE, SIZE))

    def build_ships(self):
        """
        Randomly places ships on the opponent's grid.
        """
        for ship in self.ships:
            while True:
                x, y = np.random.randint(0, SIZE, 2)
                orientation = np.random.choice(["horizontal", "vertical"])
                # print(orientation)
                if self.check_ship(x, y, ship, orientation):
                    self.place_ship(x, y, ship, orientation)
                    break

    def check_ship(self, x, y, ship, orientation):
        """
        Checks if a ship can be placed at the given coordinates with the given orientation.

        Parameters:
        -----------
        x : int
           The x-coordinate of the starting position.
        y : int
           The y-coordinate of the starting position.
        ship : Ship
           The ship object to be placed.
        orientation : str
           The orientation of the ship, either 'horizontal' or 'vertical'.

        Returns:
        --------
        bool
           True if the ship can be placed, False otherwise.
        """
        # print(f"Checking ship {ship.size} at {x}, {y} with orientation {orientation}")
        if orientation == "vertical":
            if x + ship.size > SIZE:
                return False
            for i in range(ship.size):
                # Check if it doesn't overlap with other ships
                if self.opponent_grid[x + i, y] != -1:
                    return False

                # Check if it doesn't touch other ships laterally
                if y > 0 and self.opponent_grid[x + i, y - 1] != -1:
                    return False
                if y < SIZE - 1 and self.opponent_grid[x + i, y + 1] != -1:
                    return False

                # Check if it doesn't touch other ships at the ends
                if i == 0 and x > 0 and self.opponent_grid[x - 1, y] != -1:
                    return False
                if i == ship.size - 1 and x + i < SIZE - 1 and self.opponent_grid[x + i + 1, y] != -1:
                    return False

                # Check if it doesn't touch other ships diagonally
                if not self._checkdiag(x, y):
                    return False
                if not self._checkdiag(x + ship.size - 1, y):
                    return False
        elif orientation == "horizontal":
            if y + ship.size > SIZE:
                return False
            for i in range(ship.size):
                # Check if it doesn't overlap with other ships
                if self.opponent_grid[x, y + i] != -1:
                    return False

                # Check if it doesn't touch other ships laterally
                if x > 0 and self.opponent_grid[x - 1, y + i] != -1:
                    return False
                if x < SIZE - 1 and self.opponent_grid[x + 1, y + i] != -1:
                    return False

                # Check if it doesn't touch other ships at the ends
                if i == 0 and y > 0 and self.opponent_grid[x, y - 1] != -1:
                    return False
                if i == ship.size - 1 and y + i < SIZE - 1 and self.opponent_grid[x, y + i + 1] != -1:
                    return False

            # Check if it doesn't touch other ships diagonally
            if not self._checkdiag(x, y):
                return False
            if not self._checkdiag(x, y + ship.size - 1):
                return False
        return True

    def _checkdiag(self, x, y):
        if x > 0 and y > 0 and self.opponent_grid[x - 1, y - 1] != -1:
            return False
        if x < SIZE - 1 and y > 0 and self.opponent_grid[x + 1, y - 1] != -1:
            return False
        if x > 0 and y < SIZE - 1 and self.opponent_grid[x - 1, y + 1] != -1:
            return False
        if x < SIZE - 1 and y < SIZE - 1 and self.opponent_grid[x + 1, y + 1] != -1:
            return False
        return True

    def place_ship(self, x, y, ship, orientation):
        """
        Places a ship on the opponent's grid at the given coordinates with the given orientation.

        Parameters:
        -----------
        x : int
            The x-coordinate of the starting position.
        y : int
            The y-coordinate of the starting position.
        ship : Ship
            The ship object to be placed.
        orientation : str
            The orientation of the ship, either 'horizontal' or 'vertical'.

        Returns:
        --------
        bool
            True if the ship is successfully placed.
        """
        # In the grid there will be the index of the ship + 1 (because 0 is unknown)
        if orientation == "vertical":
            for i in range(ship.size):
                self.opponent_grid[x + i, y] = self.ships.index(ship) + 1
            ship.place(x, y, "vertical")
        elif orientation == "horizontal":
            for i in range(ship.size):
                self.opponent_grid[x, y + i] = self.ships.index(ship) + 1
            ship.place(x, y, "horizontal")
        return True

    def action(self, x, y, last_action):
        """
        Performs an action on the player's grid at the given coordinates.

        Parameters:
        -----------
        x : int
            The x-coordinate of the action.
        y : int
            The y-coordinate of the action.
        last_action : bool
            False if the last action was a miss, True if it was a hit.

        Returns:
        --------
        int
            The reward for the action.
        bool
            True if the action is a hit, False otherwise.
        """
        rew = 0
        if self.player_grid[x, y] != 0:  # already hit
            # print("Already hit")
            return -ALREADY_HIT_PENALTY, False

        value = int(self.ask(x, y))
        if value == -1:  # miss
            self.update_grid(x, y, -MISS_PENALTY)
            if last_action:
                rew = -CONSECUTIVEMISS_PENALTY
            else:
                rew = 0
            return -MISS_PENALTY + rew, False
        elif value > 0:  # hit
            self.update_grid(x, y, HIT_REWARD)
            ship = self.ships[value - 1]
            if ship.hit(x, y):
                if self.ships.index(ship) not in self.sunken_ships:
                    self.sunken_ships.append(self.ships.index(ship))
                    rew = SUNK_REWARD
                else:
                    rew = 0
                if len(self.sunken_ships) == len(self.ships):
                    return WIN_REWARD, True
                if last_action:
                    rew += CONSECUTIVEHIT_REWARD
            return HIT_REWARD + rew, True

    def ask(self, x, y):
        return self.opponent_grid[x, y]

    def update_grid(self, x, y, reward):
        if reward == HIT_REWARD:
            self.player_grid[x, y] = -2
        else:
            self.player_grid[x, y] = -3


def state_to_key(state):
    return tuple(state.flatten())


def get_q_values(state):
    """
    Retrieves the Q-values for a given state from the Q-table.
    If the state is not in the Q-table, initializes it with random values.

    Parameters:
    -----------
    state : numpy.ndarray
        The current state of the game represented as a 2D array.

    Returns:
    --------
    numpy.ndarray
        The Q-values for the given state.
    """
    key = state_to_key(state)
    # print(key)
    if key not in q_table:
        q_table[key] = np.random.uniform(-5, 0, (SIZE, SIZE))
    return q_table[key]


def show_img(grid):
    env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    for j in range(SIZE):
        for k in range(SIZE):
            if grid[j, k] == 0:  # unknown
                env[j, k] = d[UNKNOWN_N]
            elif grid[j, k] == -1 or grid[j, k] == -3:  # sea
                env[j, k] = d[SEA_N]
            else:  # ship
                env[j, k] = d[SHIP_N]
    img = Image.fromarray(env, "RGB")
    img = img.resize((300, 300), Image.NEAREST)
    cv2.imshow("image", np.array(img))


if __name__ == "__main__":
    start_time = time.time()

    if start_q_table is None:
        q_table = {}
    else:
        with open(start_q_table, "rb") as f:
            q_table = pickle.load(f)

    episode_rewards = []
    for episode in range(HM_EPISODES):
        player = Battleship()
        show_img(player.opponent_grid)
        # cv2.waitKey(50000)

        if episode % SHOW_EVERY == 0:
            print(f"on #{episode}, epsilon is {epsilon}")
            print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
            show = True
        else:
            show = False

        episode_reward = 0
        # TURN_PENALTY = 1.5
        last_action = False  # False if miss, True if hit
        for i in range(1000):
            # Choose an action: it can be random or the one with the highest Q-value
            if np.random.random() > epsilon:
                q_values = get_q_values(player.player_grid)
                action = np.argmax(q_values)
                x, y = np.unravel_index(action, q_values.shape)
            else:
                x = np.random.randint(0, SIZE)
                y = np.random.randint(0, SIZE)

            current_q = get_q_values(player.player_grid)[x, y]
            reward, hit = player.action(x, y, last_action)
            reward -= TURN_PENALTY
            last_action = hit
            # print(reward)

            max_future_q = np.max(get_q_values(player.player_grid))

            # Update the Q-value for the current state and action
            if reward == WIN_REWARD - TURN_PENALTY:
                print(f"WIN on episode {episode}")
                new_q = WIN_REWARD - TURN_PENALTY + (ZEROCELLS_REWARD * np.count_nonzero(player.player_grid == 0))
            else:
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[state_to_key(player.player_grid)][x, y] = new_q

            if show:
                show_img(player.player_grid)
                if reward == WIN_REWARD - TURN_PENALTY:
                    if cv2.waitKey(500) & 0xFF == ord("q"):
                        break
                else:
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            episode_reward += reward
            if reward == WIN_REWARD - TURN_PENALTY:
                print("WIN")
                # TURN_PENALTY = 0
                break

        episode_rewards.append(episode_reward)
        epsilon *= EPSILON_DECAY

    moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')

    plt.plot([i for i in range(len(moving_avg))], moving_avg)
    plt.ylabel(f"Reward {SHOW_EVERY}ma")
    plt.xlabel("episode #")
    plt.show()

    with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
        pickle.dump(q_table, f)

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
