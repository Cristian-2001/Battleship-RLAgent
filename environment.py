import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle

from PIL.GimpGradientFile import EPSILON
from matplotlib import style
import time

SIZE = 4  # grid dimension
HM_EPISODES = 25000
TURN_PENALTY = 1.5
HIT_REWARD = 10
MISS_PENALTY = 2
ALREADY_HIT_PENALTY = 60
WIN_REWARD = 100
epsilon = 0.5
EPSILON_DECAY = 0.99999
SHOW_EVERY = 1000

# start_q_table = 'qtable-1738664528.pickle'    # None or Filename (qtable-1738664833.pickle)
# start_q_table = 'qtable-1738665202.pickle'    # TURN_PENALTY = 0.5 *i
# start_q_table = 'qtable-1738665606.pickle'    # TURN_PENALTY = 0.5, 1 after 16 turns
# start_q_table = 'qtable-1738925166.pickle'    # Higher penalties
# start_q_table = 'qtable-1738925776.pickle'    # New check_ship
# start_q_table = 'qtable-1739108506.pickle'    # Higher ALREADY_HIT_PENALTY, different miss representation
# start_q_table = 'qtable-1739118229.pickle'    # Higher ALREADY_HIT_PENALTY, new Ship.hit() method
start_q_table = None

LEARNING_RATE = 0.1
DISCOUNT = 0.9

SHIP_N = 1
SEA_N = 2
UNKNOWN_N = 3

d = {1: (0, 0, 255),
     2: (255, 175, 0),
     3: (128, 128, 128)}


class Ship:
    def __init__(self, size):
        self.size = size
        self.hits = []
        self.x1 = self.x2 = self.y1 = self.y2 = 0
        self.orientation = None

    def place(self, x1, y1, orientation):
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
        if (x, y) not in self.hits:
            self.hits.append((x, y))
            if show:
                print(f"Ship hit! {len(self.hits)}/{self.size}")
                print(self.hits)
        return len(self.hits) == self.size


class Battleship:
    '''
    In this case:
    - opponent_grid is the known grid of the player with the ships, the agent will try to hit the ships
    -1 means sea, 1, 2, 3, ... are the ships
    - player_grid is the unknown grid: the agent will update it with the hits and misses
    0 means unknown, -3 means sea, -2 is a hit
    '''

    def __init__(self):
        # print("Battleship")
        self.ships = [Ship(3), Ship(2)]
        self.opponent_grid = self.create_grid(opponent=True)
        self.build_ships()
        self.player_grid = self.create_grid(opponent=False)
        self.sunken_ships = 0

    def create_grid(self, opponent: bool):
        if opponent:
            return np.full((SIZE, SIZE), -1)
        else:
            return np.zeros((SIZE, SIZE))

    def build_ships(self):
        for ship in self.ships:
            while True:
                x, y = np.random.randint(0, SIZE, 2)
                orientation = np.random.choice(["horizontal", "vertical"])
                # print(orientation)
                if self.check_ship(x, y, ship, orientation):
                    self.place_ship(x, y, ship, orientation)
                    break

    def check_ship(self, x, y, ship, orientation):
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
        return True

    def place_ship(self, x, y, ship, orientation):
        # print(f"Placing ship {ship.size} at {x}, {y} with orientation {orientation}")
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

    def action(self, x, y):
        if self.player_grid[x, y] != 0:  # already hit
            # print("Already hit")
            return -ALREADY_HIT_PENALTY

        value = int(self.ask(x, y))
        if value == -1:  # miss
            self.update_grid(x, y, -MISS_PENALTY)
            return -MISS_PENALTY
        elif value > 0:  # hit
            self.update_grid(x, y, HIT_REWARD)
            ship = self.ships[value - 1]
            if ship.hit(x, y):
                self.sunken_ships += 1
                if self.sunken_ships == len(self.ships):
                    return WIN_REWARD
            return HIT_REWARD

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
        for i in range(100):
            if i > 16:
                TURN_PENALTY = 1
            if np.random.random() > epsilon:
                q_values = get_q_values(player.player_grid)
                action = np.argmax(q_values)
                x, y = np.unravel_index(action, q_values.shape)
            else:
                x = np.random.randint(0, SIZE)
                y = np.random.randint(0, SIZE)

            current_q = get_q_values(player.player_grid)[x, y]
            reward = player.action(x, y) - TURN_PENALTY
            # print(reward)

            max_future_q = np.max(get_q_values(player.player_grid))

            if reward == WIN_REWARD - TURN_PENALTY:
                print(f"WIN on episode {episode}")
                new_q = WIN_REWARD - TURN_PENALTY
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
