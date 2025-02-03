import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle

from PIL.GimpGradientFile import EPSILON
from matplotlib import style
import time

SIZE = 4
HM_EPISODES = 25000
TURN_REWARD = -1
HIT_REWARD = 10
MISS_REWARD = -10
ALREADY_HIT_REWARD = -20
WIN_REWARD = 100
epsilon = 0.5
EPSILON_DECAY = 0.9998
SHOW_EVERY = 1

start_q_table = None  # None or Filename

LEARNING_RATE = 0.1
DISCOUNT = 0.95

SHIP_N = 1
SEA_N = 2
UNKNOWN_N = 3

d = {1: (255, 175, 0),
     2: (0, 0, 255),
     3: (128, 128, 128)}


class Battleship:
    def __init__(self):
        print("Battleship")
        self.ships = [3, 2]
        self.player_grid = self.create_grid()
        self.build_ships(player=True)
        self.opponent_grid = self.create_grid()
        self.build_ships(player=False)

    def create_grid(self):
        return np.zeros((SIZE, SIZE))

    def build_ships(self, player):
        for ship in self.ships:
            while True:
                x, y = np.random.randint(0, SIZE, 2)
                orientation = np.random.choice(["horizontal", "vertical"])
                if self.check_ship(x, y, ship, orientation, player):
                    self.place_ship(x, y, ship, orientation, player)
                    break

    def check_ship(self, x, y, ship, orientation, player):
        if player:
            grid = self.player_grid
        else:
            grid = self.opponent_grid
        if orientation == "horizontal":
            if x + ship > SIZE:
                return False
            for i in range(ship):
                if grid[x + i, y] != 0:
                    return False
        elif orientation == "vertical":
            if y + ship > SIZE:
                return False
            for i in range(ship):
                if grid[x, y + i] != 0:
                    return False
        return True

    def place_ship(self, x, y, ship, orientation, player):
        if player:
            grid = self.player_grid
        else:
            grid = self.opponent_grid
        if orientation == "horizontal":
            for i in range(ship):
                grid[x + i, y] = ship
        elif orientation == "vertical":
            for i in range(ship):
                grid[x, y + i] = ship
        return True

    def action(self, x, y):
        value = self.ask(x, y)
        if value < 0:
            return ALREADY_HIT_REWARD
        elif value != 0:
            self.update_grid(x, y, HIT_REWARD)
            return HIT_REWARD
        else:
            self.update_grid(x, y, MISS_REWARD)
            return MISS_REWARD

    def ask(self, x, y):
        return self.player_grid[x, y]

    def update_grid(self, x, y, reward):
        if reward == HIT_REWARD:
            self.player_grid[x, y] = -1
        else:
            self.player_grid[x, y] = -2


def state_to_key(state):
    return tuple(state.flatten())


def get_q_values(state):
    key = state_to_key(state)
    # print(key)
    if key not in q_table:
        print("ECCOMI")
        q_table[key] = np.random.uniform(-5, 0, (SIZE, SIZE))
    return q_table[key]


if start_q_table is None:
    q_table = {}
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

episode_rewards = []
for episode in range(HM_EPISODES):
    player = Battleship()

    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        if np.random.random() > epsilon:
            q_values = get_q_values(player.opponent_grid)
            action = np.argmax(q_values)
            x, y = np.unravel_index(action, q_values.shape)
        else:
            x = np.random.randint(0, SIZE)
            y = np.random.randint(0, SIZE)

        current_q = get_q_values(player.opponent_grid)[x, y]
        reward = player.action(x, y) + TURN_REWARD

        max_future_q = np.max(get_q_values(player.opponent_grid))

        if reward == WIN_REWARD:
            new_q = WIN_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            for i in range(SIZE):
                for j in range(SIZE):
                    if player.opponent_grid[i, j] == -1:
                        env[i, j] = d[SHIP_N]
                    elif player.opponent_grid[i, j] == -2:
                        env[i, j] = d[SEA_N]
                    else:
                        env[i, j] = d[UNKNOWN_N]
            img = Image.fromarray(env, "RGB")
            img = img.resize((300, 300))
            cv2.imshow("image", np.array(img))
            if reward == WIN_REWARD:
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        episode_reward += reward
        if reward == WIN_REWARD:
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
