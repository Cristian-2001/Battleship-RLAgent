import numpy as np
from PIL import Image
import cv2
import pickle
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.animation import FuncAnimation

from space import BlockingTupleSpace
from environment import Ship, SIZE, SHIP_N, SEA_N, UNKNOWN_N, d

q_table_name = 'qtables/qtable-1738925776.pickle'
epsilon = 0.1


class BattleshipAgent:
    def __init__(self, tuple_space: BlockingTupleSpace, turn: bool, name):
        print("Agent created")
        self.name = name
        self.ts = tuple_space
        self.turn = turn
        self.id = 0

        self.q_table = self.load_q_table(q_table_name)

        self.grid_size = SIZE
        self.ships = [Ship(3), Ship(2)]
        self.sunken_ships = 0
        self.done = False

        self.player_grid = np.full((self.grid_size, self.grid_size), -1)
        self.build_ships()
        print("Ships placed")
        self.opponent_grid = np.zeros((self.grid_size, self.grid_size))

    def get_state(self):
        return self.opponent_grid

    def load_q_table(self, name):
        with open(name, "rb") as f:
            q_table = pickle.load(f)
        return q_table

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
                if self.player_grid[x + i, y] != -1:
                    return False

                # Check if it doesn't touch other ships laterally
                if y > 0 and self.player_grid[x + i, y - 1] != -1:
                    return False
                if y < SIZE - 1 and self.player_grid[x + i, y + 1] != -1:
                    return False

                # Check if it doesn't touch other ships at the ends
                if i == 0 and x > 0 and self.player_grid[x - 1, y] != -1:
                    return False
                if i == ship.size - 1 and x + i < SIZE - 1 and self.player_grid[x + i + 1, y] != -1:
                    return False
        elif orientation == "horizontal":
            if y + ship.size > SIZE:
                return False
            for i in range(ship.size):
                # Check if it doesn't overlap with other ships
                if self.player_grid[x, y + i] != -1:
                    return False

                # Check if it doesn't touch other ships laterally
                if x > 0 and self.player_grid[x - 1, y + i] != -1:
                    return False
                if x < SIZE - 1 and self.player_grid[x + 1, y + i] != -1:
                    return False

                # Check if it doesn't touch other ships at the ends
                if i == 0 and y > 0 and self.player_grid[x, y - 1] != -1:
                    return False
                if i == ship.size - 1 and y + i < SIZE - 1 and self.player_grid[x, y + i + 1] != -1:
                    return False
        return True

    def place_ship(self, x, y, ship, orientation):
        # In the grid there will be the index of the ship + 1 (because 0 is unknown)
        if orientation == "vertical":
            for i in range(ship.size):
                self.player_grid[x + i, y] = self.ships.index(ship) + 1
            ship.place(x, y, "vertical")
        elif orientation == "horizontal":
            for i in range(ship.size):
                self.player_grid[x, y + i] = self.ships.index(ship) + 1
            ship.place(x, y, "horizontal")
        return True

    def step(self):
        print("Step")
        if self.turn:
            x, y = self.choose_action()
            value = int(self.ask(x, y))
            if value == -1:  # miss
                self.update_opponent_grid(x, y, 'miss')
            elif value > 0:  # hit
                self.update_opponent_grid(x, y, 'hit')
            elif self.opponent_grid[x, y] != 0:
                print("Already hit or missed")
            else:
                print("Error")

            if self.check_game_over():
                self.done = True
                print(f"Game over, {self.name} won. {self.id}")

            self.change_turn()
        else:
            # Wait for the request
            request = self.ts.remove(("Request", self.id, None, None))
            x, y = request[2], request[3]
            # Respond to the request
            self.ts.add(("Response", self.id, x, y, self.player_grid[x, y]))
            ship = self.ships[self.player_grid[x, y] - 1]
            if ship.hit():
                self.sunken_ships += 1
            # Check if the game is over and publish the result
            if self.sunken_ships == len(self.ships):
                self.ts.add(("Game over", self.id, True))
                self.done = True
                print(f"Game over, {self.name} lost. {self.id}")
            else:
                self.ts.add(("Game over", self.id, False))
            # Change turn
            self.change_turn()

    def state_to_key(self, state):
        return tuple(state.flatten())

    def get_q_values(self, state):
        key = self.state_to_key(state)
        # print(key)
        if key not in self.q_table:
            self.q_table[key] = np.random.uniform(-5, 0, (SIZE, SIZE))
        return self.q_table[key]

    def choose_action(self):
        if np.random.random() > epsilon:
            q_values = self.get_q_values(self.get_state())
            x, y = np.unravel_index(np.argmax(q_values, axis=None), q_values.shape)
        else:
            x = np.random.randint(0, SIZE)
            y = np.random.randint(0, SIZE)
        return x, y

    def ask(self, x, y):
        self.ts.add(("Request", self.id, x, y))
        result = self.ts.remove(("Response", self.id, x, y, None))
        return result[4]

    def update_opponent_grid(self, x, y, result):
        if result == 'hit':
            self.opponent_grid[x, y] = -2
        else:
            self.opponent_grid[x, y] = -1

    def check_game_over(self):
        response = self.ts.remove(("Game over", self.id, None))
        return response[2]

    def change_turn(self):
        # self.update_grids(self.player_grid, "Player")
        # self.update_grids(self.opponent_grid, "Opponent")
        self.id += 1
        if self.turn:
            self.ts.add(("Turn", self.id))
            self.turn = False
            self.ts.add(("Update_grids", self.id))
        else:
            self.ts.remove(("Turn", self.id))
            self.turn = True

    def loop(self, delay=2):
        print("Loop")
        while not self.done:
            self.step()
            time.sleep(delay)

    def update_grids(self, grid, text):
        env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        for j in range(SIZE):
            for k in range(SIZE):
                if grid[j, k] == 0:  # unknown
                    env[j, k] = d[UNKNOWN_N]
                elif grid[j, k] == -1:  # sea
                    env[j, k] = d[SEA_N]
                else:  # ship
                    env[j, k] = d[SHIP_N]
        img = Image.fromarray(env, "RGB")
        img = img.resize((300, 300), Image.NEAREST)
        cv2.imshow(self.name + " " + text, np.array(img))


if __name__ == "__main__":
    ts = BlockingTupleSpace()
    agent1 = BattleshipAgent(ts, True, "Player 1")
    agent2 = BattleshipAgent(ts, False, "Player 2")


    def player1():
        agent1.loop()


    def player2():
        agent2.loop()


    threading.Thread(target=player1, daemon=True).start()
    threading.Thread(target=player2, daemon=True).start()

    # Set up the plot
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    custom_cmap = ListedColormap (['blue', 'blue', 'grey', 'red', 'red', 'red'])

    # Define boundaries: ensures correct color assignment
    boundaries = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(boundaries, custom_cmap.N)

    config_plot = [
        (agent1.player_grid, 'Player 1 self grid', axs[0, 0]),
        (agent1.opponent_grid, 'Player 1 opponent grid', axs[1, 0]),
        (agent2.player_grid, 'Player 2 self grid', axs[0, 1]),
        (agent2.opponent_grid, 'Player 2 opponent grid', axs[1, 1]),
    ]

    imgs = []
    for matrix, text, ax in config_plot:
        im = ax.plot(matrix)
        ax.set_title(text)
        print(type(im))
        imgs.append(im)

    def update(frame, imgs):
        for im in imgs:
            im.set_data(agent1.player_grid)
        return imgs

    # while not agent1.done and not agent2.done:
    #     ts.remove(("Update_grids", None))
    #     print("ECCOMI")

    anim = FuncAnimation(fig, update, fargs=(imgs,), frames=1, repeat=False)
    plt.show()


