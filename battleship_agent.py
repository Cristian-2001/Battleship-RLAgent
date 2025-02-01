import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from space import BlockingTupleSpace


class BattleshipAgent:
    def __init__(self, tuple_space: BlockingTupleSpace, turn: bool):
        self.ts = tuple_space
        self.turn = turn
        self.grid_size = 10
        self.ships = [5, 4, 3, 3, 2]

        # 0: unknown, -1: hit, -2: miss, 2-5: ship
        self.player_grid = np.zeros((self.grid_size, self.grid_size))
        self.player_grid += -2                                          # Mark as empty
        self.opponent_grid = np.zeros((self.grid_size, self.grid_size))

        self.ships_remaining = self.ships.copy()
        self.id = 0

        set1_colors = ["#999999", "#377EB8", "#E41A1C"]                 # Colors for the grids: grey, blue, red (0, -2, -1)
        custom_cmap = mcolors.ListedColormap(set1_colors)
        self.fig, (self.ax1, self.ax2) = plt.subplots(self.grid_size, self.grid_size, figsize=(10, 10))
        self.im1 = self.ax1.imshow(self.player_grid, cmap=custom_cmap)       # Player grid
        self.im2 = self.ax2.imshow(self.opponent_grid, cmap=custom_cmap)     # Opponent grid

        self.ax1.set_title("Player grid")
        self.ax2.set_title("Opponent grid")

    def get_state(self):
        return np.stack([self.player_grid, self.opponent_grid], axis=-1)

    def step(self, action):
        if self.turn:
            x, y = action
            reward = 0
            done = False

            if self.opponent_grid[x, y] != 0:  # Already hit or missed: BAD!
                reward = -2
            elif self.ask(x, y):  # Hit
                reward = 1
                self.opponent_grid[x, y] = -1  # Mark as hit
            elif not self.ask(x, y):  # Miss
                reward = -1
                self.opponent_grid[x, y] = -2  # Mark as miss
            done = self.check_game_over()
            if done:
                reward = 10
                self.id += 1
                self.ts.add(("Turn", self.id))
                self.turn = False
            self.update_grids()
            return self.get_state(), reward, done
        else:
            request = self.ts.remove(("Request", self.id, None, None))
            if self.player_grid[request[2], request[3]] != 0:
                self.ts.add(("Response", self.id, request[2], request[3], True))
            else:
                self.ts.add(("Response", self.id, request[2], request[3], False))

            if not self.ships_remaining:
                self.ts.add(("Game over", self.id, True))
                reward = -10
                done = True
            else:
                self.ts.add(("Game over", self.id, False))

            self.id += 1
            self.ts.remove(("Turn", self.id))
            self.turn = True

    def ask(self, x, y):
        self.ts.add(("Request", self.id, x, y))
        result = self.ts.remove(("Response", self.id, x, y, None))
        return result[4]

    def check_game_over(self):
        result = self.ts.remove(("Game over", self.id, None))
        return result[2]

    def loop(self):
        while True:
            self.step(None)


    def build_ships(self):
        for ship in self.ships:
            while True:
                x, y = np.random.randint(0, self.grid_size, 2)
                orientation = np.random.choice(["horizontal", "vertical"])
                if self.check_ship(x, y, ship, orientation):
                    self.place_ship(x, y, ship, orientation)
                    break

    def check_ship(self, x, y, ship, orientation):
        if orientation == "horizontal":
            if x + ship > self.grid_size:
                return False
            for i in range(ship):
                if self.player_grid[x + i, y] != 0:
                    return False
        elif orientation == "vertical":
            if y + ship > self.grid_size:
                return False
            for i in range(ship):
                if self.player_grid[x, y + i] != 0:
                    return False
        return True

    def place_ship(self, x, y, ship, orientation):
        if orientation == "horizontal":
            for i in range(ship):
                self.player_grid[x + i, y] = ship
        elif orientation == "vertical":
            for i in range(ship):
                self.player_grid[x, y + i] = ship
        self.ships_remaining.remove(ship)
        return True

    def update_grids(self):
        self.im1.set_data(self.player_grid)
        self.im2.set_data(self.opponent_grid)
        plt.pause(0.1)
        self.fig.canvas.draw()
        plt.show()


if __name__ == "__main__":
    ts = BlockingTupleSpace()
    agent = BattleshipAgent(ts, True)
    agent.build_ships()
    agent.loop()
