import numpy as np
from PIL import Image
import cv2
import pickle
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import sys

from space import BlockingTupleSpace
from environment import Ship, SIZE, SHIP_N, SEA_N, UNKNOWN_N, d

q_table_name = 'qtables/qtable-1739615698.pickle'
epsilon = 0.1


class BattleshipAgent:
    """
    A class to represent an agent in the Battleship game.

    Attributes:
    -----------
    id : str
        The name of the agent.
    ts : BlockingTupleSpace
        The tuple space for communication between agents.
    turn : bool
        A flag indicating if it's the agent's turn.
    counter : int
        A counter for the number of steps taken.
    q_table : dict
        The Q-table for storing Q-values.
    grid_size : int
        The size of the game grid.
    ships : list
        A list of Ship objects representing the ships in the game.
    sunken_ships : list
        A list to keep track of the indices of sunken ships.
    done : bool
        A flag indicating if the game is over.
    player_grid : numpy.ndarray
        A 2D array representing the player's grid with ship positions.
    opponent_grid : numpy.ndarray
        A 2D array representing the opponent's grid with hits and misses.
    """

    def __init__(self, tuple_space: BlockingTupleSpace, turn: bool, name):
        print("Agent created")
        self.id = name
        self.ts = tuple_space
        self.turn = turn
        self.counter = 0

        self.q_table = self.load_q_table(q_table_name)

        self.grid_size = SIZE
        self.ships = [Ship(3), Ship(2)]
        self.sunken_ships = []
        self.done = False

        self.player_grid = np.full((self.grid_size, self.grid_size), -1)  # It fills the grid with -1 because it's sea
        self.build_ships()
        print("Ships placed")
        print(self.player_grid)
        self.opponent_grid = np.zeros((self.grid_size, self.grid_size))  # It fills the grid with 0 because it's unknown

    def get_state(self):
        return self.opponent_grid

    def load_q_table(self, name):
        with open(name, "rb") as f:
            q_table = pickle.load(f)
        return q_table

    def build_ships(self):
        """
        Randomly places ships on the player's grid.
        """
        for ship in self.ships:
            while True:
                x, y = np.random.randint(0, SIZE, 2)
                orientation = np.random.choice(["horizontal", "vertical"])
                # print(orientation)
                if self.check_ship(x, y, ship, orientation):
                    self.place_ship(x, y, ship, orientation)
                    print("---")
                    print(self.id)
                    print(x, y, orientation)
                    print(ship.x1, ship.y1, ship.x2, ship.y2, ship.orientation)
                    print("---")
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

            # Check if it doesn't touch other ships diagonally
            if not self._checkdiag(x, y):
                return False
            if not self._checkdiag(x, y + ship.size - 1):
                return False
        return True

    def _checkdiag(self, x, y):
        if x > 0 and y > 0 and self.player_grid[x - 1, y - 1] != -1:
            return False
        if x < SIZE - 1 and y > 0 and self.player_grid[x + 1, y - 1] != -1:
            return False
        if x > 0 and y < SIZE - 1 and self.player_grid[x - 1, y + 1] != -1:
            return False
        if x < SIZE - 1 and y < SIZE - 1 and self.player_grid[x + 1, y + 1] != -1:
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
        """
        Performs a step in the game, either making a move or responding to a request.
        """
        print("Step")
        # If it's the agent's turn, make a move
        if self.turn:
            x, y = self.choose_action()
            value = int(self.ask(x, y))
            # print(self.id, x, y, value)
            if value == -1:  # miss
                print("Miss")
                self.update_opponent_grid(x, y, 'miss')
            elif value > 0:  # hit
                print("Hit")
                self.update_opponent_grid(x, y, 'hit')
            elif value != 0:
                print("Already hit or missed!!!")
            else:
                print("Error")

            # Check if the game is over
            if self.check_game_over():
                self.done = True
                print(f"Game over, {self.id} won. {self.counter}")
                print(self.opponent_grid)

            # Change turn
            self.change_turn()
        else:
            # Wait for the request
            request = self.ts.remove(("Request", self.counter, None, None))
            x, y = request[2], request[3]

            # Respond to the request
            value = self.player_grid[x, y]
            self.ts.add(("Response", self.counter, x, y, value))
            ship = self.ships[value - 1]

            if value > 0:  # If a ship was hit
                if ship.hit(x, y, True):  # Check if the ship was sunk
                    if self.ships.index(ship) not in self.sunken_ships:
                        self.sunken_ships.append(self.ships.index(ship))
                        print(
                            f"Agent {self.id}: Affondata nave {ship.x1} {ship.y1} {ship.x2} {ship.y2} {ship.orientation}")
                        print(self.sunken_ships)
                print(f"Agent {self.id}: [{ship.hits}]")

                # Check if the game is over and publish the result
                if len(self.sunken_ships) == len(self.ships):
                    self.ts.add(("Game over", self.counter, True))
                    self.done = True
                    print(f"Game over, {self.id} lost. {self.counter}")
                    print(self.opponent_grid)
                else:
                    self.ts.add(("Game over", self.counter, False))
            else:
                self.ts.add(("Game over", self.counter, False))

            # Change turn
            self.change_turn()

    def state_to_key(self, state):
        return tuple(state.flatten())

    def get_q_values(self, state):
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
        key = self.state_to_key(state)
        # print(key)
        if key not in self.q_table:
            self.q_table[key] = np.random.uniform(-5, 0, (SIZE, SIZE))
        return self.q_table[key]

    def choose_action(self):
        """
        Chooses an action based on the current state and Q-values.
        It follows an epsilon-greedy policy with a probability of epsilon to choose a random action.

        Returns:
        --------
        tuple
            The coordinates (x, y) of the chosen action.
        """
        if np.random.random() > epsilon:
            q_values = self.get_q_values(self.get_state())
            x, y = np.unravel_index(np.argmax(q_values, axis=None), q_values.shape)

            # Uncomment this to hardcode the prohibition of the same move
            if self.opponent_grid[x, y] != 0:
                print("Already hit or missed")
                while self.opponent_grid[x, y] != 0:
                    x = np.random.randint(0, SIZE)
                    y = np.random.randint(0, SIZE)
        else:
            print("Random")
            x = np.random.randint(0, SIZE)
            y = np.random.randint(0, SIZE)
            if self.opponent_grid[x, y] != 0:
                while self.opponent_grid[x, y] != 0:
                    x = np.random.randint(0, SIZE)
                    y = np.random.randint(0, SIZE)
        return x, y

    def ask(self, x, y):
        """
        Sends a request to the opponent and returns the response.
        """
        self.ts.add(("Request", self.counter, x, y))
        result = self.ts.remove(("Response", self.counter, x, y, None))
        return result[4]

    def update_opponent_grid(self, x, y, result):
        if result == 'hit':
            self.opponent_grid[x, y] = -2
        else:
            self.opponent_grid[x, y] = -3

    def check_game_over(self):
        response = self.ts.remove(("Game over", self.counter, None))
        return response[2]

    def change_turn(self):
        """
        Changes the turn to the other player.
        It is used to synchronize the agents.
        """
        self.counter += 1
        if self.turn:
            self.ts.add(("Turn", self.counter))
            self.turn = False
            self.ts.add(("Update_grids", self.counter))
        else:
            self.ts.remove(("Turn", self.counter))
            self.turn = True

    def loop(self, delay=2):
        """
        Runs the game loop until the game is over.

        Parameters:
        -----------
        delay : int, optional
            The delay between steps in seconds (default is 2).
        """
        print("Loop")
        while not self.done:
            self.step()
            time.sleep(delay)


def plot_grids(n_turn, agent1, agent2):
    # Set up the plot
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    custom_cmap = ListedColormap(['blue', 'red', 'blue', 'grey', 'red', 'red', 'red'])

    # Define boundaries: ensures correct color assignment
    boundaries = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(boundaries, custom_cmap.N)

    config_plot = [
        (agent1.player_grid, 'Player 1 self grid', axs[0, 0]),
        (agent1.opponent_grid, 'Player 1 opponent grid', axs[1, 0]),
        (agent2.player_grid, 'Player 2 self grid', axs[0, 1]),
        (agent2.opponent_grid, 'Player 2 opponent grid', axs[1, 1]),
    ]

    # Plot matrices
    for matrix, title, ax in config_plot:
        im = ax.imshow(matrix, cmap=custom_cmap, norm=norm)
        ax.set_title(title)
        # ax.axis("off")  # Hide axis for better visualization

    # Save the figure
    fig.suptitle(f"Turn {n_turn}")
    plt.show()


if __name__ == "__main__":
    np.printoptions(threshold=sys.maxsize)
    n_turn = 0

    # Create the tuple space
    ts = BlockingTupleSpace()

    # Create the agents
    agent1 = BattleshipAgent(ts, True, "Player 1")
    agent2 = BattleshipAgent(ts, False, "Player 2")


    def player1():
        agent1.loop()


    def player2():
        agent2.loop()


    # Start the game
    threading.Thread(target=player1, daemon=True).start()
    threading.Thread(target=player2, daemon=True).start()

    # Plot the initial grids
    plot_grids(n_turn, agent1, agent2)

    # Until game is over, wait for the turn to change and plot the grids each turn
    while not agent1.done and not agent2.done:
        ts.remove(("Update_grids", None))
        print("ECCOMI")
        n_turn += 1

        plot_grids(n_turn, agent1, agent2)
