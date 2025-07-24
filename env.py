import numpy as np
import random

class MinesweeperEnv:
    def __init__(self, width=9, height=9, n_mines=10,
                 rewards={'win':5, 'lose':-5, 'progress':0.5, 'guess':-1, 'no_progress': -0.5}):
        self.nrows, self.ncols, self.n_mines, self.rewards = width, height, n_mines, rewards
        self.ntiles = self.nrows * self.ncols
        self.reset()

    def init_grid(self):
        board = np.zeros((self.nrows, self.ncols), dtype='object')
        mines = self.n_mines
        while mines:
            i, j = random.randrange(self.nrows), random.randrange(self.ncols)
            if board[i, j] != 'B':
                board[i, j] = 'B'
                mines -= 1
        return board

    def get_neighbors(self, coord):
        x, y = coord
        return [self.grid[i, j]
                for i in range(x-1, x+2)
                for j in range(y-1, y+2)
                if 0 <= i < self.nrows and 0 <= j < self.ncols and (i, j) != coord]

    def count_bombs(self, coord):
        return sum(1 for n in self.get_neighbors(coord) if n == 'B')

    def get_board(self):
        b = self.grid.copy()
        for x in range(self.nrows):
            for y in range(self.ncols):
                if b[x, y] != 'B':
                    b[x, y] = self.count_bombs((x, y))
        return b

    def get_state_im(self, state):
        im = np.array([t['value'] for t in state], dtype='object').reshape(self.nrows, self.ncols)
        im[im == 'U'] = -1
        im[im == 'B'] = -2
        return (im.astype(float) / 8).astype(np.float32)[..., None]

    def init_state(self):
        self.grid = self.init_grid()
        self.board = self.get_board()
        self.state = [{'coord': (i, j), 'value': 'U'}
                      for i in range(self.nrows) for j in range(self.ncols)]
        self.n_clicks = 0
        return self.get_state_im(self.state)

    def reset(self):
        return self.init_state()

    def click(self, idx):
        coord = self.state[idx]['coord']
        val = self.board[coord]

        if val == 'B' and self.n_clicks == 0:
            safe = np.where(self.grid.reshape(-1) != 'B')[0]
            idx = random.choice(safe)
            coord = self.state[idx]['coord']
            val = self.board[coord]

        self.state[idx]['value'] = val
        if val == 0:
            self._reveal(coord, [])

        self.n_clicks += 1

    def _reveal(self, coord, seen):
        seen.append(coord)
        for i in range(coord[0]-1, coord[0]+2):
            for j in range(coord[1]-1, coord[1]+2):
                if (i, j) in seen or not (0 <= i < self.nrows and 0 <= j < self.ncols):
                    continue
                idx = next(k for k, t in enumerate(self.state) if t['coord'] == (i, j))
                self.state[idx]['value'] = self.board[i, j]
                if self.board[i, j] == 0:
                    self._reveal((i, j), seen)

    def step(self, action):
        prev = self.get_state_im(self.state)
        self.click(action)
        state_im = self.get_state_im(self.state)
        self.state_im = state_im
        done = False

        tile_val = self.state[action]['value']
        if tile_val == 'B':
            return state_im, self.rewards['lose'], True
        if (state_im == -0.125).sum() == self.n_mines:
            return state_im, self.rewards['win'], True
        if (state_im == prev).all():
            return state_im, self.rewards['no_progress'], False

        neighbors = self.get_neighbors(self.state[action]['coord'])
        reward = self.rewards['guess'] if all(n == 'U' for n in neighbors) else self.rewards['progress']
        return state_im, reward, done
