import time
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering

class ArrayEscapeEnvV2(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.grid_size = 10
        self.num_mines = self.grid_size * 2
        self.num_coords = self.num_mines + 3
        self.movement = [[0, 1, 0,-1], [1, 0, 1, 0]]
        self.SUCCESS_REWARD = 1000
        self.FAILURE_REWARD = -1000
        self.STEP_REWARD = -1
        self.viewer = None
        low = np.full((self.grid_size, self.grid_size), 0)
        high = np.full((self.grid_size, self.grid_size), 4)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low, high, dtype=np.int32)
        self._seed()

    def step(self, action):
        done = bool(self.coords[1][0] + self.movement[0][action] > self.grid_size -1
                    or self.coords[1][1] + self.movement[1][action] > self.grid_size -1
                    or self.coords[1][0] + self.movement[0][action] < 0
                    or self.coords[1][1] + self.movement[1][action] < 0
                    or self.grid[int(self.coords[1][0] + self.movement[0][action])][int(self.coords[1][1] + self.movement[1][action])] != 0
                    )
        reward = self.STEP_REWARD
        if not done:
            self.coords[0][0] = self.coords[1][0]
            self.coords[0][1] = self.coords[1][1]
            self.coords[1][0] += self.movement[0][action]
            self.coords[1][1] += self.movement[1][action]
            self.state = self._update_grid()
            #self.reward = self.STEP_REWARD
        else:
            if self.coords[1][0] != self.coords[2][0]:
                reward = self.FAILURE_REWARD
            elif self.coords[1][1] == self.coords[2][1]:
                reward = self.SUCCESS_REWARD

        return np.array(self.state), reward, done, {}

    def reset(self):
        self._place_items()
        self._place_items()
        self._set_initial_grid()
        self.state = self._update_grid()
        self._reset_render()
        return np.array(self.state)

    def render(self, mode='human'):
        
        self.screen_side = 800
        self.line_dist = self.screen_side / self.grid_size

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_side, self.screen_side)
        time.sleep(.15)
        for y in range(self.num_coords):
            box = rendering.FilledPolygon([
                (self.coords[y][0] * self.line_dist, self.coords[y][1] * self.line_dist), 
                ((self.coords[y][0] * self.line_dist) + self.line_dist, self.coords[y][1] * self.line_dist), 
                ((self.coords[y][0] * self.line_dist) + self.line_dist, (self.coords[y][1] * self.line_dist) + self.line_dist), 
                ((self.coords[y][0] * self.line_dist), (self.coords[y][1] * self.line_dist) + self.line_dist)])
            if(y == 0):
                box.set_color(0, 50, 50)
            if(y == 1):
                box.set_color(0, 0, 255)
            if(y == 2):
                box.set_color(0, 255, 0)
            if(y > 2):
                    box.set_color(255, 0, 0)
            self.viewer.add_geom(box)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def _place_items(self):
        self.grid = np.full((self.grid_size, self.grid_size), 0)
        self.coords = np.empty((self.num_coords, 2))
        self.coords[1][0] = 0
        self.coords[1][1] = 0

        for x in range(2, self.num_coords):
            self.coords[x][0] = self.np_random.randint(2, (self.grid_size -1))
            self.coords[x][1] = self.np_random.randint(2, (self.grid_size -1))
        
    def _set_initial_grid(self):
        for x in range(1, self.num_coords):
            int_column_coord = int(self.coords[x][0])
            int_row_coord = int(self.coords[x][1])
            if x == 1:                                    #current pose
                self.grid[int_column_coord] [int_row_coord] = 2
            elif x == 2:                                    #goal pose
                self.grid[int_column_coord] [int_row_coord] = 3
            else:                                           #mine
                self.grid[int_column_coord] [int_row_coord] = 4
    
    def _update_grid(self):
        if self.coords[0][0] != None:
            self.grid[int(self.coords[0][0])][int(self.coords[0][1])] = 1
            
        self.grid[int(self.coords[1][0])][int(self.coords[1][1])] = 2
        return self.grid
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset_render(self):
        if self.viewer is not None:
            box = rendering.FilledPolygon([
                (0, 0), 
                (0, self.screen_side), 
                (self.screen_side, self.screen_side), 
                (self.screen_side, 0)])

            box.set_color(255, 255, 255)
            self.viewer.add_geom(box)
            self.viewer.render()
