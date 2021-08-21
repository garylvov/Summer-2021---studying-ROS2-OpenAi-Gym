import math
import random
import gym
import numpy as np
from gym import spaces, error
from gym.utils import seeding
from gym.envs.classic_control import rendering

class ArrayEscapeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.grid_size = 10 #grid is square, minumum size of 3
        high = np.array(self._create_populated_array(0))
        low = np.array(self._create_populated_array(2))

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low, high, dtype=np.int32)
        
        self.state = None
        self.viewer = None
        self.last_runner_pos = [None, None]
        
        '''
        action corresponding to movement

        0 is current pos
        7 8 1 
        6 0 2 
        5 4 3
        '''
        #.................0..1..2..3..4..5..6..7..8
        self.movement = [[0,-1, 0, 1, 1, 1, 0,-1,-1], 
                         [0, 1, 1, 1, 0,-1,-1,-1, 0]]
        
        self.seed()
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.grid = np.array(self._create_populated_array(0))
        self.mine_pos = [self.np_random.randint(0, (self.grid_size -1)) , self.np_random.randint(0, (self.grid_size -1))]
        self.runner_pos = [self.np_random.randint(0, (self.grid_size -1)) , self.np_random.randint(0, (self.grid_size -1))]
        
        while(self.mine_pos == self.runner_pos):
             self.runner_pos = [self.np_random.randint(0, (self.grid_size -1)) , self.np_random.randint(0, (self.grid_size -1))]
        
        self._update_state()
        return np.array(self.state)

    def step(self, action):
        self.last_runner_pos[0] = self.runner_pos[0]
        self.last_runner_pos[1] = self.runner_pos[1]

        done = bool(
            self.runner_pos[0] + self.movement[0][action] < 0
            or self.runner_pos[1] + self.movement[1][action] < 0
            or self.runner_pos[0] + self.movement[0][action] > self.grid_size -1
            or self.runner_pos[1] + self.movement[1][action] > self.grid_size -1
            or self.runner_pos == self.mine_pos
        )
        
        self.runner_pos[0] += self.movement[0][action]
        self.runner_pos[1] += self.movement[1][action]
        
        if done == False:
            self._update_state()

        reward = ((self.runner_pos[0] - self.mine_pos[0])**2 + (self.runner_pos[1] - self.mine_pos[1])**2)**.5
        return np.array(self.state), reward, done, {}

    def render(self, mode='human'):
        self.screen_side = 800
        self.line_dist = self.screen_side / self.grid_size

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_side, self.screen_side)
            mine = rendering.FilledPolygon([(self.mine_pos[0] * self.line_dist, self.mine_pos[1] * self.line_dist), 
                                           ((self.mine_pos[0] * self.line_dist) + self.line_dist, self.mine_pos[1] * self.line_dist), 
                                           ((self.mine_pos[0] * self.line_dist) + self.line_dist, (self.mine_pos[1] * self.line_dist) + self.line_dist), 
                                           ((self.mine_pos[0] * self.line_dist), (self.mine_pos[1] * self.line_dist) + self.line_dist)])
            mine.set_color(255, 0, 0)
            self.viewer.add_geom(mine)

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                box = rendering.FilledPolygon([(x * self.line_dist, y * self.line_dist), 
                                             ((x * self.line_dist) + self.line_dist, y * self.line_dist), 
                                             ((x * self.line_dist) + self.line_dist, (y * self.line_dist) + self.line_dist), 
                                            ((x * self.line_dist), (y * self.line_dist) + self.line_dist)])
                if self.grid[x][y] == 0:
                    box.set_color(255, 255, 255)
                elif self.grid[x][y] == 1:
                    box.set_color(255, 0, 0)
                else:
                    box.set_color(0, 0, 255)
                self.viewer.add_geom(box)
                
        self._draw_grid()
        self.viewer.render(return_rgb_array=mode == "rgb_array")
   
        #print(np.matrix(self.grid))

    def _create_populated_array(self, num):
        array = np.empty((self.grid_size, self.grid_size), int)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                array[x][y] = num
        return array 

    def _update_state(self):
        self.grid[self.last_runner_pos[0]][self.last_runner_pos[1]] = 0
        self.grid[self.mine_pos[0]][self.mine_pos[1]] = 1
        self.grid[self.runner_pos[0]][self.runner_pos[1]] = 2
        self.state = self.grid

    def _draw_grid(self):
        for x in range(self.grid_size):
            x_line = rendering.Line((0, (x * self.line_dist)),(self.screen_side, (x * self.line_dist)))
            y_line = rendering.Line(((x * self.line_dist), 0),((x * self.line_dist), self.screen_side))
            x_line.set_color(0, 0, 0)
            y_line.set_color(0, 0, 0)
            self.viewer.add_geom(x_line)
            self.viewer.add_geom(y_line)