import time
import random
import gym
import numpy as np
from gym import spaces, error
from gym.utils import seeding
from gym.envs.classic_control import rendering

class ArrayEscapeEnvV1(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        self.grid_size = 10 #grid is square, minumum size of 3
        self.max_dist = ((self.grid_size**2) * 2) ** .5
        self.max_steps = self.grid_size * 1.5
        self.mine_num = int(self.grid_size * 1.75) + 1
        self.items_depth = self.mine_num + 3
        self.last_item = [0,0]
        self.items = None
        self.state = None
        self.viewer = None
        #X 4 X
        #3 0 1
        #X 2 X
        #.................0..1..2..3..4
        self.movement = [[0, 1, 0,-1], 
                         [1, 0, 1, 0]]
        
        high = np.array(self._create_populated_array(0))
        low = np.array(self._create_populated_array(self.grid_size -1))
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low, high, dtype=np.int32)
        self.seed()
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.items = self._create_populated_items_array(self.items_depth)
        self.state = self.items
        self._reset_render()
        self.steps_elapsed_in_ep = 0
        return np.array(self.state)

    def step(self, action):
        self.steps_elapsed_in_ep +=1
        self.last_item[0] = self.items[0][0]
        self.last_item[1] = self.items[0][1]

        self.items[0][0] = self.items[1][0]
        self.items[0][1] = self.items[1][1]

        dist = (((self.items[1][0] - self.items[2][0])**2 + (self.items[1][1] - self.items[2][1])**2)**.5)

        new_column = self.items[1][0] + self.movement[0][action]
        new_row = self.items[1][1] + self.movement[1][action]

        done = bool(
            new_column < 0
            or new_row < 0
            or new_column > self.grid_size -1
            or new_row > self.grid_size -1
            or self._check_mines()
            #or self.steps_elapsed_in_ep > self.max_steps
        )

        if not done: 
            self.items[1][0] += self.movement[0][action]
            self.items[1][1] += self.movement[1][action]

            if(self.last_item[0] == self.items[1][0] and self.last_item[1] == self.items[1][1]):
                #print("REPEAT")
                reward = self.grid_size * -3
                done = True

            self.state = self.items
            reward = 1 #* -dist
        else:
            if dist != 0:
                reward = self.grid_size * -3
            else:
                reward = self.grid_size * 10
        
        return np.array(self.state), reward, done, {}

    def render(self, mode='human'):
        
        self.screen_side = 800
        self.line_dist = self.screen_side / self.grid_size

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_side, self.screen_side)
        time.sleep(.15)
        for y in range(self.items_depth):
            box = rendering.FilledPolygon([
                (self.items[y][0] * self.line_dist, self.items[y][1] * self.line_dist), 
                ((self.items[y][0] * self.line_dist) + self.line_dist, self.items[y][1] * self.line_dist), 
                ((self.items[y][0] * self.line_dist) + self.line_dist, (self.items[y][1] * self.line_dist) + self.line_dist), 
                ((self.items[y][0] * self.line_dist), (self.items[y][1] * self.line_dist) + self.line_dist)])
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

    def _create_populated_array(self, num):
        array = np.empty((self.items_depth, 2)).astype(np.int32)
        for y in range(self.items_depth):
            for x in range(2):
                array[y][x] = num
        return array 
    
    def _create_populated_items_array(self, num):
        array = np.empty((num, 2)).astype(np.int32)
        for y in range(1, num):
            for x in range(2):
                array[y][x] = self.np_random.randint(0, (self.grid_size -1))
        
        #ensure that moving square doesn't start on a mine
        for y in range(2, num):
            if(array[1][0] == array[y][0] or array[2][0] == array[y][0]):
                array[y][0] = self.np_random.randint(0, (self.grid_size -1))
            elif(array[1][1] == array[y][0] or array[2][1] == array[y][0]):
                array[y][1] = self.np_random.randint(0, (self.grid_size -1))
        return array

    def _check_mines(self):
        collided = False
        for x in range(2, self.items_depth):
            if(self.items[1][0] == self.items[x][0] and self.items[1][1] == self.items[x][1]):
                collided = True
        return collided
        
    def _draw_grid(self):
        for x in range(self.grid_size):
            x_line = rendering.Line((0, (x * self.line_dist)),(self.screen_side, (x * self.line_dist)))
            y_line = rendering.Line(((x * self.line_dist), 0),((x * self.line_dist), self.screen_side))
            x_line.set_color(0, 0, 0)
            y_line.set_color(0, 0, 0)
            self.viewer.add_geom(x_line)
            self.viewer.add_geom(y_line)

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
