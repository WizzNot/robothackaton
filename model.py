import numpy as np
from heapq import heappop, heappush
import gym
from pogema.animation import AnimationMonitor
from pogema import GridConfig
from IPython.display import SVG, display
import random
random.seed(2)

class Node:
    def __init__(self, coord: (int, int) = (0, 0), g: int = 0, h: int = 0):
        self.i, self.j = coord
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f or ((self.f == other.f) and (self.g < other.g))
    
class AStar:
    def __init__(self):
        self.start = (0, 0)
        self.goal = (0, 0)
        self.max_steps = 10000  # due to the absence of information about the map size we need some other stop criterion
        self.OPEN = list()
        self.CLOSED = dict()
        self.obstacles = set()
        self.other_agents = set()
        self.coords = list()
        self.counts = random.randrange(0, 1)
        self.stay = False

    def compute_shortest_path(self, start, goal):
        self.start = start
        self.coords.append(start)
        self.goal = goal
        self.CLOSED = dict()
        self.OPEN = list()
        heappush(self.OPEN, Node(self.start))
        u = Node()
        steps = 0
        while len(self.OPEN) > 0 and steps < self.max_steps and (u.i, u.j) != self.goal:
            u = heappop(self.OPEN)
            steps += 1
            for d in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                n = (u.i+d[0], u.j + d[1])
                if n not in self.obstacles and n not in self.CLOSED and n not in self.other_agents:
                    h = abs(n[0] - self.goal[0]) + abs(n[1] - self.goal[1])  # Manhattan distance as a heuristic function
                    heappush(self.OPEN, Node(n, u.g + 1, h))
                    self.CLOSED[n] = (u.i, u.j)  # store information about the predecessor

    def get_next_node(self):
        next_node = self.start  # if path not found, current start position is returned
        if self.stay:
            next_node = self.start
        elif any([self.coords.count(i) > 2 for i in self.coords]):
            if self.counts < random.randrange(0,2):
                if any([abs(self.start[0] - i[0]) == 1 and abs(self.start[1] - i[1]) == 1 for i in self.obstacles]):
                    a = (self.start[0], self.start[1]+random.randrange(-1, 1))
                    while a in self.obstacles:
                        b = random.choice([True, False])
                        if b:
                            a = (self.start[0]+random.randrange(-1, 1), self.start[1])
                        else:
                            a = (self.start[0], self.start[1]+random.randrange(-1, 1))
                    next_node = a
                    self.stay = True
                    x=random.randrange(1, 2)
                    if x<2:
                        self.stay = False
                self.counts += 1
            else:
                self.coords = []
                self.counts = random.randrange(0,1)
        elif self.goal in self.CLOSED:  # if path found
            next_node = self.goal
            while self.CLOSED[next_node] != self.start:  # get node in the path with start node as a predecessor
                next_node = self.CLOSED[next_node]
        return next_node

    def update_obstacles(self, obs, other_agents, n):
        obstacles = np.transpose(np.nonzero(obs))  # get the coordinates of all obstacles in current observation
        for obstacle in obstacles:
            self.obstacles.add((n[0] + obstacle[0], n[1] + obstacle[1]))  # save them with correct coordinates
        self.other_agents.clear()  # forget previously seen agents as they move
        agents = np.transpose(np.nonzero(other_agents))  # get the coordinates of all agents that are seen
        for agent in agents:
            self.other_agents.add((n[0] + agent[0], n[1] + agent[1]))  # save them with correct coordinates


class Model:
    def __init__(self):
        self.agents = False
        self.actions = {tuple(GridConfig().MOVES[i]): i for i in
                        range(len(GridConfig().MOVES))}  # make a dictionary to translate coordinates of actions into id

    def act(self, obs, dones, positions_xy, targets_xy) -> list:
        # print(positions_xy)
        if self.agents ==False:
            self.agents = [AStar() for _ in range(len(obs))]  # create a planner for each of the agents
        actions = []
        for k in range(len(obs)):
            if positions_xy[k] == targets_xy[k]:  # don't waste time on the agents that have already reached their goals
                actions.append(0)  # just add useless action to save the order and length of the actions
                continue
            self.agents[k].update_obstacles(obs[k][0], obs[k][1], (positions_xy[k][0] - 5, positions_xy[k][1] - 5))
            self.agents[k].compute_shortest_path(start=positions_xy[k], goal=targets_xy[k])
            next_node = self.agents[k].get_next_node()
            actions.append(self.actions[(next_node[0] - positions_xy[k][0], next_node[1] - positions_xy[k][1])])
        return actions

