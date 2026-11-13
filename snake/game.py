"""Headless Snake: relative actions 0=straight, 1=right, 2=left."""
import random

DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]

class Snake:
    def __init__(self, seed=0, size=10):
        self.rng = random.Random(seed)
        self.size = size
        self.body = [(size//2, size//2), (size//2-1, size//2), (size//2-2, size//2)]
        self.direction = 0
        self.score = self.steps = self.hungry = 0
        self.done = False
        self.food = self._food()

    def _food(self):
        free = [(x,y) for x in range(self.size) for y in range(self.size) if (x,y) not in self.body]
        return self.rng.choice(free) if free else None

    def danger(self, direction):
        dx,dy = DIRECTIONS[direction]
        x,y = self.body[0]
        target = (x+dx,y+dy)
        occupied = self.body if target == self.food else self.body[:-1]
        return not (0 <= target[0] < self.size and 0 <= target[1] < self.size) or target in occupied

    def state(self):
        x,y = self.body[0]
        fx,fy = self.food or (x,y)
        bits = [self.danger((self.direction+d)%4) for d in (0,1,-1)]
        bits += [self.direction == d for d in range(4)]
        bits += [fx<x, fx>x, fy<y, fy>y]
        return sum(int(b)<<i for i,b in enumerate(bits))

    def step(self, action):
        if self.done:
            raise ValueError('Game already ended')
        if action not in (0,1,2):
            raise ValueError('Invalid action')
        direction = (self.direction + (0,1,-1)[action]) % 4
        self.steps += 1
        self.hungry += 1
        if self.danger(direction):
            self.done = True
            return -10.0
        self.direction = direction
        x,y = self.body[0]
        dx,dy = DIRECTIONS[direction]
        self.body.insert(0,(x+dx,y+dy))
        reward = -0.01
        if self.body[0] == self.food:
            self.score += 1
            self.hungry = 0
            reward = 10.0
            self.food = self._food()
        else:
            self.body.pop()
        self.done = self.food is None or self.hungry >= 100 or self.steps >= 3000
        return reward

    def snapshot(self):
        return dict(size=self.size, body=self.body, food=self.food, score=self.score,
                    steps=self.steps, done=self.done)
