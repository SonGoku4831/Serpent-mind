import numpy as np

class Agent:
    def __init__(self, seed=0):
        self.q = np.zeros((2048, 3), dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def act(self, state, epsilon=0):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(3))
        return int(np.argmax(self.q[state]))

    def learn(self, state, action, reward, next_state, done):
        target = reward + (0 if done else 0.95 * self.q[next_state].max())
        self.q[state,action] += 0.1 * (target-self.q[state,action])

    @classmethod
    def load(cls, path):
        agent = cls()
        with np.load(path, allow_pickle=False) as data:
            agent.q = data['q']
        if agent.q.shape != (2048,3) or not np.isfinite(agent.q).all():
            raise ValueError('Invalid model')
        return agent
