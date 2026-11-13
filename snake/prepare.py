"""Ray workers generate baseline rollouts and summarize raw episode data."""
import json
from pathlib import Path
import ray
from snake.game import Snake

@ray.remote
def rollout(seed):
    game = Snake(seed)
    transitions = []
    while not game.done:
        state = game.state()
        action = game.rng.randrange(3)
        reward = game.step(action)
        transitions.append([state, action, reward, game.state(), game.done])
    return dict(seed=seed, score=game.score, steps=game.steps, transitions=transitions)

def main():
    ray.init(num_cpus=2, include_dashboard=False, object_store_memory=100 * 1024 * 1024)
    try:
        rows = ray.get([rollout.remote(seed) for seed in range(100)])
        Path('data').mkdir(exist_ok=True)
        Path('data/rollouts.json').write_text(json.dumps(rows))
        print(f'Prepared {len(rows)} episodes using Ray')
    finally:
        ray.shutdown()

if __name__ == '__main__':
    main()
