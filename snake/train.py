import argparse
import hashlib
import json
import os
from pathlib import Path
import mlflow
import numpy as np
from snake.agent import Agent
from snake.game import Snake

def evaluate(agent, episodes=100):
    scores=[]
    for seed in range(100000,100000+episodes):
        game=Snake(seed)
        while not game.done:
            game.step(agent.act(game.state()))
        scores.append(game.score)
    return float(np.mean(scores))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=30000)
    parser.add_argument('--seed', type=int, default=42)
    args=parser.parse_args()
    if args.episodes < 1:
        parser.error('episodes must be positive')
    raw=Path('data/rollouts.json').read_bytes()
    rows=json.loads(raw)
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI','http://localhost:5000'))
    mlflow.set_experiment('snake-q-learning')
    agent=Agent(args.seed)
    with mlflow.start_run() as run:
        mlflow.log_params(dict(episodes=args.episodes, seed=args.seed, alpha=0.1,
                               gamma=0.95, grid_size=10, data_sha256=hashlib.sha256(raw).hexdigest()))
        for row in rows:
            for s,a,r,n,d in row['transitions']:
                agent.learn(s,a,r,n,d)
        scores=[]
        for episode in range(args.episodes):
            game=Snake(args.seed+episode)
            epsilon=max(0.02, 1-episode/(args.episodes*0.8))
            while not game.done:
                state=game.state()
                action=agent.act(state,epsilon)
                reward=game.step(action)
                agent.learn(state,action,reward,game.state(),game.done)
            scores.append(game.score)
            if (episode+1)%100 == 0:
                mlflow.log_metrics({'train_score_mean_100':float(np.mean(scores[-100:])),
                                    'epsilon':epsilon}, step=episode+1)
        score=evaluate(agent)
        baseline=evaluate(Agent(args.seed))
        metrics=dict(eval_mean_score=score, untrained_mean_score=baseline)
        Path('models').mkdir(exist_ok=True)
        # Replace atomically; API loads once at startup.
        with open('models/q.tmp','wb') as f:
            np.savez_compressed(f,q=agent.q)
        os.replace('models/q.tmp','models/q.npz')
        Path('models/metrics.json').write_text(json.dumps(metrics,indent=2))
        Path('models/metadata.json').write_text(json.dumps(dict(run_id=run.info.run_id,
            episodes=args.episodes,seed=args.seed,data_sha256=hashlib.sha256(raw).hexdigest()),indent=2))
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts('models',artifact_path='model')
        print(json.dumps(metrics,indent=2))

if __name__ == '__main__':
    main()
