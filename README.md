# Serpent Mind — Snake AI + minimal AIOps

Python Snake and tabular Q-learning, with Ray, MLflow, DVC, FastAPI,
Streamlit, Docker Compose, Prometheus and Grafana. Designed for a local Ubuntu
VM with 4 GB RAM and two CPU cores. No GPU, CUDA, PyTorch or Pygame required.
The browser draws the board; the same headless game runs during training and serving.

## Run on Ubuntu

Install Docker Engine and its Compose v2 plugin using Docker's official Ubuntu
instructions: https://docs.docker.com/engine/install/ubuntu/ . Install Git with
`sudo apt-get update && sudo apt-get install -y git`. Verify
`docker compose version` and `docker info`; use `sudo docker` consistently if your
account does not have Docker access. Allow at least 6 GB free disk space (10 GB recommended) for
images, packages, model data and tracking. These commands assume Bash.

```bash
git clone https://github.com/SonGoku4831/Serpent-mind.git
cd Serpent-mind
# Build once: all Python services reuse this image's layers.
docker compose build trainer
# DVC is local, with its cache stored in this project directory.
[ -d .dvc ] || docker compose run --rm --no-deps trainer dvc init
# Train with only MLflow running to keep peak memory down.
docker compose up -d mlflow
docker compose run --rm trainer dvc repro
# Record reproducible pipeline hashes and configuration in Git.
git add .
git -c user.name="Snake Developer" -c user.email="snake@localhost" commit -m "Track Snake pipeline and model"
# Start web application and monitoring after training completes.
docker compose up -d api ui prometheus grafana
```

Open http://localhost:8501 and click **Play**. Different game seeds produce
different food layouts. The model must finish training before starting the API;
there is deliberately no untrained fallback. First training runs 30,000 episodes;
runtime varies by CPU. Watch the printed evaluation score and MLflow metrics.

| Service | Local URL |
|---|---|
| Game UI | http://localhost:8501 |
| FastAPI docs | http://localhost:8000/docs |
| MLflow experiments | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana dashboard | http://localhost:3000/d/snake |

Grafana is provisioned automatically and allows anonymous local viewing.
All published ports bind to loopback. If Ubuntu is a remote VM, run this on your
own computer, replacing USER and VM_IP, then open the local URLs:

```bash
ssh -L 8501:localhost:8501 -L 5000:localhost:5000 -L 8000:localhost:8000 -L 9090:localhost:9090 -L 3000:localhost:3000 USER@VM_IP
```

## Commands after the first run

```bash
# View service status and logs.
docker compose ps
docker compose logs -f api ui
# Run the automated tests.
docker compose run --rm --no-deps trainer python -m pytest -q
# Retrain: pause other services on a 4 GB machine.
docker compose stop api ui prometheus grafana
docker compose run --rm trainer dvc repro --force
# Save new hashes and metrics; existing MLflow experiments remain available.
git add dvc.lock models/metrics.json
git -c user.name="Snake Developer" -c user.email="snake@localhost" commit -m "Retrain Snake agent"
# API restart loads the new model.
docker compose up -d api ui prometheus grafana
docker compose restart api
# Stop everything, preserving MLflow, DVC and monitoring data.
docker compose down
```

Edit `--episodes` or `--seed` in **dvc.yaml** to change training; `dvc repro`
then reruns the affected stage. Every actual training execution opens a new
MLflow run, including parameters, dataset SHA-256, scores, evaluation, and model
artifacts. A cached `dvc repro` skips work and correctly creates no training run.
MLflow's named Docker volume stores SQLite and proxied artifacts across restarts.

DVC caches raw Ray rollouts and model files locally, while Git stores dvc.yaml,
dvc.lock, model metadata pointers and evaluation metrics. No cloud account is
required. Do not delete `.dvc/cache` if you want to restore historical models.
To restore a past run, stop the API, check out the desired Git commit and execute:

```bash
docker compose run --rm --no-deps trainer dvc checkout
docker compose up -d api
```

For sharing or off-machine backup, configure a DVC remote yourself and use
`dvc push` / `dvc pull`; a local cache alone is not a backup. To reproduce from
source when the cache is missing, use `dvc repro --force` with MLflow running.

## Design choices and limits

- **Game:** 10×10 board, three relative actions, +10 food, -10 collision, -0.01
  per move. End after 100 moves without food, 3,000 total moves, collision, or a
  full board. Moving into a tail cell is legal when the tail vacates it.
- **Agent:** 11 boolean observations encode immediate danger, direction and food
  direction. A 2,048×3 float32 Q-table is about 24 KB. Learning rate 0.1,
  discount 0.95, decaying epsilon to 0.02. This compact observation omits the
  full body layout, so it can get trapped and is not expected to fill the board.
- **Data engineering:** two Ray CPU workers create 100 seeded random-policy
  rollouts. Training consumes these as warm-start transitions, then learns from
  fresh online episodes. Ray is short-lived and its object store is 100 MB.
- **Evaluation:** 100 fixed held-out seeds compare the greedy trained agent
  against a zero-table greedy agent. These metrics measure food eaten per game;
  competence varies with training length, and no guaranteed score is claimed.
- **Serving:** a single FastAPI worker loads the immutable model at startup.
  In-memory game sessions expire after ten minutes, with at most 100 sessions.
  Restart the API after retraining. Sessions are ephemeral by design.
- **UI:** Streamlit requests one server-side move at a time and renders a NumPy
  image. Click Play again when a game finishes. No desktop display is required.
- **Monitoring:** Prometheus collects game count, active games, decision count,
  decision latency, score distribution, process and Python metrics. Grafana shows
  games, active sessions, moves/sec, mean score, p95 decision latency and API up.
  Training metrics belong to MLflow. Scores appear after a game completes;
  latency rates need several scrapes. Prometheus retains two days / 256 MB.
- **Deployment:** one shared Python image keeps setup simple; it contains all
  requested tools. MLflow has one worker and services have memory limits.
  Training and the live stack run separately on the small VM. Docker's daemon,
  build and OS also consume memory; close other apps. MX250 is unused because
  this agent is too small to benefit from a GPU.
- **Scope:** local educational pipeline without authentication or automatic
  model promotion. MLflow and DVC retain each run; serving loads the latest
  locally produced model. External production hosting needs separate hardening.

## Python-only development (optional)

Use Python 3.11 or 3.12. Docker remains the simplest complete deployment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
# Keep MLflow running via Compose, then:
python -m snake.prepare
python -m snake.train --episodes 30000
# Separate terminals, each with the virtual environment activated:
uvicorn snake.api:app --host 127.0.0.1 --port 8000
streamlit run ui.py --server.address 127.0.0.1
```
