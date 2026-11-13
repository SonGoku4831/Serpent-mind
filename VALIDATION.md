# Validation

Validated in the managed Linux environment (Python 3.12; Docker image uses 3.11).

- Four automated tests pass: food/collision rules, tail-cell movement, Q-learning
  update and model round trip, and complete API game plus metrics.
- Ray generated 100 deterministic seeded rollout episodes.
- DVC executed both pipeline stages and skipped them on a subsequent unchanged run.
- MLflow stored parameters, training metrics, model artifacts and evaluation.
- After 30,000 training episodes, held-out 100-game mean score was **16.58**;
  the untrained greedy Q-table scored **0.06**.
- Chromium browser automation clicked Play and watched a complete game with
  seed 42: **20 food items**. The board rendered and the game ended successfully.

These scores describe the fixed evaluation seeds and this implementation, not a
promise of a score on every game. The target is a 4 GB / 2 CPU machine; this
managed environment has more RAM. Compose sets service memory limits and separates
training from the live stack to accommodate the smaller target.

- The final shared Docker image built successfully with the pinned dependencies.
- Full Docker Compose execution and the configured memory limits could not be
  validated here: this executor has an 8 GB filesystem and Docker uses the vfs
  storage driver, which exhausted disk space during container creation. The
  temporary test image and Python virtual environment were removed afterward.
  Use Ubuntu's normal Docker storage setup and allow at least 6 GB free disk
  space, preferably 10 GB. Prometheus/Grafana configuration is included but live
  scraping and dashboard provisioning were not verified in this environment.
