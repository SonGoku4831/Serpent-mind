# Serpent Mind — Snake AI + minimal AIOps

A Python Snake agent trained with tabular Q-learning, using **Ray** for data preparation,
**MLflow** for experiment tracking, **DVC** for data/model versioning, **FastAPI** for
serving, **Streamlit** for the game UI, **Docker Compose** for deployment, and
**Prometheus + Grafana** for monitoring. This project uses Ray, not Spark.

These instructions follow the successful setup on an **Ubuntu 24.04 VM in VirtualBox
on Windows**, with approximately **4 GB RAM and 2 CPU cores**. No GPU, CUDA,
Pygame, Ubuntu desktop, or Ubuntu browser is needed. A Windows browser can show
the application through the SSH tunnel below.

## Documentation

- **Installation and operating commands:** this README.
- **Code and developer guide:** [CODE_DOCUMENTATION.md](CODE_DOCUMENTATION.md) — architecture, file responsibilities, game rules, Q-learning, data/model formats, API examples, metrics and maintenance.
- **Original validation report:** [VALIDATION.md](VALIDATION.md).

## Already set up? Start here

If you have already built the image and completed training, run this **inside Ubuntu**:

```bash
cd ~/Serpent-mind
sudo systemctl start docker
sudo docker compose up -d --wait mlflow api ui prometheus grafana
```

If using Windows with the VirtualBox SSH forwarding rule already configured,
open **Windows PowerShell** and run the following as **one complete line**.
The example Ubuntu username is `vishnu`; replace it if your username differs.

```powershell
ssh -p 2222 -L 8501:127.0.0.1:8501 vishnu@127.0.0.1
```

Keep PowerShell connected and the VM running. Open **http://localhost:8501**
in your Windows browser and click **Play**. You do not need to rebuild or retrain
for each game. If SSH forwarding has not been configured yet, follow section 6.

## First-time setup

Run each command in order and wait for it to finish. Unless a step explicitly says
**Windows PowerShell** or **VirtualBox**, commands belong in the **Ubuntu VM terminal**.
Do not type the terminal prompt, and do not split a one-line command across separate
Enter presses. If a command fails, stop and inspect its error before moving on.

### 1. Check the VM

```bash
free -h
nproc
df -h /
```

Target approximately 4 GB RAM, two CPU cores, and **10 GB free disk space**.
The demonstrated VM had 3.8 GiB RAM, two cores, existing swap, and 24 GB available disk.
Close unnecessary applications while building or training. Build and download times
depend heavily on the VM and internet connection; there is no guaranteed duration.

### 2. Install Docker and the required tools on Ubuntu 24.04

This guide uses **Ubuntu's packages**, not Docker's separate package repository.
Ubuntu names the Buildx package `docker-buildx`, not `docker-buildx-plugin`.

```bash
sudo apt-get update
sudo apt-get install -y git docker.io docker-buildx docker-compose-v2
sudo systemctl enable --now docker
```

Verify the installation:

```bash
sudo docker version
sudo docker buildx version
sudo docker compose version
sudo docker run --rm hello-world
```

The last command should print **Hello from Docker!** Use `sudo docker` consistently
throughout this guide; joining the Docker group is not required.

If Ubuntu cannot locate `docker-buildx` or `docker-compose-v2`, enable its Universe
repository, then repeat the installation:

```bash
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y universe
sudo apt-get update
sudo apt-get install -y git docker.io docker-buildx docker-compose-v2
```

These package instructions target Ubuntu 24.04. If you already installed Docker CE
from Docker's official repository, keep that installation and its corresponding
plugins; do not mix the two package installation methods.

### 3. Download the project and build once

For a new checkout:

```bash
cd ~
git clone https://github.com/SonGoku4831/Serpent-mind.git
cd ~/Serpent-mind
```

If the repository is already present, use this instead of cloning again:

```bash
cd ~/Serpent-mind
git pull --ff-only
```

Build the shared image:

```bash
sudo docker compose --progress plain build trainer
```

Wait for `snake-aiops:local Built` and the terminal prompt to return.
All Python services use this same image.

The current Dockerfile includes Git, trusts the mounted `/app` repository for Git,
and uses pip 25.2 with download caching, a longer timeout, and resumable downloads.
Do not delete the cache or add `--no-cache` when retrying a network failure.
The pip root-user warning during this container build is not itself a build failure.

The repository already includes `.dvc` configuration, so **do not run `dvc init` again**.
Python and the project libraries are installed inside the image; no separate host
Python, pip, or virtual environment installation is required for this Docker route.

### 4. Start MLflow, then prepare data and train

Keep the game and monitoring services stopped during training on a small VM.
On the first run they have not been started yet.

```bash
sudo docker compose up -d --wait mlflow
sudo docker compose run --rm trainer dvc repro
```

Use that training command exactly. **Do not add `--user "$(id -u):$(id -g)"` or
`-e HOME=/tmp`**: the earlier numeric-user command caused
`getpwuid(): uid not found: 1000` in this image. The current image's default user
works with DVC, and Git is installed and configured in the image.

Successful output includes:

- `Prepared 100 episodes using Ray`.
- Training with `--episodes 30000 --seed 42`.
- Evaluation scores and an MLflow run URL.
- An updated `dvc.lock` and the terminal prompt returning without an error.

The demonstrated run produced:

```json
{
  "eval_mean_score": 16.58,
  "untrained_mean_score": 0.06
}
```

These are results from that run, not a promised score on every game.
DVC's suggestions to use `git add`, enable auto-staging, or configure a remote
are informational; they are not errors. If the pipeline is unchanged and its
outputs are already present, `dvc repro` may correctly skip the stages.

### 5. Start the application and monitoring

Only after training completes successfully:

```bash
sudo docker compose up -d --wait api ui prometheus grafana
sudo docker compose ps
```

The first start downloads the Prometheus and Grafana images. Wait for the command
to finish successfully. MLflow remains running from the previous step.

If Ubuntu has its own browser, open **http://localhost:8501** inside Ubuntu.
For a terminal-only VM, continue below to use the Windows browser.

### 6. Open the application from Windows

#### In the Ubuntu VM: enable SSH once

These are separate commands; do not add a backslash after the first line.

```bash
sudo apt-get install -y openssh-server
sudo systemctl enable --now ssh
systemctl is-active ssh
whoami
```

The SSH status should be `active`. Note the Ubuntu username printed by `whoami`.

#### In VirtualBox on Windows: add an SSH forwarding rule once

Open the VM's **Settings → Network → Adapter 1**.
The demonstrated adapter uses **NAT**. Click **Port Forwarding**.
Some versions put that button under **Advanced**; in Expert mode it may appear
directly below **Virtual Cable Connected**.

Add this rule, then save both dialogs:

| Field | Value |
|---|---|
| Name | SSH |
| Protocol | TCP |
| Host IP | `127.0.0.1` |
| Host Port | `2222` |
| Guest IP | Leave blank |
| Guest Port | `22` |

This rule applies to NAT. If your adapter uses a different network mode, do not
change it blindly; the host-to-VM connection method will differ.

#### In Windows PowerShell: open the tunnel

Replace `vishnu` with the Ubuntu username from `whoami` if necessary.
Paste the entire command as **one line**, then press Enter once:

```powershell
ssh -p 2222 -L 8501:127.0.0.1:8501 vishnu@127.0.0.1
```

For the first connection to your own VM, confirm its host-key prompt with `yes`.
Enter the **Ubuntu login password**, not the Windows password. No characters appear
while typing the password; press Enter when finished.

Leave PowerShell connected and the VM running. Open **http://localhost:8501** in
your **Windows browser** and click **Play**. After a game ends, click Play again,
or change the seed for a different food layout.

Windows and Ubuntu have separate loopback interfaces. Windows's `localhost:8501`
reaches the game only while this SSH tunnel is connected. Forwarding VirtualBox
port 8501 directly to the guest is not equivalent here, because Compose publishes
the service only on the VM's loopback address.

#### Optional: access MLflow, Grafana, Prometheus and API docs too

In the PowerShell window with the active SSH session, type `exit` and press Enter
to close the game-only tunnel. Then use this **single-line** command instead:

```powershell
ssh -p 2222 -L 8501:127.0.0.1:8501 -L 5000:127.0.0.1:5000 -L 3000:127.0.0.1:3000 -L 9090:127.0.0.1:9090 -L 8000:127.0.0.1:8000 vishnu@127.0.0.1
```

With that tunnel connected, these URLs work in Windows. They also work directly
in an Ubuntu browser without a tunnel.

| Service | URL |
|---|---|
| Snake game | http://localhost:8501 |
| MLflow experiments | http://localhost:5000 |
| Grafana dashboard | http://localhost:3000/d/snake |
| Prometheus | http://localhost:9090 |
| FastAPI documentation | http://localhost:8000/docs |

Grafana is provisioned automatically and permits anonymous local viewing.
Play a complete game, then allow a few Prometheus scrapes for metrics to appear.
The `http://mlflow:5000` address printed during training is a container-network
address; use `http://localhost:5000` in your browser through the tunnel.

## Stop, start again, and retrain

### Stop without deleting saved data

Inside Ubuntu:

```bash
cd ~/Serpent-mind
sudo docker compose down
```

This preserves the model files and named volumes holding tracking and monitoring
data. Do not add `-v` if you want to keep those volumes. Close the SSH session
with `exit` when finished.

### Start later without building or training again

Inside Ubuntu:

```bash
cd ~/Serpent-mind
sudo systemctl start docker
sudo docker compose up -d --wait mlflow api ui prometheus grafana
```

Then reconnect the Windows SSH tunnel from section 6 and open the game.

### Retrain when you actually want a new run

Inside Ubuntu:

```bash
cd ~/Serpent-mind
sudo docker compose stop api ui prometheus grafana
sudo docker compose up -d --wait mlflow
sudo docker compose run --rm trainer dvc repro --force
```

Wait for successful completion before starting the app again:

```bash
sudo docker compose up -d --wait api ui prometheus grafana
sudo docker compose restart api
```

The API reads the model at startup, so restarting it loads the newly trained model.
Edit `--episodes` or `--seed` in `dvc.yaml` to change the training settings.
A normal `dvc repro` reruns affected stages; `--force` explicitly reruns them.

## Troubleshooting the errors encountered during setup

| Message or symptom | Meaning and action |
|---|---|
| Cannot locate `docker-buildx-plugin` | Ubuntu's package is `docker-buildx`. Use section 2. |
| `docker: unknown command: docker buildx` | Install `docker-buildx`, then verify its version. |
| `ReadTimeoutError` from `files.pythonhosted.org` | Package download stalled. Pull the current Dockerfile and retry the cached build below. Resuming helps, but still requires a working internet connection. |
| `getpwuid(): uid not found: 1000` | Use `sudo docker compose run --rm trainer dvc repro` without a numeric `--user` override. |
| `git: not found` inside the trainer | Pull and rebuild the current image, which now includes Git. Installing Git only on the VM does not install it inside the container. |
| SSH prints its usage text | The destination was omitted or entered separately. Paste the complete SSH command on one line. |
| SSH says connection refused | Confirm SSH is `active`, the VM is running, and the VirtualBox NAT rule forwards host 2222 to guest 22. |
| Windows browser cannot reach the game | Keep SSH connected; check the containers and tunnel. |
| SSH says local port 8501 is already in use | An existing tunnel or another app may already use that port. Reuse or close the previous tunnel before starting another. |
| CPU soft-lockup/watchdog messages | The VM stalled. These messages alone do not prove low RAM or a Python error; inspect VM/host resource usage if they recur. |

For an older checkout or a failed download, run these in Ubuntu:

```bash
cd ~/Serpent-mind
git pull --ff-only
sudo docker compose --progress plain build trainer
```

If `git pull` reports local changes or conflicts, stop and inspect them; do not
discard your work. After the build succeeds, continue with the relevant MLflow
and training steps. Do not reinstall Docker or erase caches for a pip timeout.

To inspect a startup failure:

```bash
sudo docker compose ps -a
sudo docker compose logs --tail=100 mlflow api ui prometheus grafana
```

For a saved build log when diagnosing repeated failures:

```bash
sudo docker compose --progress plain build trainer 2>&1 | tee build.log
```

The success or failure is shown in the log; the shell pipeline's exit status alone
does not establish that the build succeeded.

## Tests and version tracking

Run the automated tests inside Ubuntu, using the existing image:

```bash
cd ~/Serpent-mind
sudo docker compose run --rm --no-deps trainer python -m pytest -q
```

The trainer's default container user can create root-owned files in the mounted
project. Before editing or committing generated training files as your Ubuntu
user, restore their ownership:

```bash
sudo chown -R "$(id -u):$(id -g)" .dvc data models dvc.lock
```

Optionally record the pipeline hashes and evaluation metrics in Git. This is not
required to watch the game. Use your own Git author name and email:

```bash
git add dvc.yaml dvc.lock models/metrics.json
git -c user.name="Your Name" -c user.email="you@example.com" commit -m "Record Snake training run"
```

Every actual training execution creates an MLflow run containing parameters, the
dataset SHA-256, training metrics, evaluation scores and model artifacts. A cached
DVC run skips training and does not create a new MLflow run. MLflow's named volume
keeps its SQLite database and artifacts across normal restarts.

DVC caches rollout data, model weights and model metadata locally. Git records
the pipeline, lock file and evaluation metrics. No DVC remote is configured:
`dvc push` is not a required setup step, and the local cache alone is not a backup.
To share historical model/data files with another machine, configure a remote
yourself and use `dvc push` / `dvc pull`.

To restore files for an older Git revision when its DVC cache is available, first
stop the API, check out the desired Git revision, then run:

```bash
sudo docker compose run --rm --no-deps trainer dvc checkout
sudo docker compose up -d --wait api
sudo docker compose restart api
```

If the required cache is missing, reproduce training with MLflow running instead.

## Demonstrated result and video

The setup was completed on the small Ubuntu VM: Ray prepared the data, training
reported a mean evaluation score of **16.58** versus **0.06** for the untrained
agent, and the Streamlit game played in a Windows browser over SSH.

[Watch or download the full walkthrough and gameplay video](videos/SerpentMind.mp4)
(approximately 2 minutes 57 seconds; silent).

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
