# Queue Worker Heartbeats

This worker runs jobs from a shared queue on a **host** machine over SSH and keeps a heartbeat file on the host so the host can requeue stale jobs.

## What to run on the host

Nothing is required for the system to work. The host only needs the queue files and scripts:

- `jobs/benchmarkname_jobs.txt`
- `jobs/failed_jobs.txt`
- `jobs/worker_state.json` (created automatically)
- `script/queue_pop_job.py`
- `script/queue_append_job.py`
- `script/queue_worker_state.py`

Optional (for monitoring):

```
QUEUE_FILE="jobs/hmmt25_jobs.txt" \
WORKER_STATE_FILE="jobs/worker_state.json" \
STALE_AFTER=1200 \
bash script/queue_top.sh
```

## Host Node

```
QUEUE_FILE="jobs/gpqa_diamond_jobs.txt" \
WORKER_ID="host_pro6000" \
GPU_ID=0 \
bash script/queue_worker.sh
```

## Worker Node

```
QUEUE_HOST="zli@100.84.104.59" \
QUEUE_FILE="jobs/gpqa_diamond_jobs.txt" \
WORKER_ID="modal_h200_1" \
GPU_ID=0 \
bash script/queue_worker.sh --ssh-pass "test1234"
```

## Defaults you no longer need to pass

When `QUEUE_HOST` is set, these default to the remote repo root:

- `QUEUE_POP_SCRIPT` → `SimpleDeco/script/queue_pop_job.py`
- `QUEUE_APPEND_SCRIPT` → `SimpleDeco/script/queue_append_job.py`
- `QUEUE_WORKER_STATE_SCRIPT` → `SimpleDeco/script/queue_worker_state.py`
- `FAILED_FILE` → `SimpleDeco/jobs/failed_jobs.txt`
- `WORKER_STATE_FILE` → `SimpleDeco/jobs/worker_state.json`

Override the remote base path with:

```
QUEUE_REMOTE_ROOT="/home/zli/SimpleDeco"
```

Heartbeat defaults:

- `PING_INTERVAL=600` (seconds)
- `STALE_AFTER=2*PING_INTERVAL` (at least `PING_INTERVAL + 60`)

## What the script paths are used for

The worker does **not** run the host’s Python directly. It SSHes to the host and runs the host’s queue scripts there:

- `queue_pop_job.py`: pop one job line from the host’s queue file
- `queue_append_job.py`: append (or prepend) a job line to the host’s queue file
- `queue_worker_state.py`: update worker heartbeat on the host and requeue stale jobs

All queue state lives on the **host**.

## Behavior notes

- Ctrl‑C on a worker: stops the current job, requeues it to the front, then exits.
- If a worker stops pinging past `STALE_AFTER`, the host requeues its last job on the next ping from any worker (or when you run any command that calls `queue_worker_state.py --reap-stale`).
