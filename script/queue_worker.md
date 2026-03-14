# Durable Queue Worker

The queue is now host-owned durable state, not a text file that workers destructively pop from.

`QUEUE_FILE` is still the entry point, but it acts as an inbox. The real queue lives beside it in:

- `jobs/<queue>_queue/pending/`
- `jobs/<queue>_queue/running/`
- `jobs/<queue>_queue/completed/`
- `jobs/<queue>_queue/failed/`

Each job stays in one of those directories until it is explicitly completed or failed. A claimed job is represented by a lease file in `running/`, so if a worker disappears after claiming, the job still exists and can be requeued safely.

## Why this is safer

- Claiming a job renames it into `running/` on the host before the worker starts executing it.
- Completing or failing a job requires the matching lease token, so a stale worker cannot overwrite a newer claim.
- Reaping stale workers renames their lease files back into `pending/`; nothing depends on reconstructing state from a second heartbeat file.
- If you keep one host-side worker running with `EXIT_ON_EMPTY=0`, stale remote workers will be reaped even when no other remote worker is alive.

## Host Worker

Use a host worker as the durable queue owner. It should stay up and keep reaping stale leases.

```bash
QUEUE_FILE="jobs/patches_jobs.txt" \
WORKER_ID="host_pro6000" \
GPU_ID=0 \
EXIT_ON_EMPTY=0 \
bash script/queue_worker.sh
```

## Remote Worker

Remote workers claim leases over SSH from the host queue.

```bash
QUEUE_HOST="zli@100.84.104.59" \
QUEUE_FILE="/home/zli/SimpleDeco/jobs/patches_jobs.txt" \
WORKER_ID="research_evcc_h200" \
GPU_ID=0 \
PING_INTERVAL=30 \
bash script/queue_worker.sh --ssh-pass "test1234"
```

When `QUEUE_HOST` is set, `QUEUE_BACKEND_SCRIPT` defaults to:

```bash
SimpleDeco/script/queue_backend.py
```

Override the remote repo root with:

```bash
QUEUE_REMOTE_ROOT="/home/zli/SimpleDeco"
```

## Submission

Existing job-generation scripts can keep writing lines into `QUEUE_FILE`. Workers will import those lines into the durable queue automatically before claiming jobs.

To submit one job directly without touching the inbox file:

```bash
python3 script/queue_append_job.py --file jobs/gpqa_diamond_jobs.txt --job 'python3 my_job.py'
```

## Monitoring

Watch queue state directly:

```bash
QUEUE_FILE="jobs/gpqa_diamond_jobs.txt" \
STALE_AFTER=180 \
bash script/queue_top.sh
```

Run a standalone reaper if you want a dedicated watchdog:

```bash
QUEUE_FILE="jobs/gpqa_diamond_jobs.txt" \
STALE_AFTER=180 \
WATCHDOG_INTERVAL=30 \
python3 script/queue_watchdog.py
```

## Defaults

- `PING_INTERVAL=60`
- `STALE_AFTER=max(3 * PING_INTERVAL, PING_INTERVAL + 60)`
- Local workers default to `EXIT_ON_EMPTY=0`
- Remote workers default to `EXIT_ON_EMPTY=1`

## Behavior notes

- `Ctrl-C` on a worker releases the current lease back to the front of the queue.
- A worker that stops heartbeating is requeued by the next reap cycle.
- If a worker loses its lease, its late `complete` or `fail` call is rejected instead of mutating the replacement lease.
