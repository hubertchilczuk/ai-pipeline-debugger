"""Generate synthetic broken logs for tests / demos."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

TEMPLATES = [
    """[2025-{m:02d}-{d:02d} 03:{mm:02d}:21] ERROR Executor: Exception in task 0.0 in stage 4.0
org.apache.spark.SparkException: Job aborted due to stage failure: Task 0 in stage 4.0 failed 4 times,
most recent failure: Lost task 0.0 in stage 4.0 (TID 12, executor 7):
java.lang.OutOfMemoryError: Java heap space
  at java.util.Arrays.copyOf(Arrays.java:3236)
""",
    """[2025-{m:02d}-{d:02d} 09:{mm:02d}:00,123] {{taskinstance.py:1463}} ERROR - Task failed with exception
airflow.exceptions.AirflowException: Bash command failed. Exit code 2.
Traceback (most recent call last):
  File "/opt/airflow/dags/etl.py", line 87, in run
    raise AirflowException('connection refused')
""",
    """Traceback (most recent call last):
  File "/app/features.py", line 42, in build
    return df['user_id'].fillna(0)
KeyError: 'user_id'
""",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/raw_logs"))
    p.add_argument("--count", type=int, default=10)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for i in range(args.count):
        tpl = random.choice(TEMPLATES)
        body = tpl.format(m=random.randint(1, 12), d=random.randint(1, 28), mm=random.randint(0, 59))
        (args.out / f"mock_{i:03d}.log").write_text(body, encoding="utf-8")
    print(f"Wrote {args.count} mock logs to {args.out}")


if __name__ == "__main__":
    main()
