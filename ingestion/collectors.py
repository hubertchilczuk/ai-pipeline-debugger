"""Log collectors for external systems (Airflow, Spark, Kubernetes, AWS CloudWatch, GCP)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from core.config import Settings


@dataclass(slots=True)
class RawLog:
    source: str            # e.g. "airflow", "spark", "k8s", "cloudwatch", "gcp", "file"
    pipeline: str          # DAG / job name
    stage: str             # task / stage id
    content: str           # raw text
    metadata: dict[str, str]


class AirflowCollector:
    def __init__(self, settings: Settings) -> None:
        self._base = (settings.airflow_base_url or "").rstrip("/")

    async def fetch_task_log(self, dag_id: str, run_id: str, task_id: str, try_number: int = 1) -> RawLog:
        if not self._base:
            raise RuntimeError("AIRFLOW_BASE_URL is not configured")
        url = f"{self._base}/api/v1/dags/{dag_id}/dagRuns/{run_id}/taskInstances/{task_id}/logs/{try_number}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return RawLog(
                source="airflow",
                pipeline=dag_id,
                stage=task_id,
                content=resp.text,
                metadata={"run_id": run_id, "try_number": str(try_number)},
            )


class FileCollector:
    """Read logs from local files — useful for tests and offline analysis."""

    @staticmethod
    def read(path: str, pipeline: str = "unknown", stage: str = "unknown") -> RawLog:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return RawLog(
                source="file",
                pipeline=pipeline,
                stage=stage,
                content=fh.read(),
                metadata={"path": path},
            )


class KubernetesCollector:
    """Pull pod logs via the Kubernetes API.

    Requires `kubernetes` package (optional dep). Uses in-cluster config when available,
    falls back to ~/.kube/config locally.
    """

    def __init__(self) -> None:
        try:
            from kubernetes import client, config  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "kubernetes package not installed. pip install kubernetes"
            ) from exc

        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()
        self._api = client.CoreV1Api()

    def fetch_pod_log(
        self,
        namespace: str,
        pod: str,
        container: str | None = None,
        tail_lines: int = 1000,
        pipeline: str | None = None,
        stage: str | None = None,
    ) -> RawLog:
        text: str = self._api.read_namespaced_pod_log(
            name=pod,
            namespace=namespace,
            container=container,
            tail_lines=tail_lines,
        )
        meta: dict[str, str] = {"namespace": namespace, "pod": pod}
        if container:
            meta["container"] = container
        return RawLog(
            source="k8s",
            pipeline=pipeline or pod,
            stage=stage or (container or "default"),
            content=text,
            metadata=meta,
        )


class CloudWatchCollector:
    """Fetch CloudWatch Logs via boto3 (optional dep)."""

    def __init__(self, region_name: str | None = None) -> None:
        try:
            import boto3  # type: ignore
        except ImportError as exc:
            raise RuntimeError("boto3 not installed. pip install boto3") from exc
        self._client = boto3.client("logs", region_name=region_name)

    def fetch_stream(
        self,
        log_group: str,
        log_stream: str,
        limit: int = 500,
        pipeline: str | None = None,
        stage: str | None = None,
    ) -> RawLog:
        events: list[dict[str, Any]] = []
        token: str | None = None
        remaining = limit
        while remaining > 0:
            kwargs: dict[str, Any] = {
                "logGroupName": log_group,
                "logStreamName": log_stream,
                "limit": min(remaining, 100),
                "startFromHead": True,
            }
            if token:
                kwargs["nextToken"] = token
            resp = self._client.get_log_events(**kwargs)
            batch = resp.get("events", [])
            if not batch:
                break
            events.extend(batch)
            remaining -= len(batch)
            new_token = resp.get("nextForwardToken")
            if new_token == token:
                break
            token = new_token

        content = "\n".join(e.get("message", "") for e in events)
        return RawLog(
            source="cloudwatch",
            pipeline=pipeline or log_group,
            stage=stage or log_stream,
            content=content,
            metadata={"log_group": log_group, "log_stream": log_stream},
        )


class GCPLogsCollector:
    """Fetch logs from Google Cloud Logging (optional dep)."""

    def __init__(self, project: str) -> None:
        try:
            from google.cloud import logging as gcp_logging  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "google-cloud-logging not installed. pip install google-cloud-logging"
            ) from exc
        self._client = gcp_logging.Client(project=project)
        self._project = project

    def fetch_entries(
        self,
        filter_: str,
        limit: int = 500,
        pipeline: str | None = None,
        stage: str | None = None,
    ) -> RawLog:
        entries = list(self._client.list_entries(filter_=filter_, max_results=limit))
        lines = []
        for entry in entries:
            payload = entry.payload if isinstance(entry.payload, str) else str(entry.payload)
            ts = entry.timestamp.isoformat() if entry.timestamp else ""
            lines.append(f"{ts} {payload}")
        return RawLog(
            source="gcp",
            pipeline=pipeline or self._project,
            stage=stage or "logs",
            content="\n".join(lines),
            metadata={"project": self._project, "filter": filter_},
        )
