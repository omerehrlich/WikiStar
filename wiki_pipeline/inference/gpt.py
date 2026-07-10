"""
Stage 4 (model) — OpenAI Batch API client.

Submits all prompts as a single OpenAI batch job, polls until it completes,
and returns the raw model responses aligned to the input order. Uses the Batch
API (rather than one request per prompt) because it is cheaper and handles
large articles' thousands of prompts in one job.
"""
import json
import os
import tempfile
import time
from typing import Callable, List, Optional

import openai


def _summarize_batch(batch_status) -> dict:
    """Pull the user-relevant fields out of an OpenAI Batch object."""
    counts = getattr(batch_status, "request_counts", None)
    usage = getattr(batch_status, "usage", None) or {}
    completed = getattr(counts, "completed", 0) if counts else 0
    failed = getattr(counts, "failed", 0) if counts else 0
    total_requests = getattr(counts, "total", 0) if counts else 0
    return {
        "batch_id": getattr(batch_status, "id", None),
        "status": getattr(batch_status, "status", None),
        "created_at": getattr(batch_status, "created_at", None),
        "in_progress_at": getattr(batch_status, "in_progress_at", None),
        "completed_at": getattr(batch_status, "completed_at", None),
        "completed": completed,
        "failed": failed,
        "total_requests": total_requests,
        "usage": dict(usage) if isinstance(usage, dict) else {},
    }


class GPT:
    """Thin wrapper over the OpenAI Batch API for prompt classification."""

    def __init__(self, model_name="gpt-5-mini", api_key=None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key not provided and OPENAI_API_KEY environment variable not set."
            )
        self.client = openai.OpenAI(api_key=self.api_key)

    def predict(
        self,
        prompts: List[str],
        status_callback: Optional[Callable[[dict], None]] = None,
    ) -> List[str]:
        """Submit all prompts as one batch job and return responses in order."""
        jsonl_path = self._write_batch_jsonl(prompts)

        try:
            with open(jsonl_path, "rb") as f:
                file_resp = self.client.files.create(file=f, purpose="batch")
        finally:
            # The JSONL is a scratch file; remove it once uploaded.
            try:
                os.remove(jsonl_path)
            except OSError:
                pass
        file_id = file_resp.id

        batch = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        batch_id = batch.id

        # Poll until completion.
        while True:
            batch_status = self.client.batches.retrieve(batch_id)
            print(batch_status)
            if status_callback is not None:
                try:
                    status_callback(_summarize_batch(batch_status))
                except Exception as cb_err:
                    print(f"status_callback error (ignored): {cb_err}")
            if batch_status.status == "completed":
                break
            elif batch_status.status in {"failed", "expired", "cancelled"}:
                raise RuntimeError(f"Batch job failed with status: {batch_status.status}")
            time.sleep(10)

        # Surface any per-request errors.
        if batch_status.error_file_id:
            error_stream = self.client.files.content(batch_status.error_file_id)
            error_log = error_stream.read().decode("utf-8")
            raise RuntimeError(f"Batch job failed. Error log:\n{error_log}")

        # Retrieve and parse the output.
        output_file_id = batch_status.output_file_id
        file_stream = self.client.files.content(output_file_id)
        content = file_stream.read().decode("utf-8")

        responses_by_id = {}
        for line in content.splitlines():
            obj = json.loads(line)
            idx = int(obj["custom_id"])
            text = obj["response"]["body"]["choices"][0]["message"]["content"]
            responses_by_id[idx] = text

        return [responses_by_id.get(i, "No response") for i in range(len(prompts))]

    def _write_batch_jsonl(self, prompts: List[str]) -> str:
        """Write the prompts to a temporary JSONL file for batch upload."""
        fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="wiki_batch_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for idx, prompt in enumerate(prompts):
                body = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                }
                entry = {
                    "custom_id": str(idx),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
                f.write(json.dumps(entry) + "\n")
        return path
