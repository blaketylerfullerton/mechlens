"""Local, OpenAI-compatible candidate-label generation for trained SAEs."""
from __future__ import annotations

import hashlib
import json
import os
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PROMPT_VERSION = "auto-interp-v1"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Profile:
    name: str
    base_url: str
    model: str
    timeout_s: int = 120
    response_mode: str = "json_schema"
    api_key_env: str | None = None

    def public(self) -> dict:
        return dict(name=self.name, provider="openai_compatible", base_url=self.base_url,
                    model=self.model, timeout_s=self.timeout_s, response_mode=self.response_mode)


def load_profiles(root: Path) -> dict[str, Profile]:
    """Read local server config; secrets remain environment-variable references."""
    config_dir = Path(os.environ.get("MECHLENS_CONFIG_DIR", root / "config"))
    path = config_dir / "config.toml"
    if not path.is_file():
        return {}
    raw = tomllib.loads(path.read_text())
    profiles = raw.get("auto_interp", {}).get("profile", {})
    result = {}
    for name, value in profiles.items():
        if value.get("provider", "openai_compatible") != "openai_compatible":
            continue
        base_url, model = value.get("base_url"), value.get("model")
        if isinstance(base_url, str) and isinstance(model, str):
            result[name] = Profile(name, base_url.rstrip("/"), model,
                int(value.get("timeout_s", 120)), str(value.get("response_mode", "json_schema")),
                value.get("api_key_env"))
    return result


def _schema() -> dict:
    return {"name": "feature_explanation", "schema": {"type": "object", "additionalProperties": False,
        "required": ["label", "summary", "evidence", "uncertainty"], "properties": {
            "label": {"type": "string", "maxLength": 160},
            "summary": {"type": "string", "maxLength": 800},
            "evidence": {"type": "array", "items": {"type": "integer"}, "maxItems": 20},
            "uncertainty": {"type": "string", "maxLength": 500},
        }}}


def candidate_request(artifact_id: str, feature_id: int, examples: list[dict]) -> tuple[dict, str]:
    evidence = [{"index": i, "activation": row["activation"], "context": row["context"]}
                for i, row in enumerate(examples)]
    prompt = ("Infer a narrow hypothesis for one sparse autoencoder feature from activation examples. "
              "Do not claim causality or certainty. Return JSON matching the supplied schema.\n"
              f"Checkpoint: {artifact_id}\nFeature: {feature_id}\nExamples: {json.dumps(evidence)}")
    payload = dict(model=None, messages=[{"role": "user", "content": prompt}], temperature=0,
                   response_format={"type": "json_schema", "json_schema": _schema()})
    fingerprint = hashlib.sha256(json.dumps(dict(prompt_version=PROMPT_VERSION, artifact_id=artifact_id,
        feature_id=feature_id, examples=evidence), sort_keys=True).encode()).hexdigest()
    return payload, fingerprint


def generate(profile: Profile, artifact_id: str, feature_id: int, examples: list[dict]) -> dict:
    payload, fingerprint = candidate_request(artifact_id, feature_id, examples)
    payload["model"] = profile.model
    request = urllib.request.Request(profile.base_url + "/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    if profile.api_key_env and (secret := os.environ.get(profile.api_key_env)):
        request.add_header("Authorization", f"Bearer {secret}")
    try:
        with urllib.request.urlopen(request, timeout=profile.timeout_s) as response:
            body = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ValueError(f"Local inference request failed: {exc}") from exc
    try:
        result = json.loads(body["choices"][0]["message"]["content"])
        if not all(isinstance(result[key], expected) for key, expected in
                   (("label", str), ("summary", str), ("evidence", list), ("uncertainty", str))):
            raise ValueError("response fields have invalid types")
    except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Local inference returned invalid structured JSON: {exc}") from exc
    return dict(candidate=result, request_fingerprint=fingerprint, prompt_version=PROMPT_VERSION,
                schema_version=SCHEMA_VERSION, profile=profile.public())
