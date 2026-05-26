from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any


_DATA_PACKAGE = "pystamps.data"
ARTIFACT_TOLERANCE_RESOURCE = "artifact_tolerances.json"


@dataclass(frozen=True, slots=True)
class ToleranceRule:
    id: str
    key: str
    dtype: str
    comparison_mode: str
    atol: float
    rtol: float
    period: float | None = None


@dataclass(frozen=True, slots=True)
class ArtifactToleranceSpec:
    path: str
    stage: int
    scope: str
    shape_policy: str
    required_keys: tuple[str, ...]
    rules: tuple[ToleranceRule, ...]

    def rule_for_key(self, key: str) -> ToleranceRule | None:
        for rule in self.rules:
            if rule.key == key:
                return rule
        return None


@dataclass(frozen=True, slots=True)
class ArtifactToleranceManifest:
    version: int
    artifacts: tuple[ArtifactToleranceSpec, ...]

    @property
    def verify_globs(self) -> tuple[str, ...]:
        return tuple(spec.path for spec in self.artifacts)

    def spec_for_path(self, relative_path: str) -> ArtifactToleranceSpec | None:
        normalized = relative_path.replace("\\", "/")
        for spec in self.artifacts:
            if fnmatch.fnmatchcase(normalized, spec.path):
                return spec
        return None


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"artifact tolerance manifest field '{key}' must be a non-empty string")
    return value


def _rule_from_payload(payload: dict[str, Any]) -> ToleranceRule:
    return ToleranceRule(
        id=_required_str(payload, "id"),
        key=_required_str(payload, "key"),
        dtype=_required_str(payload, "dtype"),
        comparison_mode=_required_str(payload, "comparison_mode"),
        atol=float(payload["atol"]),
        rtol=float(payload["rtol"]),
        period=float(payload["period"]) if payload.get("period") is not None else None,
    )


def _spec_from_payload(payload: dict[str, Any]) -> ArtifactToleranceSpec:
    required_keys = payload.get("required_keys")
    if not isinstance(required_keys, list) or not all(isinstance(key, str) for key in required_keys):
        raise ValueError("artifact tolerance manifest field 'required_keys' must be a list of strings")
    rules_payload = payload.get("rules")
    if not isinstance(rules_payload, list):
        raise ValueError("artifact tolerance manifest field 'rules' must be a list")
    rules = tuple(_rule_from_payload(rule) for rule in rules_payload)
    rule_keys = {rule.key for rule in rules}
    missing_rules = sorted(set(required_keys) - rule_keys)
    if missing_rules:
        raise ValueError(f"artifact tolerance manifest has required keys without rules: {missing_rules}")
    return ArtifactToleranceSpec(
        path=_required_str(payload, "path"),
        stage=int(payload["stage"]),
        scope=_required_str(payload, "scope"),
        shape_policy=_required_str(payload, "shape_policy"),
        required_keys=tuple(required_keys),
        rules=rules,
    )


@lru_cache(maxsize=1)
def load_artifact_tolerance_manifest() -> ArtifactToleranceManifest:
    payload = json.loads((files(_DATA_PACKAGE) / ARTIFACT_TOLERANCE_RESOURCE).read_text(encoding="utf-8"))
    artifacts_payload = payload.get("artifacts")
    if not isinstance(artifacts_payload, list):
        raise ValueError("artifact tolerance manifest field 'artifacts' must be a list")
    artifacts = tuple(_spec_from_payload(artifact) for artifact in artifacts_payload)
    rule_ids = [rule.id for spec in artifacts for rule in spec.rules]
    if len(rule_ids) != len(set(rule_ids)):
        raise ValueError("artifact tolerance manifest rule IDs must be unique")
    return ArtifactToleranceManifest(version=int(payload["manifest_version"]), artifacts=artifacts)
