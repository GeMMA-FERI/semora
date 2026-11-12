"""Bounded provenance values; publication and human review are separate facts."""

from __future__ import annotations

import json
import re


def annotation(actor):
    reviewed = re.fullmatch(r"admin:[1-9]\d*", actor) is not None
    return {"origin": "human" if reviewed else "unknown", "human_reviewed": reviewed, "provenance": {}}


def autonomous_annotation(provenance):
    if not isinstance(provenance, dict):
        raise ValueError("Autonomous provenance must be an object.")
    policy = provenance.get("policy_version")
    if not isinstance(policy, str) or not 1 <= len(policy.strip()) <= 128 or "\x00" in policy:
        raise ValueError("Autonomous provenance requires policy_version.")
    try:
        encoded = json.dumps(provenance, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Autonomous provenance must contain JSON values.") from error
    if len(encoded.encode()) > 16 * 1024:
        raise ValueError("Autonomous provenance exceeds 16 KiB.")
    return {"origin": "autonomous", "human_reviewed": False, "provenance": json.loads(encoded)}


def normalize_group(group):
    """Old audit snapshots did not contain annotations; treat them conservatively."""
    if group is None:
        return None
    result = dict(group)
    result.setdefault("origin", "unknown")
    result.setdefault("human_reviewed", False)
    result.setdefault("provenance", {})
    result.setdefault(
        "member_provenance",
        {name: {"origin": "unknown", "human_reviewed": False, "provenance": {}} for name in result["members"]},
    )
    return result
