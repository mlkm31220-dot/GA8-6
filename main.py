"""
Content-addressed ML pipeline controller.

DAG: verify_data -> prepare -> train -> evaluate -> register -> publish

Run:
    python3 main.py
Serves POST /pipeline on http://0.0.0.0:$PORT (default 8080)
"""
import copy
import hashlib
import json
import os
import threading

from flask import Flask, request, jsonify

app = Flask(__name__)

NODES = ["verify_data", "prepare", "train", "evaluate", "register", "publish"]
PARENT = {
    "verify_data": None,
    "prepare": "verify_data",
    "train": "prepare",
    "evaluate": "train",
    "register": "evaluate",
    "publish": "register",
}
# Fields (in order) hashed for each node's content-addressed key.
# Fields ending in "Artifact" are resolved from the parent's bound artifact digest,
# everything else is read straight from the request's `inputs`.
NODE_FIELDS = {
    "verify_data": ["generation", "checksum"],
    "prepare": ["canonicalData", "prepareCode", "prepareConfig"],
    "train": ["prepareArtifact", "trainCode", "trainConfig", "runtime"],
    "evaluate": ["trainArtifact", "canonicalData", "evaluateCode", "evaluateConfig"],
    "register": ["evaluateArtifact", "schemaDigest"],
    "publish": ["registerArtifact", "publishConfig"],
}
REQUIRED_INPUT_FIELDS = [
    "generation", "checksum", "canonicalData", "prepareCode", "prepareConfig",
    "trainCode", "trainConfig", "runtime", "evaluateCode", "evaluateConfig",
    "schemaDigest", "publishConfig",
]
STATUSES = {"started", "succeeded", "retryable_failed", "terminal_failed"}
EVENT_FIELDS = {"eventId", "revision", "node", "attempt", "status", "key", "artifactDigest", "receiptId"}

_LOCK = threading.Lock()
_SESSIONS = {}  # session -> SessionState


class ConflictError(Exception):
    def __init__(self, code):
        self.code = code


def is_positive_safe_int(v):
    return isinstance(v, int) and not isinstance(v, bool) and 0 < v <= 2**53 - 1


def is_non_empty_str(v):
    return isinstance(v, str) and len(v) > 0


def canonical_json(obj):
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def sha256_of_array(values):
    compact = json.dumps(values, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(compact.encode("utf-8")).hexdigest()


class SessionState:
    def __init__(self):
        self.revisions = {}  # revision_number -> inputs dict (for conflict detection)
        self.current_revision = None
        self.current_inputs = None
        # transient per-node attempt/terminal state, cleared on new revision
        # node -> {"status": started|retryable_failed|terminal_failed, "attempt": int, "eventId": str}
        self.node_state = {}
        # permanent content-addressed cache: (node, key) -> {"artifactDigest", "eventId", "receiptId"}
        self.cache = {}
        # permanently bound accepted events: eventId -> canonical_json(event)
        self.event_log = {}


def validate_request_shape(body):
    if not isinstance(body, dict):
        raise ConflictError("INVALID_REQUEST")
    if not is_non_empty_str(body.get("session")):
        raise ConflictError("INVALID_REQUEST")
    if not is_positive_safe_int(body.get("revision")):
        raise ConflictError("INVALID_REQUEST")
    inputs = body.get("inputs")
    if not isinstance(inputs, dict):
        raise ConflictError("INVALID_REQUEST")
    for f in REQUIRED_INPUT_FIELDS:
        if not is_non_empty_str(inputs.get(f)):
            raise ConflictError("INVALID_REQUEST")
    events = body.get("events")
    if not isinstance(events, list):
        raise ConflictError("INVALID_REQUEST")
    return body["session"], body["revision"], inputs, events


def validate_event_shape(ev):
    if not isinstance(ev, dict) or set(ev.keys()) != EVENT_FIELDS:
        raise ConflictError("INVALID_EVENT")
    if not is_non_empty_str(ev.get("eventId")):
        raise ConflictError("INVALID_EVENT")
    if not is_positive_safe_int(ev.get("revision")):
        raise ConflictError("INVALID_EVENT")
    if not is_non_empty_str(ev.get("node")):
        raise ConflictError("INVALID_EVENT")
    if not is_positive_safe_int(ev.get("attempt")):
        raise ConflictError("INVALID_EVENT")
    if not isinstance(ev.get("status"), str):
        raise ConflictError("INVALID_EVENT")
    if not is_non_empty_str(ev.get("key")):
        raise ConflictError("INVALID_EVENT")
    ad = ev.get("artifactDigest")
    if ad is not None and not is_non_empty_str(ad):
        raise ConflictError("INVALID_EVENT")
    rid = ev.get("receiptId")
    if rid is not None and not is_non_empty_str(rid):
        raise ConflictError("INVALID_EVENT")


def compute_node_key(node, inputs, cache, current_keys):
    """Returns (key_or_None, resolved_field_values_dict)."""
    fields = NODE_FIELDS[node]
    values = []
    resolved = {}
    for f in fields:
        if f in inputs:
            val = inputs[f]
        else:
            parent = f[: -len("Artifact")]
            parent_key = current_keys.get(parent)
            if parent_key is None:
                return None, resolved
            parent_entry = cache.get((parent, parent_key))
            if parent_entry is None:
                return None, resolved
            val = parent_entry["artifactDigest"]
        values.append(val)
        resolved[f] = val
    if len(values) != len(fields):
        return None, resolved
    return sha256_of_array(values), resolved


def compute_all_keys(inputs, cache):
    current_keys = {}
    for node in NODES:
        key, _ = compute_node_key(node, inputs, cache, current_keys)
        current_keys[node] = key
    return current_keys


def apply_revision(state, revision, inputs):
    """Mutates working `state` copy. Raises ConflictError on REVISION_CONFLICT."""
    if revision in state.revisions:
        if canonical_json(state.revisions[revision]) != canonical_json(inputs):
            raise ConflictError("REVISION_CONFLICT")
        # idempotent re-declaration of an already-seen revision: no state change
        state.current_revision = revision
        state.current_inputs = state.revisions[revision]
        return
    # brand new revision: replace inputs, clear transient attempt/terminal state
    state.revisions[revision] = inputs
    state.current_revision = revision
    state.current_inputs = inputs
    state.node_state = {}


def process_event(state, ev):
    """Returns 'accepted' or 'ignored'. Raises ConflictError for hard 409s."""
    eid = ev["eventId"]

    # --- global event id replay / conflict check ---
    if eid in state.event_log:
        if state.event_log[eid] == canonical_json(ev):
            return "ignored"  # exact replay
        raise ConflictError("EVENT_ID_CONFLICT")

    # --- ignore rules (well-formed but not applicable) ---
    if ev["revision"] != state.current_revision:
        return "ignored"  # stale/old revision
    if ev["node"] not in NODES:
        return "ignored"  # wrong node
    if ev["status"] not in STATUSES:
        return "ignored"  # invalid status value

    node = ev["node"]
    status = ev["status"]
    attempt = ev["attempt"]
    artifact = ev.get("artifactDigest")
    receipt = ev.get("receiptId")

    # artifact/receipt shape rules relative to status/node
    if status == "succeeded":
        if not is_non_empty_str(artifact):
            return "ignored"  # invalid artifact
    else:
        if artifact is not None:
            return "ignored"  # invalid artifact (must be null)

    if node in ("register", "publish") and status == "succeeded":
        expected_receipt = f"receipt:{node}:{ev['key']}"
        if receipt != expected_receipt:
            return "ignored"  # invalid receipt
    else:
        if receipt is not None:
            return "ignored"  # invalid receipt (must be null)

    # current computed cache key for this node, given already-applied events in this batch
    current_keys = compute_all_keys(state.current_inputs, state.cache)
    expected_key = current_keys.get(node)
    if expected_key is None:
        return "ignored"  # unavailable parent
    if ev["key"] != expected_key:
        return "ignored"  # wrong key

    cur = state.node_state.get(node)  # None or dict
    cache_entry = state.cache.get((node, expected_key))

    # --- cached/succeeded handling (per content-addressed key) ---
    if cache_entry is not None:
        if status == "succeeded":
            if artifact == cache_entry["artifactDigest"]:
                return "ignored"  # redundant success, already bound
            raise ConflictError("EVIDENCE_CONFLICT")
        raise ConflictError("STATUS_CONFLICT")

    # --- transient state machine (started / retryable_failed / terminal_failed / none) ---
    if cur is None:
        if status == "started" and attempt == 1:
            state.node_state[node] = {"status": "started", "attempt": 1, "eventId": eid}
            _log_and_accept(state, ev)
            return "accepted"
        return "ignored"  # completion or attempt > 1 with no prior start

    n = cur["attempt"]

    if cur["status"] == "started":
        if attempt < n:
            return "ignored"
        if status in ("succeeded", "retryable_failed", "terminal_failed") and attempt == n:
            if status == "succeeded":
                state.cache[(node, expected_key)] = {
                    "artifactDigest": artifact,
                    "eventId": eid,
                    "receiptId": receipt,
                }
                state.node_state[node] = {"status": "succeeded", "attempt": n, "eventId": eid}
            else:
                state.node_state[node] = {"status": status, "attempt": n, "eventId": eid}
            _log_and_accept(state, ev)
            return "accepted"
        raise ConflictError("STATUS_CONFLICT")

    if cur["status"] == "retryable_failed":
        if attempt < n:
            return "ignored"
        if status == "started" and attempt == n + 1:
            state.node_state[node] = {"status": "started", "attempt": n + 1, "eventId": eid}
            _log_and_accept(state, ev)
            return "accepted"
        raise ConflictError("STATUS_CONFLICT")

    if cur["status"] == "terminal_failed":
        raise ConflictError("STATUS_CONFLICT")

    # cur["status"] == "succeeded" but no cache_entry found is inconsistent; treat defensively
    raise ConflictError("STATUS_CONFLICT")


def _log_and_accept(state, ev):
    state.event_log[ev["eventId"]] = canonical_json(ev)


def build_response(state):
    cache = state.cache
    inputs = state.current_inputs
    current_keys = compute_all_keys(inputs, cache)

    nodes_out = []
    blocked_upstream_terminal = set()
    blocked_upstream_pending = set()

    # precompute per-node readiness classification, walking DAG order (parents already computed)
    node_status_kind = {}  # node -> "cached" | "running" | "terminal" | "pending" (not cached)
    for node in NODES:
        parent = PARENT[node]
        entry = cache.get((node, current_keys.get(node))) if current_keys.get(node) else None
        cur = state.node_state.get(node)
        if entry is not None:
            node_status_kind[node] = "cached"
        elif cur is not None and cur["status"] == "started":
            node_status_kind[node] = "running"
        elif cur is not None and cur["status"] == "terminal_failed":
            node_status_kind[node] = "terminal"
        else:
            node_status_kind[node] = "pending"  # includes none / retryable_failed

    for node in NODES:
        parent = PARENT[node]
        key, resolved = compute_node_key(node, inputs, cache, current_keys)
        cache_entry = cache.get((node, key)) if key else None
        cur = state.node_state.get(node)

        dep_digests = dict(resolved)
        dep_digests["cacheKey"] = key

        # upstream blocking checks first
        ancestor_terminal = parent is not None and node_status_kind.get(parent) == "terminal"
        ancestor_pending = parent is not None and node_status_kind.get(parent) not in ("cached",)

        if ancestor_terminal:
            action, reason, trig = "block", "UPSTREAM_TERMINAL", []
        elif cache_entry is not None:
            action, reason, trig = "reuse", "CACHE_HIT", [cache_entry["eventId"]]
        elif cur is not None and cur["status"] == "started":
            action, reason, trig = "block", "RUNNING", [cur["eventId"]]
        elif cur is not None and cur["status"] == "terminal_failed":
            action, reason, trig = "block", "TERMINAL_FAILURE", [cur["eventId"]]
        elif ancestor_pending:
            action, reason, trig = "block", "UPSTREAM_PENDING", []
        elif cur is not None and cur["status"] == "retryable_failed":
            action, reason, trig = "rerun", "RETRYABLE_FAILURE", [cur["eventId"]]
        else:
            action, reason, trig = "rerun", "CACHE_MISS", []

        nodes_out.append({
            "node": node,
            "action": action,
            "reasonCodes": [reason],
            "dependencyDigests": dep_digests,
            "triggeringEventIds": trig,
        })

    return nodes_out


@app.route("/pipeline", methods=["POST"])
def pipeline():
    try:
        body = request.get_json(force=True, silent=True)
        session, revision, inputs, events = validate_request_shape(body)
    except ConflictError as e:
        return jsonify({"error": e.code}), 409

    with _LOCK:
        existing = _SESSIONS.get(session)
        working = copy.deepcopy(existing) if existing else SessionState()

        accepted_ids = []
        ignored_ids = []

        try:
            apply_revision(working, revision, inputs)
            for ev in events:
                validate_event_shape(ev)
                result = process_event(working, ev)
                if result == "accepted":
                    accepted_ids.append(ev["eventId"])
                else:
                    ignored_ids.append(ev["eventId"])
        except ConflictError as e:
            return jsonify({"error": e.code}), 409

        # commit
        _SESSIONS[session] = working
        nodes_out = build_response(working)

    return jsonify({
        "revision": working.current_revision,
        "acceptedEventIds": accepted_ids,
        "ignoredEventIds": ignored_ids,
        "nodes": nodes_out,
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
