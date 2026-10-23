"""Finalize QALCO v3 on complete, ordered GSPO groups before advantages.

The ordinary ORM performs local COMET scoring first. This bridge exchanges only
scalar metadata, never audio, translations, reference text or model tensors.
World-rank order is deliberately identical to the audited trainer's reward
``gather`` followed by ``view(-1, num_generations)``. Repeated task IDs must NOT
merge separate prompt draws.

There are two unconditional world collectives per successful v3 scoring call:
metadata/error envelopes, then finalize/commit status. A local scoring or file
write failure therefore cannot leave healthy peers entering advantage gathers.
Broken process groups, killed workers and network failures remain torchrun/NCCL
failures; this bridge does not promise recovery from a missing process.
"""
from __future__ import annotations

import ast
import copy
from contextvars import ContextVar
from functools import wraps
import hashlib
import importlib
import inspect
import json
import math
import re

from .rewards import RewardError


PATCH_ID = "flex-global-gated-reward-v3"
_SCORING_ACTIVE = ContextVar("flex_complete_group_scoring", default=False)
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_ITEM_INPUTS = {"eligible", "quality", "latency", "units", "repetition", "reward"}
_INVALID_SAFE_FIELDS = {"group_brevity_excess", "weighted_brevity_penalty", "length_penalty",
                        "weighted_repetition_penalty", "anchor_pool_size", "group_length_anchor",
                        "group_length_slack", "anchor_quality_threshold"}


def is_group_scoring_active():
    """The v3 ORM must refuse provisional rewards outside this bridge."""
    return _SCORING_ACTIVE.get() is True


def _canonical_hash(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RewardError(f"Global reward {field} must be finite")
    return float(value)


def _positive_int(value, field):
    if type(value) is not int or value < 1:
        raise RewardError(f"Global reward {field} must be a positive integer")
    return value


def _error(exc):
    # Never transport arbitrary exception messages: a service may echo a whole
    # source/reference text, auth token, URL credentials, or candidate payload.
    return {"type": type(exc).__name__, "phase": "local_score_or_metadata"}


def _identity(sample, record, field):
    observed = getattr(sample, field, None)
    stored = record.get(field)
    for value in (observed, stored):
        if value is not None and not isinstance(value, str):
            raise RewardError(f"Global reward {field} must be a string or null")
    if observed and stored and observed != stored:
        raise RewardError(f"Global reward sample/record {field} mismatch")
    value = observed or stored or ""
    if field == "request_id" and not value:
        raise RewardError("Global reward requires a nonempty request_id for every candidate")
    return value


def _task_hash(sample, record):
    digest = record.get("task_hash")
    if not isinstance(digest, str) or not _HASH.fullmatch(digest):
        raise RewardError("Global reward requires the SHA256 full-task hash")
    extra = getattr(sample, "extra", None)
    if not isinstance(extra, dict) or "flex_task" not in extra:
        raise RewardError("Global reward sample is missing extra.flex_task")
    task = extra["flex_task"]
    if isinstance(task, str):
        task = json.loads(task)
    # Match OnlineQalcoReward's existing canonical JSON byte contract exactly
    # (default JSON separators, unlike the internal compact envelope hash).
    observed = hashlib.sha256(json.dumps(task, ensure_ascii=False, sort_keys=True,
                                        allow_nan=False).encode("utf-8")).hexdigest()
    if not isinstance(task, dict) or observed != digest:
        raise RewardError("Global reward sample/record full-task hash mismatch")
    if record.get("task_id") != task.get("id"):
        raise RewardError("Global reward sample/record task ID mismatch")
    return digest


def _local_metadata(samples, records):
    if not isinstance(records, list) or len(records) != len(samples):
        raise RewardError("Global reward local records must match samples exactly")
    result = []
    for index, (sample, record) in enumerate(zip(samples, records)):
        if not isinstance(record, dict) or not isinstance(record.get("components"), dict):
            raise RewardError("Global reward record/components must be objects")
        if record.get("candidate_index_in_local_batch") != index:
            raise RewardError("Global reward local record order mismatch")
        components = record["components"]
        eligible = components.get("eligible")
        if type(eligible) is not bool:
            raise RewardError("Global reward eligibility must be explicitly boolean")
        units = components.get("emitted_units")
        if units is None:
            units = components.get("diagnostics", {}).get("emitted_units")
        if type(units) is not int or units < 0:
            raise RewardError("Global reward emitted_units must be a nonnegative integer")
        quality, latency = components.get("quality"), components.get("latency_seconds")
        repetition = components.get("repetition_penalty", 0.0)
        if eligible:
            quality = _finite(quality, "quality")
            latency = _finite(latency, "latency")
            if not 0 <= quality <= 1 or latency < 0 or units == 0:
                raise RewardError("Global reward eligible candidate has invalid scalar data")
        elif quality is not None:
            quality = _finite(quality, "ineligible quality")
        if not eligible and latency is not None:
            latency = _finite(latency, "ineligible latency")
        repetition = _finite(repetition, "repetition")
        if not 0 <= repetition <= 1:
            raise RewardError("Global reward repetition penalty must be in [0, 1]")
        item = {"eligible": eligible, "quality": quality, "latency": latency,
                "units": units, "repetition": repetition,
                "reward": _finite(record.get("reward"), "provisional reward")}
        result.append({"task_hash": _task_hash(sample, record),
                       "prompt_id": _identity(sample, record, "prompt_id"),
                       "request_id": _identity(sample, record, "request_id"), "item": item})
    return result


def _runtime_contract(trainer):
    funcs = getattr(trainer, "reward_funcs", None)
    names = getattr(trainer, "reward_func_names", None)
    if not isinstance(funcs, (list, tuple)) or len(funcs) != 1:
        raise RewardError("QALCO v3 requires exactly one registered reward function")
    if list(names or ()) not in (["QalcoOnlineORM"], ["qalco_online"]):
        raise RewardError("QALCO v3 requires the sole qalco_online reward")
    delegate = getattr(funcs[0], "scorer", None)
    settings = getattr(delegate, "settings", None)
    if not isinstance(settings, dict) or settings.get("version") != 3 or settings.get("provider") != "qalco":
        raise RewardError("QALCO v3 global bridge received a different reward configuration")
    if not callable(getattr(delegate, "commit_group_rewards", None)):
        raise RewardError("QALCO v3 reward adapter lacks deferred commit_group_rewards")
    weights = getattr(trainer, "reward_weights", None)
    if hasattr(weights, "detach"):
        weights = weights.detach().cpu().tolist()
    elif hasattr(weights, "tolist"):
        weights = weights.tolist()
    if not isinstance(weights, (list, tuple)) or len(weights) != 1 or _finite(weights[0], "weight") != 1:
        raise RewardError("QALCO v3 requires reward_weights=[1]")
    if getattr(trainer, "dynamic_sample", None) is not False:
        raise RewardError("QALCO v3 global bridge requires dynamic_sample=false")
    if getattr(trainer, "use_gym_env", False):
        raise RewardError("QALCO v3 global bridge does not support appended gym rewards")
    if getattr(trainer, "scale_rewards", None) != "group":
        raise RewardError("QALCO v3 requires upstream group advantage normalization")
    models = getattr(trainer, "unwrapped_models", None)
    if models and not models[0].training:
        raise RewardError("QALCO v3 group bridge is source-audited for training only, not evaluation")
    return delegate, settings


def finalize_global_metadata(envelopes, settings, finalizer):
    """Pure CPU validation/finalization in the upstream world-rank-major order."""
    if not isinstance(envelopes, list) or not envelopes:
        raise RewardError("Global reward collective returned no envelopes")
    if any(not isinstance(envelope, dict) for envelope in envelopes):
        raise RewardError("Global reward collective returned malformed envelopes")
    failures = [str(index) for index, envelope in enumerate(envelopes) if envelope.get("error") is not None]
    if failures:
        raise RewardError("Global QALCO local scoring/metadata failed on ranks " + ",".join(failures))
    contract = envelopes[0].get("contract")
    if not isinstance(contract, dict):
        raise RewardError("Global reward contract is missing")
    if any(envelope.get("contract") != contract for envelope in envelopes):
        raise RewardError("Global QALCO configuration/batch contract differs across ranks")
    if contract.get("settings_hash") != _canonical_hash(settings):
        raise RewardError("Global QALCO local settings hash does not match envelopes")
    group_size = _positive_int(contract.get("group_size"), "group_size")
    expected = _positive_int(contract.get("generation_batch_size"), "generation_batch_size")
    if expected % group_size:
        raise RewardError("Global reward generation batch is not divisible by group size")
    counts = []
    flat = []
    for rank, envelope in enumerate(envelopes):
        if envelope.get("rank") != rank or not isinstance(envelope.get("items"), list):
            raise RewardError("Global reward envelopes are not in world-rank order")
        counts.append(len(envelope["items"]))
        flat.extend(envelope["items"])
    if not counts[0] or len(set(counts)) != 1 or sum(counts) != expected:
        raise RewardError("Global reward rank counts must be equal, nonempty, and cover the planned generation batch")
    request_ids = []
    for entry in flat:
        if not isinstance(entry, dict) or not isinstance(entry.get("item"), dict):
            raise RewardError("Global reward candidate metadata is malformed")
        request_id = entry.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise RewardError("Global reward candidate request_id is missing")
        request_ids.append(request_id)
    if len(set(request_ids)) != len(request_ids):
        raise RewardError("Global reward candidate request IDs must be unique")
    finalized = []
    for start in range(0, len(flat), group_size):
        block = flat[start:start + group_size]
        task_hashes = {entry.get("task_hash") for entry in block}
        if len(task_hashes) != 1 or not all(isinstance(value, str) and _HASH.fullmatch(value) for value in task_hashes):
            raise RewardError("Contiguous GSPO group contains different full-task hashes")
        prompt_ids = [entry.get("prompt_id") for entry in block]
        if any(not isinstance(value, str) for value in prompt_ids):
            raise RewardError("Global reward prompt IDs must be strings")
        if any(prompt_ids) and (not all(prompt_ids) or len(set(prompt_ids)) != 1):
            raise RewardError("Contiguous GSPO group contains missing/mismatched prompt instance IDs")
        # Separate same-task draws remain separate blocks even when prompt IDs
        # are absent. The group identity includes this ordered candidate set.
        group_id = _canonical_hash({"offset": start, "task_hash": block[0]["task_hash"],
                                    "prompt_id": prompt_ids[0],
                                    "request_ids": [entry["request_id"] for entry in block]})
        values = finalizer([copy.deepcopy(entry["item"]) for entry in block], settings)
        if not isinstance(values, list) or len(values) != group_size:
            raise RewardError("Global group finalizer changed the candidate count")
        for index, (entry, value) in enumerate(zip(block, values)):
            if not isinstance(value, dict):
                raise RewardError("Global group finalizer returned a non-object")
            reward = _finite(value.get("reward"), "final reward")
            if not entry["item"]["eligible"] and reward != entry["item"]["reward"]:
                raise RewardError("Global group finalizer changed an ineligible candidate reward")
            finalized.append({"reward": reward,
                              "fields": {key: val for key, val in value.items() if key not in _ITEM_INPUTS},
                              "group": {"group_finalized": True, "global_group_id": group_id,
                                        "global_group_index": start // group_size,
                                        "group_candidate_index": index, "group_size": group_size,
                                        "group_prompt_id": prompt_ids[0],
                                        "group_request_id": entry["request_id"]}})
    return finalized, counts


def run_group_reward_scoring(trainer, samples, original_score, distributed, *, finalizer=None):
    """Production scoring boundary, dependency-injected for CPU collectives tests."""
    if not distributed.is_initialized():
        raise RewardError("QALCO v3 requires an initialized distributed process group")
    rank, size = distributed.get_rank(), distributed.get_world_size()
    original_tensor = delegate = settings = None
    local_records = []
    score_exception = None
    envelope = {"rank": rank, "error": None, "contract": None, "items": []}
    try:
        delegate, settings = _runtime_contract(trainer)
        contract = {"settings_hash": _canonical_hash(settings),
                    "group_size": _positive_int(trainer.num_generations, "num_generations"),
                    "generation_batch_size": _positive_int(trainer.generation_batch_size, "generation_batch_size")}
        if getattr(trainer, "process_index", None) != rank or getattr(trainer, "world_size", None) != size:
            raise RewardError("QALCO v3 world rank does not match upstream reward gather ownership")
        # Avoid accidentally reusing a previous callback's records after an ORM
        # failure or an upstream change which stops invoking the registered ORM.
        delegate.last_records = []
        token = _SCORING_ACTIVE.set(True)
        try:
            original_tensor = original_score(trainer, samples)
        finally:
            _SCORING_ACTIVE.reset(token)
        if tuple(original_tensor.shape) != (len(samples), 1):
            raise RewardError("QALCO v3 upstream score tensor must have shape [local_candidates, 1]")
        local_records = delegate.last_records
        metadata = _local_metadata(samples, local_records)
        envelope.update(contract=contract, items=metadata)
    except BaseException as exc:
        score_exception = exc
        envelope["error"] = _error(exc)
    gathered = [None] * size
    # If the collective itself fails, do NOT start another collective on the
    # broken group. torchrun/NCCL owns process failure propagation in that case.
    distributed.all_gather_object(gathered, envelope)
    result = None
    status = {"rank": rank, "error": None}
    local_exception = None
    try:
        if finalizer is None:
            from .gated_reward import finalize_group
            finalizer = finalize_group
        # Even the locally failing rank follows the same finalization-status
        # collective. Settings may be absent after a local config failure.
        if settings is None:
            raise RewardError("QALCO v3 local configuration/scoring failed before metadata finalization")
        finalized, counts = finalize_global_metadata(gathered, settings, finalizer)
        start = sum(counts[:rank])
        selected = finalized[start:start + counts[rank]]
        committed = []
        for record, value in zip(local_records, selected):
            record = copy.deepcopy(record)
            record["reward"] = value["reward"]
            record["prompt_id"] = value["group"]["group_prompt_id"]
            record["request_id"] = value["group"]["group_request_id"]
            if record["components"]["eligible"]:
                record["components"].update(value["fields"])
            else:
                record["components"].update({key: val for key, val in value["fields"].items()
                                             if key in _INVALID_SAFE_FIELDS})
            # Preserve invalid policy errors/tier and None quality. They cannot
            # acquire fabricated semantic/latency components during reshaping.
            record["components"].update(value["group"])
            committed.append(record)
        result = original_tensor.new_tensor([[record["reward"]] for record in committed])
        delegate.commit_group_rewards(committed, getattr(trainer, "state", None))
    except BaseException as exc:
        local_exception = score_exception or exc
        status["error"] = {"type": type(exc).__name__, "phase": "finalize_or_commit"}
    statuses = [None] * size
    distributed.all_gather_object(statuses, status)
    if (any(not isinstance(value, dict) or value.get("rank") != index for index, value in enumerate(statuses))
            or any(value.get("error") is not None for value in statuses)):
        ranks = [str(index) for index, value in enumerate(statuses)
                 if not isinstance(value, dict) or value.get("error") is not None]
        raise RewardError("Global QALCO finalize/commit failed; no advantages were computed; ranks="
                          + ",".join(ranks)) from local_exception
    return result


def validate_group_reward_source(source):
    """Reject a changed upstream scorer/gather boundary before installing."""
    tree = ast.parse(source)
    cls = next((node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == "MegatronGRPOTrainer"), None)
    if cls is None:
        raise RewardError("SWIFT has no MegatronGRPOTrainer for QALCO v3")
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    required = {"_score_completions", "_generate_and_score_completions", "_compute_advantages"}
    if not required.issubset(methods):
        raise RewardError("SWIFT global group reward scoring/advantage boundary changed")
    scorer = methods["_score_completions"]
    if [arg.arg for arg in scorer.args.args] != ["self", "samples"]:
        raise RewardError("SWIFT _score_completions signature changed")
    calls = [node for node in ast.walk(scorer) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "score_completions"]
    if len(calls) != 1 or not {"reward_funcs", "trainer_state", "device"}.issubset(
            {keyword.arg for keyword in calls[0].keywords}):
        raise RewardError("SWIFT score_completions delegation contract changed")
    if len(calls[0].args) != 1 or not isinstance(calls[0].args[0], ast.Name) or calls[0].args[0].id != "samples":
        raise RewardError("SWIFT score_completions no longer receives unchanged ordered samples")
    advantages = methods["_compute_advantages"]
    gathers = [node for node in ast.walk(advantages) if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Name) and node.func.id == "gather"
               and len(node.args) == 1 and isinstance(node.args[0], ast.Name)
               and node.args[0].id == "rewards_per_func"]
    if len(gathers) != 1 or gathers[0].keywords:
        raise RewardError("SWIFT rewards are no longer gathered in audited default world order")
    text = ast.unparse(advantages)
    if ("self.process_index * len(samples)" not in text
            or "num_generations=num_generations" not in text
            or "compute_advantages(" not in text):
        raise RewardError("SWIFT global GSPO grouping/slice ownership changed")
    generate = methods["_generate_and_score_completions"]
    scored = [node for node in ast.walk(generate) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute) and node.func.attr == "_score_completions"]
    normalized = [node for node in ast.walk(generate) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute) and node.func.attr == "_compute_advantages"]
    if len(scored) != 1 or len(normalized) != 1 or scored[0].lineno >= normalized[0].lineno:
        raise RewardError("SWIFT no longer scores once before global advantage normalization")
    if ([ast.unparse(arg) for arg in scored[0].args] != ["samples"]
            or [ast.unparse(arg) for arg in normalized[0].args] != ["samples", "rewards_per_func"]):
        raise RewardError("SWIFT changed ordered sample/reward ownership between scoring and advantages")
    return {"patch_id": PATCH_ID, "call_site_supported": True,
            "group_order": "world-rank-major contiguous num_generations blocks",
            "collectives_per_score": 2, "transport": "scalar metadata only; no text/audio/tensors"}


def install_group_reward_patch(module=None, distributed=None, *, source=None, finalizer=None):
    """Install only in v3 processes after trainer imports finish.

    The scheduler's config guard must not install this bridge in independent
    v1/v2 runs. Once installed, switching a FLEX trainer to v2 in this process
    is rejected synchronously, not allowed to skip peer collectives. Other
    scheduler types remain untouched.
    """
    if module is None:
        module = importlib.import_module("swift.megatron.trainers.grpo_trainer")
    if getattr(module, "_flex_group_reward_patch", None) == PATCH_ID:
        return
    validate_group_reward_source(inspect.getsource(module) if source is None else source)
    if distributed is None:
        distributed = importlib.import_module("torch.distributed")
    original = module.MegatronGRPOTrainer._score_completions
    from .online import FlexStreamHooks

    @wraps(original)
    def finalized_score(trainer, samples):
        # Version/reward/weight validation belongs INSIDE the synchronized
        # boundary. A rank-local malformed delegate must not skip its peers.
        applies = isinstance(getattr(trainer, "multi_turn_scheduler", None), FlexStreamHooks)
        if not applies:
            return original(trainer, samples)
        return run_group_reward_scoring(trainer, samples, original, distributed, finalizer=finalizer)

    module.MegatronGRPOTrainer._score_completions = finalized_score
    module._flex_group_reward_patch = PATCH_ID
