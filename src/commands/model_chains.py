"""View and edit the per-category mixture model chains.

The chains live as user config in `<omh-home>/routing/model-chains.json`
(`mixture_chain_overrides/v1`): the document replaces only the categories it
names and the shipped defaults keep serving the rest. This command is the
supported editing surface over that file:

* `omh model-chains` (bare, on a terminal) — the arrow-key picker: one row
  per category, left/right steps the head model, -/+ its effort, Enter saves.
  `omh model` is the same command under a shorter name. Off a terminal the
  bare form prints `show`.
* `omh model-chains show` — the current effective chain per category, with
  its origin (shipped default vs override) and the document status.
* `omh model-chains set <category> "model[:effort], model[:effort]"` — write
  one category's replacement chain; `--clear` returns it to the default.
* `omh model-chains interview` — walk every category with numbered choices
  (keep / shipped default / Ultrafast tier / custom entry) on a terminal.
* `omh model-chains provider set <id> <kind>` / `clear <id>` — record what
  one of this machine's providers serves, which is what reorders the chains
  above. The scriptable counterpart of the `omh setup` provider question,
  which a `--yes`, `--json`, or non-TTY install never reaches.

Editing the JSON file directly stays equally supported; every path converges
on the same validated document.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from ..catalogs.model_chain_table import CHAIN_SURFACE_PURPOSES, MODEL_DISPLAY_LABELS
from ..coding.data_handling_policy import (
    SENSITIVE_WORK_CHAIN_CLAIM_BOUNDARY,
    data_handling_filtered_chain,
)
from ..coding.model_routing import NON_GENERATIVE_MODEL_CLASS, REASONING_EFFORT_LADDER, model_class
from ..local_store import atomic_write_text
from ..plugin_bundle.omh.hermes_delegation import (
    APPROX_PRICE_PER_MTOK,
    HERMES_MIXTURE_CATEGORY_CHAINS,
    PROVIDER_FAMILY_VOCABULARY,
    PROVIDER_KIND_VOCABULARY,
    alias_is_served,
    chains_with_overrides,
    entitlement_shaped_chain,
    load_mixture_chain_overrides,
    load_model_provider_routes,
    effective_provider_entitlements,
    mixture_chain_overrides_path,
    model_provider_routes_path,
    is_provider_id_token,
    parse_mixture_chain_overrides,
    provider_entitlements_path,
    provider_family_for,
    routes_to_unknown_providers,
    split_unknown_routes,
    unknown_route_labels,
)
from ..plugin_bundle.omh.model_chain_picker import (
    chain_text,
    compose_override_document,
    picker_rows,
    read_override_document,
)
from ..install.config_adapter import display_skin_selection, read_config
from ..skin_pack import effective_theme, skin_colors
from ..system.stdin_probe import stdin_is_tty
from .common import _paths
from .model_chain_picker import pick_chains_interactively

# From `quickstart`, not `setup`, for the reason `theme` gives: importing the
# parser module back here would close an import cycle.
from .quickstart import _use_color
from .theme_picker import picker_available

_ENTRY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
# A model id may carry a colon (`custom:cli-proxy`, an Ollama tag such as
# `qwen3:8b`), the same grammar the override document accepts.
_MODEL_ENTRY_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._/-]|:(?=[A-Za-z0-9])){0,127}$")
_CHAIN_EFFORTS = frozenset((*REASONING_EFFORT_LADDER, "none", "auto"))

# The Ultrafast-tier interview option: a chain member is offered its
# `<model>-ultrafast` speed variant when that token is a known model (shipped
# chains or the price table), so no variant is invented. Editorial offer, not
# a capability claim; the option is offered only when it changes the chain.
_KNOWN_MODEL_TOKENS = frozenset(APPROX_PRICE_PER_MTOK) | {
    model for chain in HERMES_MIXTURE_CATEGORY_CHAINS.values() for model, _ in chain
}
_ULTRAFAST_SWAPS = {
    model: f"{model}-ultrafast"
    for model in _KNOWN_MODEL_TOKENS
    if f"{model}-ultrafast" in _KNOWN_MODEL_TOKENS
}


def _parsechain_text(text: str) -> tuple[tuple[str, str], ...]:
    """Parse `model[:effort], model[:effort]` into chain entries or raise ValueError."""
    entries: list[tuple[str, str]] = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        # The tail after the last colon is an effort only when it names one,
        # so `qwen3:8b` stays a model id and `custom:proxy:low` splits once.
        model, sep, effort = piece.rpartition(":")
        if not sep or effort.strip().casefold() not in _CHAIN_EFFORTS:
            model, effort = piece, ""
        model = model.strip()
        effort = effort.strip()
        if not _MODEL_ENTRY_RE.fullmatch(model):
            raise ValueError(f"{model!r} is not a plain model identifier")
        if effort and not _ENTRY_RE.match(effort):
            raise ValueError(f"{effort!r} is not a plain reasoning-effort token")
        entries.append((model, effort))
    if not entries:
        raise ValueError("a chain needs at least one `model[:effort]` entry")
    return tuple(entries)


def _write_document(omh_home, document: dict[str, object]) -> str:
    _, status = parse_mixture_chain_overrides(document)
    if status.startswith("invalid"):
        raise ValueError(status)
    path = mixture_chain_overrides_path(omh_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(document, indent=2, sort_keys=True) + "\n")
    return str(path)


def _state(omh_home, hermes_home=None) -> dict[str, object]:
    overrides, status = load_mixture_chain_overrides(omh_home)
    entitlements, entitlement_status, providers = effective_provider_entitlements(omh_home, hermes_home)
    routes, routes_status = load_model_provider_routes(omh_home)
    chains = chains_with_overrides(overrides)
    categories = []
    for name, chain in chains.items():
        shaped = chain
        if entitlements is not None:
            shaped = entitlement_shaped_chain(chain, entitlements, routes)
        categories.append(
            {
                "category": name,
                "chain": [
                    {
                        "model": model,
                        "reasoning_effort": effort,
                        "served": (
                            alias_is_served(model, entitlements, routes)
                            if entitlements is not None
                            else True
                        ),
                    }
                    for model, effort in shaped
                ],
                "chain_text": chain_text(shaped),
                "origin": "override" if name in overrides else "default",
                "entitlement_shaped": shaped != chain,
            }
        )
    return {
        "schema_version": "model_chain_state/v1",
        "path": str(mixture_chain_overrides_path(omh_home)),
        "document_status": status,
        "entitlements_path": str(provider_entitlements_path(omh_home)),
        "entitlements_status": entitlement_status,
        "providers": [dict(row) for row in providers],
        # Providers whose kind names no model family: `gateway`, `unknown`,
        # or an id nothing on this machine records. The serving rule counts
        # every alias as served for such a provider, so on its own it can
        # only ever leave a chain in the order it already had. Resolved
        # through `provider_family_for` -- the one id -> family lookup this
        # surface has -- so the printed line cannot drift from a second copy
        # of the rule.
        "unplaced_providers": [
            row["id"]
            for row in providers
            if provider_family_for(str(row["id"]), entitlements) not in PROVIDER_FAMILY_VOCABULARY
        ],
        "routes_path": str(model_provider_routes_path(omh_home)),
        "routes_status": routes_status,
        # Routes to a provider neither recorded nor linked, each with its
        # effect: `chain` rows name an alias a chain names (demoted behind
        # every served entry); `dispatch` rows reach no chain and only
        # resolve a pinned dispatch, unchecked.
        "unserved_routes": [dict(row) for row in routes_to_unknown_providers(routes, entitlements, chains)],
        "categories": categories,
    }


def _print_state(state: dict[str, object]) -> None:
    print("Current model chains (category -> effective order):")
    for row in state["categories"]:
        marker = " (override)" if row["origin"] == "override" else ""
        if row.get("entitlement_shaped"):
            marker += " (reordered by this machine's providers)"
        print(f"  {row['category']}: {row['chain_text']}{marker}")
    print(f"Overrides file: {state['path']} [{state['document_status']}]")
    print(f"Provider entitlements: {state['entitlements_path']} [{state['entitlements_status']}]")
    if str(state["entitlements_status"]).startswith("invalid:"):
        # An invalid record yields no document: its kinds are dropped and a
        # linked row it excluded counts again. Said here, not left to the
        # bracketed status.
        print("  providers.json is ignored: its recorded kinds are dropped and any providers it excluded count again")
    linked = [row for row in state.get("providers", []) if row["source"] != "recorded"]
    if linked:
        print("Linked Hermes providers: " + ", ".join(f"{row['id']} ({row['source']})" for row in linked))
        unplaced = set(state.get("unplaced_providers", []))
        reordered = any(row.get("entitlement_shaped") for row in state["categories"])
        # Said only when it is the whole story: every provider this machine
        # holds is one whose kind names no family, and no category came out
        # reordered. A named multi-vendor family (`openrouter`, `opencode`)
        # leaves the chains unshaped too, but correctly -- it does serve
        # every family, and no answer would change that -- so a machine
        # holding one is not told to go and fix something. A route in
        # model-providers.json can shape a chain past a gateway kind, and
        # then the reordering is not doing nothing, so the line stays off.
        #
        # The command is the only way out named, because for this reader it
        # is the only one there is. Measured on a machine whose one provider
        # is a `providers.custom` block with an unknown host: the interview
        # ticks the row at its detected kind and writes `gateway`, and typing
        # the same id at the add prompt is refused as already recorded, so
        # neither path a person would take records a family. (Unticking it
        # and then re-entering it by name does work, but naming that here
        # would be teaching a workaround.) A `--yes`, `--json`, or non-TTY
        # setup asks nothing at all, and that install path overlaps heavily
        # with the machines that land here. `tests/test_custom_provider_hint`
        # pins the measurement: if the interview ever gains a way to record a
        # family for a detected row, revisit this sentence.
        if unplaced >= {str(row["id"]) for row in state.get("providers", [])} and not reordered:
            print("  none of these names a model family, so no chain is reordered. Record what one serves")
            print("  with `omh model-chains provider set <id> <kind>`.")
    else:
        print("Linked Hermes providers: none found (a `hermes auth` login, a config provider, or a key name counts)")
    routes_status = str(state.get("routes_status", ""))
    if routes_status.startswith("invalid:"):
        # An absent document is the common case and says nothing; an
        # invalid one silently drops every route, so it is named.
        print(f"Provider routes: {state.get('routes_path', '')} [{routes_status}] (ignored: every alias dispatches unchanged)")
    demoted, dispatch_only = split_unknown_routes(state.get("unserved_routes", []))
    if demoted:
        print(
            "Chain entries routed to a provider neither recorded nor linked: "
            + unknown_route_labels(demoted)
            + " (each sorts behind the served entries of every chain naming it)"
        )
    if dispatch_only:
        print(
            "Dispatch-only routes to a provider neither recorded nor linked: "
            + unknown_route_labels(dispatch_only)
            + " (no chain names these; a dispatch pinning one asks Hermes for a provider it is not linked to)"
        )
    print("Edit a category with `omh model-chains set <category> \"model[:effort], ...\"`,")
    print("walk all of them with `omh model-chains interview`, or edit the JSON directly.")


def _sensitive_state(state: dict[str, object]) -> dict[str, object]:
    """Re-shape the chain state for declared-sensitive work.

    Sensitivity arrives as the caller's `--sensitive` flag and nothing else;
    the gate never reads a message, a path, or a category name to decide it.
    Every category keeps its row even when nothing survives, because a
    category that vanished would read as one with no chain configured.
    """
    categories = state["categories"]
    assert isinstance(categories, list)
    shaped: list[dict[str, object]] = []
    for row in categories:
        chain = row["chain"]
        assert isinstance(chain, list)
        report = data_handling_filtered_chain(
            [str(entry["model"]) for entry in chain],
            work_is_sensitive=True,
        )
        permitted = set(report["chain"])
        kept = [entry for entry in chain if str(entry["model"]) in permitted]
        shaped.append(
            {
                **row,
                "chain": kept,
                "chain_text": chain_text(
                    tuple((str(entry["model"]), str(entry["reasoning_effort"])) for entry in kept)
                ),
                "data_handling": report,
            }
        )
    empty = [str(row["category"]) for row in shaped if not row["chain"]]
    return {
        **state,
        "categories": shaped,
        "work_is_sensitive": True,
        "categories_with_no_permitted_model": empty,
        "claim_boundary": SENSITIVE_WORK_CHAIN_CLAIM_BOUNDARY,
    }


def _print_sensitive_state(state: dict[str, object]) -> None:
    print("Model chains for work declared sensitive (category -> permitted order):")
    categories = state["categories"]
    assert isinstance(categories, list)
    for row in categories:
        report = row["data_handling"]
        assert isinstance(report, dict)
        summary = report["summary"]
        assert isinstance(summary, dict)
        text = row["chain_text"] or "(no model in this chain has a permitting documented policy)"
        print(f"  {row['category']}: {text}")
        for excluded in report["excluded"]:
            print(f"    excluded {excluded['model']}: {excluded['verdict']} ({excluded['reason']})")
    empty = state["categories_with_no_permitted_model"]
    assert isinstance(empty, list)
    if empty:
        print("Categories with nothing left to route sensitive work to: " + ", ".join(empty))
    print(str(state["claim_boundary"]))


def _sensitive_chain_exit_code(state: object) -> int:
    """0 only when every category still has a model declared-sensitive work may use.

    1 when at least one category was emptied. A wrapper reading only the
    status must not start declared-sensitive work against a chain the gate
    emptied, and an empty chain is the concrete finding: there is no
    recoverable lane in which "nothing is permitted" is a normal result. A
    plain `show` never sets the key, so it keeps returning 0.

    Generic failure signals -- a refused or interrupted run, or a unit
    carrying a failure kind -- are never success either, so this mapper
    cannot be passed by ignoring them.
    """
    summary = state if isinstance(state, dict) else {}
    if summary.get("refused") or summary.get("interrupted"):
        return 1
    units = summary.get("units")
    if isinstance(units, list) and any(isinstance(unit, dict) and unit.get("failure_kind") for unit in units):
        return 1
    return 1 if summary.get("categories_with_no_permitted_model") else 0


def cmd_model_chains_show(args: argparse.Namespace) -> int:
    paths = _paths(args)
    state = _state(paths.omh_home, paths.hermes_home)
    sensitive = bool(getattr(args, "sensitive", False))
    if sensitive:
        state = _sensitive_state(state)
    if getattr(args, "json", False):
        print(json.dumps(state, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    elif sensitive:
        _print_sensitive_state(state)
    else:
        _print_state(state)
    return _sensitive_chain_exit_code(state)


def cmd_model_chains_set(args: argparse.Namespace) -> int:
    paths = _paths(args)
    omh_home = paths.omh_home
    category = str(args.category)
    if category not in HERMES_MIXTURE_CATEGORY_CHAINS:
        print(
            f"omh: unknown category {category!r}; choose one of "
            + ", ".join(HERMES_MIXTURE_CATEGORY_CHAINS),
            file=sys.stderr,
        )
        return 2
    document = read_override_document(omh_home)
    categories = document["categories"]
    assert isinstance(categories, dict)
    if args.clear:
        categories.pop(category, None)
    else:
        if not args.chain:
            print("omh: set needs a chain (`model[:effort], ...`) or --clear", file=sys.stderr)
            return 2
        try:
            entries = _parsechain_text(" ".join(args.chain))
        except ValueError as exc:
            print(f"omh: {exc}", file=sys.stderr)
            return 2
        # Shape is not enough here. A chain entry is a model that will be
        # handed a coding unit, and a non-generative model answers typed
        # questions instead of writing code, so it can never be one. Refused
        # before the document is written, like every other refusal above.
        refused = next(
            (model for model, _effort in entries if model_class(model) == NON_GENERATIVE_MODEL_CLASS),
            "",
        )
        if refused:
            print(
                f"omh: {refused!r} is a {NON_GENERATIVE_MODEL_CLASS} model "
                "(it answers typed questions and cannot write code) and cannot join a model chain",
                file=sys.stderr,
            )
            return 2
        categories[category] = [
            {"model": model, "reasoning_effort": effort} for model, effort in entries
        ]
    try:
        path = _write_document(omh_home, document)
    except ValueError as exc:
        print(f"omh: refused to write an invalid document: {exc}", file=sys.stderr)
        return 2
    state = _state(omh_home, paths.hermes_home)
    if getattr(args, "json", False):
        print(json.dumps(state, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    else:
        row = next(r for r in state["categories"] if r["category"] == category)
        print(f"{category}: {row['chain_text']} ({row['origin']})")
        print(f"Written to {path}. New delegations use this order; running children keep theirs.")
    return 0


def _active_palette(paths) -> dict[str, str]:
    """The colours of the theme the OMH TUI opens with, so the picker matches it.

    `effective_theme`'s order: the theme saved under the OMH home, else an
    OMH palette `display.skin` names, else the default.
    """
    theme, _source = effective_theme(paths.omh_home, display_skin_selection(read_config(paths.hermes_config_path)))
    return skin_colors(theme.skin_name)


def cmd_model_chains_pick(args: argparse.Namespace) -> int:
    """Bare `omh model-chains` picks interactively; anything else prints `show`.

    The split is the theme command's: `show` is the scriptable surface and
    stays byte-predictable, the bare form is what a person types to SEE the
    chains and move them. `--json` and any non-terminal end degrade the bare
    form to the same listing.
    """
    if getattr(args, "json", False) or not picker_available():
        return cmd_model_chains_show(args)
    paths = _paths(args)
    omh_home = paths.omh_home
    payload = picker_rows(
        omh_home, hermes_home=paths.hermes_home, labels=MODEL_DISPLAY_LABELS, purposes=CHAIN_SURFACE_PURPOSES
    )
    changes = pick_chains_interactively(payload, use_color=_use_color(), palette=_active_palette(paths))
    if changes is None:
        print(f"Cancelled; {payload['path']} was not changed.")
        return 0
    if not changes:
        print("No changes.")
        return 0
    try:
        document = compose_override_document(read_override_document(omh_home), changes)
        path = _write_document(omh_home, document)
    except ValueError as exc:
        print(f"omh: refused to write an invalid document: {exc}", file=sys.stderr)
        return 2
    print(f"Saved {len(changes)} categor{'y' if len(changes) == 1 else 'ies'} to {path}.")
    _print_state(_state(omh_home, paths.hermes_home))
    return 0


def _ultrafast_variant(
    chain: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...] | None:
    swapped = tuple((_ULTRAFAST_SWAPS.get(model, model), effort) for model, effort in chain)
    return swapped if swapped != chain else None


def _stdin_is_tty() -> bool:
    return stdin_is_tty()


CATEGORY_METADATA = {
    "ultrabrain": ("Deepest reasoning & multi-step problems", "xhigh", "reasoning"),
    "architect": ("Architecture, system design & DAG plan gates", "xhigh", "reasoning"),
    "deep-work": ("Long-running frontier tasks", "max", "reasoning"),
    "deep": ("Strong default tier for complex requests", "high", "reasoning"),
    "visual-engineering": ("Frontend UI/UX and visual polish", "high", "general"),
    "artistry": ("Unconventional creative synthesis", "high", "general"),
    "capable": ("Strong general coding & problem solving", "medium", "general"),
    "unspecified-high": ("Default working tier for untyped tasks", "medium", "general"),
    "writing": ("Documentation, specs & written explanations", "medium", "general"),
    "unspecified-low": ("Cheaper fallback for low-complexity steps", "low", "fast"),
    "quick": ("Short, low-latency utility steps", "low", "fast"),
    "simple-work": ("Everyday file edits & small chores", "low", "fast"),
}


def _get_hermes_bin() -> str:
    import os
    import shutil
    from pathlib import Path
    w = shutil.which("hermes")
    if w:
        return w
    for p in [
        Path.home() / ".local/bin/hermes",
        Path.home() / ".hermes/hermes-agent/.hermes/bin/hermes",
    ]:
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return "hermes"


def _discover_hermes_inventory() -> dict[str, object]:
    import os
    import subprocess
    from pathlib import Path
    h_bin = _get_hermes_bin()

    def query(key: str) -> object:
        try:
            r = subprocess.run([h_bin, "config", "get", key, "--json"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                return json.loads(r.stdout)
        except Exception:
            pass
        return None

    model_cfg = query("model") or {}
    default_model = model_cfg.get("default", "") if isinstance(model_cfg, dict) else str(model_cfg)
    custom_providers = query("custom_providers") or []
    aliases = query("model_aliases") or {}

    profile = "default"
    try:
        r_p = subprocess.run([h_bin, "config", "path"], capture_output=True, text=True, timeout=5)
        path_str = r_p.stdout.strip()
        if "profiles" in path_str:
            parts = path_str.split("/")
            if "profiles" in parts:
                idx = parts.index("profiles")
                if idx + 1 < len(parts):
                    profile = parts[idx + 1]
    except Exception:
        pass

    featured: list[dict[str, str]] = []
    seen: set[str] = set()

    # 1. Active default model
    if default_model:
        seen.add(default_model)
        featured.append({
            "id": default_model,
            "label": f"{default_model} (Hermes Default)",
            "provider": model_cfg.get("provider", "custom") if isinstance(model_cfg, dict) else "custom",
            "tier": "default",
        })

    # 2. Model aliases
    if isinstance(aliases, dict):
        for al, info in aliases.items():
            if al not in seen:
                seen.add(al)
                target = info.get("model", "") if isinstance(info, dict) else ""
                al_lower = al.lower()
                target_lower = target.lower() if isinstance(target, str) else ""
                clean_target = target_lower.replace("gemini", "")
                if "flash" in al_lower or "flash" in clean_target or "mini" in al_lower or "mini" in clean_target or "mimo" in al_lower:
                    al_tier = "fast"
                elif any(w in al_lower or w in target_lower for w in ["opus", "sonnet", "codex", "pro", "27b"]):
                    al_tier = "reasoning"
                else:
                    al_tier = "general"
                featured.append({
                    "id": al,
                    "label": f"{al} -> {target} (Alias)",
                    "provider": "alias",
                    "tier": al_tier,
                })

    # 3. Custom providers (excluding rootsys)
    all_models: list[dict[str, str]] = list(featured)
    if isinstance(custom_providers, list):
        for cp in custom_providers:
            cp_name = cp.get("name", "Custom")
            if "rootsys" in cp_name.lower() or "rootsys" in cp.get("base_url", "").lower():
                continue
            cp_models = cp.get("models", {})
            if isinstance(cp_models, dict):
                is_small_provider = len(cp_models) <= 15
                for m_id in cp_models.keys():
                    if m_id not in seen:
                        seen.add(m_id)
                        lower = m_id.lower()
                        clean_lower = lower.replace("gemini", "")
                        if any(w in clean_lower for w in ["flash", "mini", "haiku", "luna"]):
                            tier = "fast"
                        elif any(w in lower for w in ["opus", "sonnet", "codex", "pro", "27b", "astra"]):
                            tier = "reasoning"
                        elif "deep" in lower:
                            tier = "reasoning"
                        else:
                            tier = "general"

                        item = {
                            "id": m_id,
                            "label": f"{m_id} [{cp_name}]",
                            "provider": cp_name,
                            "tier": tier,
                        }
                        all_models.append(item)
                        if ":batch" not in m_id and (is_small_provider or any(k in lower for k in [
                            "claude-sonnet", "gpt-5.3-codex", "gemini-3.1-pro", "gemini-3.8-flash",
                            "deepseek-v4-pro", "glm-5.3", "qwen3.8-27b", "claude-opus"
                        ])):
                            featured.append(item)

    return {
        "profile": profile,
        "default_model": default_model,
        "featured": featured,
        "all": all_models,
    }


def _resolve_model_token(token: str, inventory: dict[str, object], default_effort: str) -> tuple[str, str] | None:
    token = token.strip()
    if not token:
        return None
    model_part, sep, effort_part = token.rpartition(":")
    if not sep or effort_part.lower() not in ("low", "medium", "high", "xhigh", "max"):
        model_part = token
        effort_part = default_effort

    if model_part.isdigit():
        idx = int(model_part)
        featured = inventory.get("featured", [])
        all_models = inventory.get("all", [])
        if isinstance(featured, list) and 1 <= idx <= len(featured):
            return featured[idx - 1]["id"], effort_part
        elif isinstance(all_models, list) and 1 <= idx <= len(all_models):
            return all_models[idx - 1]["id"], effort_part

    return model_part, effort_part


def _print_preview_table(categories_dict: dict[str, list[dict[str, str]]], use_color: bool = True) -> None:
    bold = "\033[1m" if use_color else ""
    cyan = "\033[36m" if use_color else ""
    dim = "\033[2m" if use_color else ""
    reset = "\033[0m" if use_color else ""

    print()
    print(f"{bold}Configured Model Chains Preview:{reset}")
    print(f"  {bold}{'Category':<20} {'Effort':<8} {'Configured Chain'}{reset}")
    print(f"  {dim}{'-'*70}{reset}")
    for cat in HERMES_MIXTURE_CATEGORY_CHAINS.keys():
        entries = categories_dict.get(cat, [])
        if not entries:
            continue
        effort = entries[0].get("reasoning_effort", "")
        chain_desc = ", ".join(e["model"] for e in entries)
        print(f"  {cyan}{cat:<20}{reset} {effort:<8} {chain_desc}")
    print()


def _legacy_category_walk(omh_home, overrides, document, categories, hermes_home) -> int:
    print("Model chain interview — Enter keeps the current order.")
    changed = 0
    for name, default_chain in HERMES_MIXTURE_CATEGORY_CHAINS.items():
        current = overrides.get(name, default_chain)
        options: list[tuple[str, tuple[tuple[str, str], ...] | None]] = [
            (f"keep current: {chain_text(current)}", current),
        ]
        if current != default_chain:
            options.append((f"shipped default: {chain_text(default_chain)}", default_chain))
        ultrafast = _ultrafast_variant(current)
        if ultrafast is not None:
            options.append((f"Ultrafast tier: {chain_text(ultrafast)}", ultrafast))
        options.append(("custom entry (`model[:effort], ...`)", None))
        print(f"\n[{name}]")
        for index, (label, _chain) in enumerate(options, start=1):
            print(f"  {index}) {label}")
        raw = input(f"choose 1-{len(options)} [1]: ").strip() or "1"
        try:
            pick = int(raw)
        except ValueError:
            pick = 0
        if not 1 <= pick <= len(options):
            print("  unrecognized choice; keeping current")
            continue
        label, selected = options[pick - 1]
        if selected is None:
            custom = input("  chain: ").strip()
            try:
                selected = _parsechain_text(custom)
            except ValueError as exc:
                print(f"  {exc}; keeping current")
                continue
        if selected == current:
            continue
        if selected == default_chain:
            categories.pop(name, None)
        else:
            categories[name] = [
                {"model": model, "reasoning_effort": effort} for model, effort in selected
            ]
        changed += 1
    if not changed:
        print("\nNo changes.")
        return 0
    try:
        path = _write_document(omh_home, document)
    except ValueError as exc:
        print(f"omh: refused to write an invalid document: {exc}", file=sys.stderr)
        return 2
    print(f"\nSaved {changed} categor{'y' if changed == 1 else 'ies'} to {path}.")
    _print_state(_state(omh_home, hermes_home))
    return 0


def model_chains_interview(paths) -> int:
    """Smart Hermes-aware model chain interview on a terminal."""
    if not _stdin_is_tty():
        print(
            "omh: interview needs a terminal; use `omh model-chains show` and "
            "`omh model-chains set <category> \"model[:effort], ...\"` instead.",
            file=sys.stderr,
        )
        return 2

    omh_home = paths.omh_home
    overrides, _ = load_mixture_chain_overrides(omh_home)
    document = read_override_document(omh_home)
    categories = document["categories"]
    assert isinstance(categories, dict)

    use_color = _use_color()
    bold = "\033[1m" if use_color else ""
    cyan = "\033[36m" if use_color else ""
    green = "\033[32m" if use_color else ""
    yellow = "\033[33m" if use_color else ""
    dim = "\033[2m" if use_color else ""
    reset = "\033[0m" if use_color else ""

    inv = _discover_hermes_inventory()
    featured = inv["featured"]
    default_model = inv["default_model"] or "default"
    profile = inv["profile"]

    # When no Hermes models are discovered (e.g. headless unit tests or fresh environment without Hermes),
    # fall back to the standard category walk.
    if not featured:
        return _legacy_category_walk(omh_home, overrides, document, categories, paths.hermes_home)

    print()
    print(f"{bold}======================================================================{reset}")
    print(f"  {bold}OMH Model Chains — Hermes-Aware Smart Setup{reset}")
    print(f"{bold}======================================================================{reset}")
    print(f"  Active Profile : {cyan}{profile}{reset}")
    print(f"  Default Model  : {green}{default_model}{reset}")
    print(f"  Hermes Models  : {bold}{len(inv['all'])}{reset} discovered ({len(featured)} featured)")
    print()

    def show_featured_list(limit: int = 15) -> None:
        print(f"{bold}Discovered Hermes Models:{reset}")
        to_show = featured[:limit] if limit > 0 else featured
        for idx, item in enumerate(to_show, start=1):
            tier_badge = f"[{item['tier'].upper()}]"
            print(f"  {cyan}[{idx:2d}]{reset} {bold}{item['id']:<34}{reset} {dim}{item['provider']:<16}{reset} {yellow}{tier_badge}{reset}")
        if limit > 0 and len(featured) > limit:
            print(f"  {dim}... and {len(featured) - limit} more models. Enter 'l' to list all, or 's <query>' to search.{reset}")
        print()

    show_featured_list(15)

    print(f"{bold}Select Configuration Mode:{reset}")
    print(f"  {cyan}1){reset} {bold}Auto-Align with Hermes Models{reset} {green}(Recommended){reset}")
    print(f"     Intelligently maps your top reasoning, fast, and default models to tiers.")
    print(f"  {cyan}2){reset} {bold}3-Tier Quick Setup{reset}")
    print(f"     Assign 3 models (Reasoning, General, Fast) to generate all 12 categories.")
    print(f"  {cyan}3){reset} {bold}Single-Model Unified{reset}")
    print(f"     Route all categories to '{default_model}' with standard effort ladder.")
    print(f"  {cyan}4){reset} {bold}Guided Category Walk{reset}")
    print(f"     Walk categories and pick models by number (e.g. '1', '4:high, 1').")
    print(f"  {cyan}5){reset} Keep current configuration & exit")
    print()

    while True:
        raw_mode = input(f"Choose option 1-5 [1]: ").strip() or "1"
        if raw_mode.lower() == "l":
            show_featured_list(0)
            continue
        elif raw_mode.lower().startswith("s "):
            term = raw_mode[2:].strip().lower()
            matches = [m for m in inv["all"] if term in m["id"].lower() or term in m["provider"].lower()]
            print()
            print(f"Search results for '{term}' ({len(matches)} matches):")
            for idx, item in enumerate(matches[:25], start=1):
                g_idx = inv["all"].index(item) + 1
                print(f"  {cyan}[#{g_idx:3d}]{reset} {item['id']} [{item['provider']}]")
            print()
            continue
        break

    new_categories: dict[str, list[dict[str, str]]] = {}

    if raw_mode == "1":
        # Auto-align
        reasoning = [m["id"] for m in featured if m["tier"] == "reasoning"]
        fast = [m["id"] for m in featured if m["tier"] == "fast"]
        top_reasoning = reasoning[:2] if reasoning else [default_model]
        top_fast = fast[:2] if fast else [default_model]

        for cat, (desc, effort, role_tier) in CATEGORY_METADATA.items():
            if role_tier == "reasoning":
                chosen = top_reasoning
            elif role_tier == "fast":
                chosen = top_fast
            else:
                chosen = [default_model]
            chain = [{"model": m, "reasoning_effort": effort} for m in chosen]
            if default_model not in chosen and role_tier != "reasoning":
                chain.append({"model": default_model, "reasoning_effort": effort})
            new_categories[cat] = chain

    elif raw_mode == "2":
        # 3-Tier Quick Setup
        print()
        print(f"{bold}3-Tier Quick Setup:{reset}")
        print("Pick models by entering their number from the list above (or type model ID).")
        def_r = next((str(i) for i, m in enumerate(featured, 1) if m["tier"] == "reasoning"), "1")
        def_f = next((str(i) for i, m in enumerate(featured, 1) if m["tier"] == "fast"), "1")

        p_r = input(f"  1. Heavy Reasoning model [{def_r}]: ").strip() or def_r
        p_g = input(f"  2. General Work model [1]: ").strip() or "1"
        p_f = input(f"  3. Fast / Low-Latency model [{def_f}]: ").strip() or def_f

        r_id, _ = _resolve_model_token(p_r, inv, "xhigh")
        g_id, _ = _resolve_model_token(p_g, inv, "medium")
        f_id, _ = _resolve_model_token(p_f, inv, "low")

        for cat, (desc, effort, role_tier) in CATEGORY_METADATA.items():
            if role_tier == "reasoning":
                chosen = [r_id, g_id] if r_id != g_id else [r_id]
            elif role_tier == "fast":
                chosen = [f_id, g_id] if f_id != g_id else [f_id]
            else:
                chosen = [g_id]
            new_categories[cat] = [{"model": m, "reasoning_effort": effort} for m in chosen]

    elif raw_mode == "3":
        # Single-Model Unified
        print()
        print(f"Routing all categories to {green}{default_model}{reset} with scaled effort.")
        for cat, (desc, effort, role_tier) in CATEGORY_METADATA.items():
            new_categories[cat] = [{"model": default_model, "reasoning_effort": effort}]

    elif raw_mode == "4":
        # Guided Category Walk
        print()
        print(f"{bold}Guided Category Walk:{reset}")
        print("Enter model numbers (e.g. '1', '2, 1', '3:high'), 'd' for shipped default, or Enter to keep current.")
        for cat, (desc, def_effort, role_tier) in CATEGORY_METADATA.items():
            curr = overrides.get(cat, HERMES_MIXTURE_CATEGORY_CHAINS.get(cat, ()))
            curr_text = chain_text(curr)
            print()
            print(f"[{bold}{cat}{reset}] — {dim}{desc}{reset}")
            print(f"  Current: {curr_text}")
            
            suggs = []
            if role_tier == "reasoning":
                top_m = [str(i) for i, m in enumerate(featured, 1) if m["tier"] == "reasoning"][:2]
            elif role_tier == "fast":
                top_m = [str(i) for i, m in enumerate(featured, 1) if m["tier"] == "fast"][:2]
            else:
                top_m = ["1"]
            for s in top_m:
                idx = int(s)
                suggs.append(f"[{s}] {featured[idx-1]['id']}")
            print(f"  Suggestions: {', '.join(suggs)}")

            val = input(f"  Choose model(s) (or Enter to keep): ").strip()
            if not val:
                continue
            if val.lower() in ("d", "default", "clear"):
                categories.pop(cat, None)
                continue

            entries = []
            for part in val.split(","):
                res = _resolve_model_token(part, inv, def_effort)
                if res:
                    entries.append({"model": res[0], "reasoning_effort": res[1]})
            if entries:
                new_categories[cat] = entries

    else:
        print("Keeping current configuration.")
        return 0

    if new_categories:
        _print_preview_table(new_categories, use_color=use_color)
        confirm = input(f"Apply this configuration to {omh_home}/routing/model-chains.json? [Y/n]: ").strip().lower()
        if confirm in ("n", "no"):
            print("Cancelled; no changes were made.")
            return 0

        categories.update(new_categories)
        try:
            path = _write_document(omh_home, document)
        except ValueError as exc:
            print(f"omh: refused to write an invalid document: {exc}", file=sys.stderr)
            return 2
        print()
        print(f"{green}Saved {len(new_categories)} categor{'y' if len(new_categories) == 1 else 'ies'} to {path}.{reset}")
        _print_state(_state(omh_home, paths.hermes_home))
    else:
        print()
        print("No changes made.")

    return 0


def cmd_model_chains_interview(args: argparse.Namespace) -> int:
    return model_chains_interview(_paths(args))


def _provider_kind_error(kind: str) -> str:
    return (
        f"omh: unknown provider kind {kind!r}; choose one of "
        + ", ".join(PROVIDER_KIND_VOCABULARY)
    )


def cmd_model_chains_provider_set(args: argparse.Namespace) -> int:
    """Record what one provider serves, without a terminal.

    The scriptable counterpart of the `omh setup` provider question, which a
    `--yes`, `--json`, or non-TTY install never reaches. A kind outside the
    vocabulary is refused by name rather than recorded: an unresolvable kind
    is the exact state the `model-chains show` hint exists to make visible,
    so writing one from a typo would manufacture it.
    """
    paths = _paths(args)
    provider_id = str(args.provider)
    kind = str(args.kind)
    if not is_provider_id_token(provider_id):
        print(f"omh: {provider_id!r} is not a plain provider identifier", file=sys.stderr)
        return 2
    if kind not in PROVIDER_KIND_VOCABULARY:
        print(_provider_kind_error(kind), file=sys.stderr)
        return 2
    return _report_provider_record(args, paths.omh_home, provider_id, kind)


def cmd_model_chains_provider_clear(args: argparse.Namespace) -> int:
    """Remove one provider from the record; detection counts it again afterwards."""
    paths = _paths(args)
    provider_id = str(args.provider)
    if not is_provider_id_token(provider_id):
        print(f"omh: {provider_id!r} is not a plain provider identifier", file=sys.stderr)
        return 2
    return _report_provider_record(args, paths.omh_home, provider_id, None)


def _report_provider_record(args: argparse.Namespace, omh_home, provider_id: str, kind: str | None) -> int:
    from .provider_entitlements import record_provider_kind

    payload = record_provider_kind(omh_home, provider_id, kind)
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    status = str(payload["status"])
    if status == "refused_invalid_document":
        # Never silently discard an operator document a script has not read.
        if not getattr(args, "json", False):
            print(
                f"omh: {payload['path']} is not readable as a provider record "
                f"[{payload['document_status']}]; fix or delete it first",
                file=sys.stderr,
            )
        return 2
    if not getattr(args, "json", False):
        if status == "unchanged":
            print(f"{provider_id} already recorded as expected in {payload['path']}; nothing written")
        elif status == "cleared":
            print(f"Cleared {provider_id} from {payload['path']}; this machine's own providers count it again")
        else:
            print(f"Recorded {provider_id} as {kind} in {payload['path']}; `omh model-chains show` prints the effect")
    return 0


def _add_model_chains_commands(sub) -> None:
    chains = sub.add_parser(
        "model-chains",
        aliases=["model"],
        help="View and edit the per-category mixture model chains (routing, fallback, HUD labels).",
    )
    chains.add_argument("--json", action="store_true", help="Print the machine-readable state payload.")
    chains.set_defaults(func=cmd_model_chains_pick, model_chains_command=None)
    chains_sub = chains.add_subparsers(dest="model_chains_command")

    show = chains_sub.add_parser("show", help="Show the effective chain per category and its origin.")
    show.add_argument("--json", action="store_true", help="Print the machine-readable state payload.")
    show.add_argument(
        "--sensitive",
        action="store_true",
        help=(
            "Declare the work sensitive: drop every model whose contract does not document a "
            "permitting data-handling default, naming each exclusion (a model of unknown policy "
            "is excluded, never silently kept). Exits 1 if a category is left with no model."
        ),
    )
    show.set_defaults(func=cmd_model_chains_show)

    set_cmd = chains_sub.add_parser("set", help="Replace one category's chain (or --clear it back to the default).")
    set_cmd.add_argument("category", help="Mixture category name (e.g. quick, architect).")
    set_cmd.add_argument("chain", nargs="*", help='Replacement chain: "model[:effort], model[:effort], ..."')
    set_cmd.add_argument("--clear", action="store_true", help="Remove the override so the shipped default applies again.")
    set_cmd.add_argument("--json", action="store_true", help="Print the machine-readable state payload.")
    set_cmd.set_defaults(func=cmd_model_chains_set)

    interview = chains_sub.add_parser(
        "interview",
        help="Walk every category with numbered choices (keep / default / Ultrafast / custom).",
    )
    interview.set_defaults(func=cmd_model_chains_interview)

    # The provider record lives beside the chains because it is what reorders
    # them, and `show` already prints it. These two are the scriptable half
    # of the `omh setup` provider question, which no `--yes`, `--json`, or
    # non-TTY install ever reaches.
    provider = chains_sub.add_parser(
        "provider",
        help="Record what one of this machine's providers serves (scriptable; `omh setup` asks a person).",
    )
    provider_sub = provider.add_subparsers(dest="model_chains_provider_command", required=True)

    provider_set = provider_sub.add_parser("set", help="Record one provider id as serving one model family.")
    provider_set.add_argument("provider", help="Provider id as Hermes names it (a `providers.<id>` key).")
    provider_set.add_argument("kind", help="One of: " + ", ".join(PROVIDER_KIND_VOCABULARY))
    provider_set.add_argument("--json", action="store_true", help="Print the machine-readable result payload.")
    provider_set.set_defaults(func=cmd_model_chains_provider_set)

    provider_clear = provider_sub.add_parser("clear", help="Remove one provider id from the record.")
    provider_clear.add_argument("provider", help="Provider id to stop recording.")
    provider_clear.add_argument("--json", action="store_true", help="Print the machine-readable result payload.")
    provider_clear.set_defaults(func=cmd_model_chains_provider_clear)
