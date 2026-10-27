#!/usr/bin/env python3
"""ComfyUI Workflow Doctor - Playwright-based load + fix loop.

Iteratively loads a workflow JSON in ComfyUI's frontend, captures
errors (missing node types, broken links, console errors, frontend
toasts), applies known-good fixes from the registry, and retries
until clean or max iterations exhausted.

Usage:
  python3 comfy_workflow_doctor.py <workflow.json> [--max-iter N] [--comfy URL]
                                                   [--headless|--headed]
                                                   [--inspect-only]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


# Console-noise we always want to ignore (environmental, not workflow-related)
NOISE_PATTERNS = [
    re.compile(r"favicon\.ico", re.I),
    re.compile(r"manifest\.json", re.I),
    re.compile(r"\.well-known/", re.I),
    re.compile(r"graph accessed before initialization", re.I),
    re.compile(r"deprecated", re.I),
    re.compile(r"deprecation", re.I),
    re.compile(r"monkey-patch", re.I),
    re.compile(r"\bnotice\b", re.I),
    re.compile(r"^Failed to load resource:.*404.*$", re.I),
]


# Known-good node type replacements: applied if a type is reported missing.
NODE_REPLACEMENTS = {
    "MarkdownNote": "Note",
}

# SetUnionControlNetType: legacy short names → modern grouped option.
UNION_TYPE_MAP = {
    "canny": "canny/lineart/anime_lineart/mlsd",
    "lineart": "canny/lineart/anime_lineart/mlsd",
    "mlsd": "canny/lineart/anime_lineart/mlsd",
    "scribble": "hed/pidi/scribble/ted",
    "hed": "hed/pidi/scribble/ted",
    "pidi": "hed/pidi/scribble/ted",
}

# Substrings that identify a Flux-family ControlNet by filename (any case).
# These ControlNets always require a VAE input on ControlNetApplyAdvanced.
FLUX_CN_FILENAME_HINTS = ("flux", "flux1", "flux2", "cnup", "union-pro")

# Runtime exception messages → structured issue tags the auto-fixer recognises.
RUNTIME_PATTERNS = [
    (
        re.compile(r"Controlnet needs a VAE.*ControlNetApply", re.I),
        "missing_cn_vae",
    ),
]


# ---------------------------------------------------------------- Inspection


def is_noise(text: str) -> bool:
    return any(p.search(text) for p in NOISE_PATTERNS)


def sibling_api_path(workflow_path: str) -> Path | None:
    """Return the API-format path for a workflow.
    If the workflow itself is already an API file (*-api.json), return it.
    Otherwise look for a sibling <stem>-api.json.
    """
    p = Path(workflow_path)
    if p.stem.endswith("-api"):
        return p
    cand = p.with_name(p.stem + "-api.json")
    if cand.exists():
        return cand
    return None


def is_api_only_input(workflow_path: str) -> bool:
    return Path(workflow_path).stem.endswith("-api")


# ---------------------------------------------------------------- Static lints


def _api_node_class(api: dict, nid) -> str | None:
    n = api.get(str(nid))
    return n.get("class_type") if n else None


def _api_find_vae_source(api: dict) -> tuple[str, int] | None:
    """Return (node_id, slot) for the first node producing a VAE."""
    for nid, n in api.items():
        ct = n.get("class_type")
        if ct == "VAELoader":
            return (nid, 0)
        if ct == "CheckpointLoaderSimple":
            return (nid, 2)  # MODEL=0 CLIP=1 VAE=2
    return None


def _api_is_flux_cn(api: dict, cn_apply_node: dict) -> bool:
    """Trace ControlNetApplyAdvanced.control_net back to the loader and check filename."""
    src = cn_apply_node.get("inputs", {}).get("control_net")
    if not isinstance(src, list) or not src:
        return False
    visited = set()
    cur_id = src[0]
    while cur_id and cur_id not in visited:
        visited.add(cur_id)
        n = api.get(str(cur_id))
        if not n:
            return False
        ct = n.get("class_type")
        if ct == "ControlNetLoader":
            fn = (n.get("inputs", {}).get("control_net_name") or "").lower()
            return any(h in fn for h in FLUX_CN_FILENAME_HINTS)
        # follow the chain (e.g. SetUnionControlNetType -> ControlNetLoader)
        ref = n.get("inputs", {}).get("control_net")
        if isinstance(ref, list) and ref:
            cur_id = ref[0]
        else:
            return False
    return False


def static_lint_api(api: dict) -> list[dict]:
    """Detect issues that won't surface until runtime (or won't surface at all)."""
    findings: list[dict] = []
    for nid, n in api.items():
        if n.get("class_type") != "ControlNetApplyAdvanced":
            continue
        if "vae" in (n.get("inputs") or {}):
            continue
        if _api_is_flux_cn(api, n):
            findings.append(
                {
                    "kind": "missing_cn_vae",
                    "nid": nid,
                    "detail": (
                        "Flux ControlNet requires VAE on ControlNetApplyAdvanced; "
                        "this node has no `vae` input wired."
                    ),
                }
            )
    return findings


def _is_cloud(comfy_url: str) -> bool:
    return "cloud.comfy.org" in comfy_url


def _api_path(comfy_url: str, suffix: str) -> str:
    """Cloud uses /api/* prefix; local accepts both /* and /api/* but /* is canonical."""
    base = comfy_url.rstrip("/")
    if _is_cloud(comfy_url):
        return f"{base}/api/{suffix.lstrip('/')}"
    return f"{base}/{suffix.lstrip('/')}"


def _auth_headers(api_key: str | None) -> dict:
    h = {"Content-Type": "application/json"}
    if api_key:
        h["X-API-Key"] = api_key
    return h


def api_execute(
    api_path: Path, comfy_url: str, api_key: str | None, timeout_s: int = 1800
) -> dict:
    """Submit + run a workflow to completion. Returns:
        {"ok": True, "outputs": [...], "elapsed": s}
      or
        {"ok": False, "runtime_errors": [{nid, type, message, traceback?}]}
    Cost: a real GPU run. Use only with --execute.
    """
    try:
        api = json.loads(api_path.read_text())
    except Exception as e:
        return {"ok": False, "runtime_errors": [{"_load_error": str(e)}]}
    payload = json.dumps({"prompt": api}).encode("utf-8")
    headers = _auth_headers(api_key)
    req = urllib.request.Request(
        _api_path(comfy_url, "prompt"), data=payload, headers=headers
    )
    try:
        body = json.loads(urllib.request.urlopen(req, timeout=30).read())
    except urllib.error.HTTPError as e:
        return {
            "ok": False,
            "runtime_errors": [
                {
                    "_http": e.code,
                    "_body": e.read().decode("utf-8", errors="replace")[:1500],
                }
            ],
        }
    pid = body.get("prompt_id")
    if not pid:
        return {"ok": False, "runtime_errors": [{"_no_prompt_id": str(body)[:500]}]}

    import time

    start = time.time()
    while True:
        elapsed = int(time.time() - start)
        if elapsed > timeout_s:
            return {
                "ok": False,
                "runtime_errors": [{"_timeout": f"exceeded {timeout_s}s"}],
            }
        try:
            h = json.loads(
                urllib.request.urlopen(
                    urllib.request.Request(
                        _api_path(comfy_url, f"history/{pid}"), headers=headers
                    ),
                    timeout=15,
                ).read()
            )
        except Exception:
            time.sleep(5)
            continue
        entry = h.get(pid)
        if not entry:
            time.sleep(5)
            continue
        st = entry.get("status", {})
        if st.get("completed") and st.get("status_str") != "error":
            outs = []
            for nid, out in entry.get("outputs", {}).items():
                for img in out.get("images", []):
                    outs.append(f"{img.get('subfolder', '')}/{img['filename']}")
            return {"ok": True, "outputs": outs, "elapsed": elapsed}
        if st.get("status_str") == "error":
            errs = []
            for m in st.get("messages", []):
                if isinstance(m, list) and len(m) >= 2 and m[0] == "execution_error":
                    info = m[1]
                    errs.append(
                        {
                            "nid": info.get("node_id"),
                            "type": info.get("node_type"),
                            "message": info.get("exception_message", ""),
                            "exception_type": info.get("exception_type", ""),
                        }
                    )
            return {
                "ok": False,
                "runtime_errors": errs or [{"_unknown": "error w/o details"}],
            }
        time.sleep(5)


def api_validate(api_path: Path, comfy_url: str, api_key: str | None) -> dict:
    """POST API workflow to /prompt; if 400, return node_errors dict.
    If 200, immediately interrupt + delete the queued prompt (Cloud and local).
    """
    try:
        api = json.loads(api_path.read_text())
    except Exception as e:
        return {"_load_error": str(e)}
    payload = json.dumps({"prompt": api}).encode("utf-8")
    headers = _auth_headers(api_key)
    req = urllib.request.Request(
        _api_path(comfy_url, "prompt"), data=payload, headers=headers
    )
    try:
        body = urllib.request.urlopen(req, timeout=20).read()
        prompt_id = None
        try:
            prompt_id = json.loads(body).get("prompt_id")
        except Exception:
            pass
        # Validation passed → cancel run before it eats GPU/Cloud credit
        try:
            urllib.request.urlopen(
                urllib.request.Request(
                    _api_path(comfy_url, "interrupt"), data=b"{}", headers=headers
                ),
                timeout=5,
            )
        except Exception:
            pass
        if prompt_id:
            # Cloud: cancel via {"delete":[prompt_id]}; local accepts same shape
            try:
                urllib.request.urlopen(
                    urllib.request.Request(
                        _api_path(comfy_url, "queue"),
                        data=json.dumps({"delete": [prompt_id]}).encode(),
                        headers=headers,
                        method="POST",
                    ),
                    timeout=5,
                )
            except Exception:
                pass
        else:
            # local also accepts {"clear": true}
            try:
                urllib.request.urlopen(
                    urllib.request.Request(
                        _api_path(comfy_url, "queue"),
                        data=json.dumps({"clear": True}).encode(),
                        headers=headers,
                        method="POST",
                    ),
                    timeout=5,
                )
            except Exception:
                pass
        return {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
            errs = parsed.get("node_errors", {}) or {}
            if not errs:
                # surface the top-level error for visibility (auth/subscription/etc.)
                return {"_raw": body[:1500], "_status": e.code}
            return errs
        except Exception:
            return {"_raw": body[:1500], "_status": e.code}
    except Exception as e:
        return {"_request_error": str(e)}


async def inspect(
    workflow_path: str,
    comfy_url: str,
    headless: bool,
    api_key: str | None = None,
    skip_frontend: bool = False,
    execute: bool = False,
    execute_timeout: int = 1800,
) -> dict:
    issues = {
        "console": [],
        "missing_nodes": [],
        "node_errors": [],
        "api_errors": {},
        "static_lints": [],
        "runtime_errors": [],
        "toasts": [],
        "load_exception": None,
    }

    def _run_static_and_api(api_path):
        try:
            api_dict = json.loads(api_path.read_text())
            issues["static_lints"] = static_lint_api(api_dict)
        except Exception as e:
            issues["static_lints"] = [{"kind": "_load_error", "detail": str(e)}]
        issues["api_errors"] = api_validate(api_path, comfy_url, api_key)
        issues["_api_path"] = str(api_path)

    if skip_frontend:
        api_path = sibling_api_path(workflow_path)
        if api_path:
            _run_static_and_api(api_path)
            if execute and not issues["api_errors"] and not issues["static_lints"]:
                res = api_execute(api_path, comfy_url, api_key, execute_timeout)
                if not res.get("ok"):
                    issues["runtime_errors"] = res.get("runtime_errors", [])
                else:
                    issues["_outputs"] = res.get("outputs", [])
                    issues["_elapsed"] = res.get("elapsed")
        return issues
    # Imported here so --api-only and the pure helpers work without Playwright installed.
    from playwright.async_api import async_playwright, TimeoutError as PWTimeout

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        ctx = await browser.new_context(viewport={"width": 1600, "height": 1000})
        page = await ctx.new_page()

        page.on(
            "console",
            lambda m: issues["console"].append({"type": m.type, "text": m.text}),
        )
        page.on(
            "pageerror",
            lambda e: issues["console"].append({"type": "pageerror", "text": str(e)}),
        )

        await page.goto(comfy_url, wait_until="domcontentloaded", timeout=30_000)
        try:
            await page.wait_for_function(
                "() => window.app && window.app.graph && window.app.graph._nodes",
                timeout=20_000,
            )
        except PWTimeout:
            issues["load_exception"] = "ComfyUI app object did not appear in 20s"
            await browser.close()
            return issues

        wf = json.loads(Path(workflow_path).read_text())

        load_result = await page.evaluate(
            """
            async (wf) => {
                try {
                    await app.loadGraphData(wf, true, true);
                    return {ok: true};
                } catch (e) {
                    return {ok: false, err: String(e), stack: e?.stack};
                }
            }
            """,
            wf,
        )
        if not load_result.get("ok"):
            issues["load_exception"] = load_result.get("err")

        # Let frontend settle (missing-nodes dialogs, validations)
        await asyncio.sleep(2.5)

        graph_state = await page.evaluate(
            """
            () => {
                if (!window.app || !window.app.graph) return {nodes: [], missing: []};
                const result = [];
                const missing = [];
                for (const n of app.graph._nodes) {
                    const entry = {
                        id: n.id,
                        type: n.type,
                        title: n.title,
                        has_errors: !!n.has_errors,
                        is_missing: false,
                    };
                    // Unknown node detection (varies across frontend versions):
                    //   - LiteGraph sets type === 'unknown' or marks comfyClass missing
                    //   - Some versions set `n.flags.unknown = true`
                    if (
                        n.type === undefined
                        || n.flags?.unknown
                        || n.constructor?.name === 'LGraphNode'  // fallback class
                        || n.has_errors
                    ) {
                        entry.is_missing = true;
                        missing.push(entry);
                    }
                    if (entry.has_errors) result.push(entry);
                }
                return {nodes: app.graph._nodes.length, errors: result, missing};
            }
            """
        )
        issues["node_errors"] = graph_state.get("errors", [])
        issues["missing_nodes"] = graph_state.get("missing", [])

        # Capture missing-nodes dialog content if open
        dialog_text = await page.evaluate(
            """
            () => {
                const sels = [
                    '.comfy-missing-nodes',
                    '[class*="missing-nodes"]',
                    '.p-dialog .p-dialog-content',
                ];
                for (const s of sels) {
                    const el = document.querySelector(s);
                    if (el && /missing|unknown|not.*found/i.test(el.textContent)) {
                        return el.textContent.trim().slice(0, 1000);
                    }
                }
                return null;
            }
            """
        )
        if dialog_text:
            issues["toasts"].append(f"[dialog] {dialog_text}")

        toasts = await page.evaluate(
            """
            () => Array.from(
                document.querySelectorAll(
                    '.p-toast-message, .p-toast .p-toast-detail, .comfyui-toast'
                )
            ).map(e => e.textContent.trim()).filter(Boolean)
            """
        )
        issues["toasts"].extend(toasts)

        await browser.close()

    # API-level validation (catches widget value errors that frontend ignores)
    api_path = sibling_api_path(workflow_path)
    if api_path:
        _run_static_and_api(api_path)
        if execute and not issues["api_errors"] and not issues["static_lints"]:
            res = api_execute(api_path, comfy_url, api_key, execute_timeout)
            if not res.get("ok"):
                issues["runtime_errors"] = res.get("runtime_errors", [])
            else:
                issues["_outputs"] = res.get("outputs", [])
                issues["_elapsed"] = res.get("elapsed")

    return issues


# ---------------------------------------------------------------- Auto-fix


def _fix_set_union_value(value: str, allowed: list[str] | None) -> str | None:
    """Map a legacy SetUnionControlNetType value to a current grouped option.
    If `allowed` is provided (from API node_errors.input_config.options),
    only return values present in `allowed`.
    """
    candidate = UNION_TYPE_MAP.get(value)
    if candidate and (allowed is None or candidate in allowed):
        return candidate
    if not allowed:
        return None
    # heuristic: find a grouped option that contains the legacy token
    for opt in allowed:
        if value in opt.split("/"):
            return opt
    return None


def _classify_runtime(message: str) -> str | None:
    for pat, tag in RUNTIME_PATTERNS:
        if pat.search(message or ""):
            return tag
    return None


def _add_vae_to_cn_apply(api: dict, wf: dict | None, nid: str) -> list[str]:
    """Add the `vae` input to a ControlNetApplyAdvanced node and wire it
    from the first VAE source found in the workflow. Mutates api+wf."""
    changes = []
    src = _api_find_vae_source(api)
    if not src:
        return ["no VAE source found (no VAELoader / CheckpointLoaderSimple)"]
    src_id, src_slot = src
    n = api.get(str(nid))
    if n and n.get("class_type") == "ControlNetApplyAdvanced":
        n["inputs"]["vae"] = [src_id, src_slot]
        changes.append(f"api[{nid}].vae = [{src_id!r}, {src_slot}]")

    if wf is not None:
        try:
            gid = int(nid)
        except ValueError:
            gid = None
        try:
            gsrc = int(src_id)
        except ValueError:
            gsrc = None
        if gid is not None and gsrc is not None:
            target = next((x for x in wf["nodes"] if x["id"] == gid), None)
            source = next((x for x in wf["nodes"] if x["id"] == gsrc), None)
            if target and source:
                if not any(i.get("name") == "vae" for i in target["inputs"]):
                    last_link = max((l[0] for l in wf["links"]), default=0) + 1
                    target["inputs"].append(
                        {"name": "vae", "type": "VAE", "link": last_link}
                    )
                    # ensure source output slot has the link
                    if src_slot < len(source["outputs"]):
                        source["outputs"][src_slot].setdefault("links", []).append(
                            last_link
                        )
                    dst_slot = len(target["inputs"]) - 1
                    wf["links"].append(
                        [last_link, gsrc, src_slot, gid, dst_slot, "VAE"]
                    )
                    wf["last_link_id"] = last_link
                    changes.append(f"graph[{gid}]: wired vae from node {gsrc}")
    return changes


def auto_fix(workflow_path: str, issues: dict) -> tuple[bool, list[str]]:
    api_only = is_api_only_input(workflow_path)
    wf = None if api_only else json.loads(Path(workflow_path).read_text())
    api_path = Path(issues.get("_api_path")) if issues.get("_api_path") else None
    api = json.loads(api_path.read_text()) if api_path else None
    changes: list[str] = []

    # Static-lint fixes
    for lint in issues.get("static_lints", []):
        if lint.get("kind") == "missing_cn_vae" and api is not None:
            changes.extend(_add_vae_to_cn_apply(api, wf, lint["nid"]))

    # Runtime-error fixes (--execute mode)
    for err in issues.get("runtime_errors", []):
        if not isinstance(err, dict):
            continue
        tag = _classify_runtime(err.get("message", ""))
        if tag == "missing_cn_vae" and api is not None:
            nid = err.get("nid")
            if nid is not None:
                changes.extend(_add_vae_to_cn_apply(api, wf, str(nid)))

    # Set of node types reported missing by the frontend
    missing_types = {
        m["type"]
        for m in issues["missing_nodes"]
        if isinstance(m, dict) and m.get("type")
    }

    # Process API node_errors: each entry has class_type + errors[*].input_name +
    # extra_info.input_config (combo options). We can patch widget values directly.
    api_errors = issues.get("api_errors") or {}
    for nid, err in api_errors.items():
        if not isinstance(err, dict):
            continue
        class_type = err.get("class_type")
        for e in err.get("errors", []):
            if e.get("type") != "value_not_in_list":
                continue
            input_name = e.get("extra_info", {}).get("input_name")
            options = (
                e.get("extra_info", {})
                .get("input_config", [None, {}])[1]
                .get("options", [])
            )
            received = e.get("extra_info", {}).get("received_value")
            replacement = None
            if class_type == "SetUnionControlNetType" and input_name == "type":
                replacement = _fix_set_union_value(received, options)
            if replacement is None:
                continue
            # Patch API file
            if api and nid in api:
                api[nid]["inputs"][input_name] = replacement
                changes.append(
                    f"api[{nid}].{input_name}: '{received}' → '{replacement}'"
                )
            # Patch corresponding graph node (matching by integer id) - graph-only mode
            if wf is not None:
                try:
                    gid = int(nid)
                except ValueError:
                    gid = None
                if gid is not None:
                    for n in wf.get("nodes", []):
                        if n["id"] == gid and n.get("type") == class_type:
                            wv = n.get("widgets_values") or []
                            for i, v in enumerate(wv):
                                if v == received and isinstance(v, str):
                                    wv[i] = replacement
                                    changes.append(
                                        f"graph[{gid}].widgets[{i}]: '{received}' → '{replacement}'"
                                    )

    # Graph-format-only fixes (skipped when input is api-only)
    if wf is not None:
        # Fix 1: replace known-missing node types with frontend-equivalent fallbacks
        for n in wf.get("nodes", []):
            ntype = n.get("type")
            if ntype in missing_types and ntype in NODE_REPLACEMENTS:
                new = NODE_REPLACEMENTS[ntype]
                n["type"] = new
                n.setdefault("properties", {})["Node name for S&R"] = new
                changes.append(f"node {n['id']}: {ntype} → {new}")

        # Fix 2: SetUnionControlNetType legacy short names → grouped values
        for n in wf.get("nodes", []):
            if n.get("type") != "SetUnionControlNetType":
                continue
            wv = n.get("widgets_values") or []
            if wv and isinstance(wv[0], str) and wv[0] in UNION_TYPE_MAP:
                old, new = wv[0], UNION_TYPE_MAP[wv[0]]
                wv[0] = new
                changes.append(
                    f"node {n['id']}: SetUnionControlNetType '{old}' → '{new}'"
                )

        # Fix 3: heuristic - generic 'Node type X not found' from console
        for c in issues["console"]:
            if c["type"] not in ("error", "warning", "warn"):
                continue
            text = c["text"]
            if "not found" in text.lower() or "unknown node" in text.lower():
                for tok in (
                    text.replace(":", " ").replace("'", " ").replace('"', " ").split()
                ):
                    if tok in NODE_REPLACEMENTS:
                        for n in wf.get("nodes", []):
                            if n.get("type") == tok:
                                new = NODE_REPLACEMENTS[tok]
                                n["type"] = new
                                n.setdefault("properties", {})["Node name for S&R"] = (
                                    new
                                )
                                changes.append(
                                    f"node {n['id']}: {tok} → {new} (from console)"
                                )

    if changes:
        if wf is not None:
            Path(workflow_path).write_text(json.dumps(wf, indent=2))
        if api_path and api:
            api_path.write_text(json.dumps(api, indent=2))
    # Dedup while preserving order
    seen, dedup = set(), []
    for c in changes:
        if c not in seen:
            dedup.append(c)
            seen.add(c)
    return bool(dedup), dedup


# ---------------------------------------------------------------- Reporting


def has_real_issues(issues: dict) -> bool:
    if issues["load_exception"]:
        return True
    if issues["missing_nodes"]:
        return True
    if issues.get("api_errors"):
        return True
    if issues.get("static_lints"):
        return True
    if issues.get("runtime_errors"):
        return True
    if any(t for t in issues["toasts"]):
        return True
    if any(
        c
        for c in issues["console"]
        if c["type"] in ("error", "pageerror") and not is_noise(c["text"])
    ):
        return True
    return False


def report(issues: dict) -> None:
    if issues["load_exception"]:
        print(f"  load exception : {issues['load_exception']}")
    real_console = [
        c
        for c in issues["console"]
        if c["type"] in ("error", "pageerror") and not is_noise(c["text"])
    ]
    print(
        f"  console (real) : {len(real_console)} (filtered {len(issues['console']) - len(real_console)} noise)"
    )
    print(f"  missing nodes  : {len(issues['missing_nodes'])}")
    print(f"  node errors    : {len(issues['node_errors'])}")
    print(f"  api errors     : {len(issues.get('api_errors', {}))}")
    print(f"  static lints   : {len(issues.get('static_lints', []))}")
    print(f"  runtime errors : {len(issues.get('runtime_errors', []))}")
    print(f"  toasts/dialogs : {len(issues['toasts'])}")
    if issues.get("_outputs"):
        print(
            f"  outputs        : {len(issues['_outputs'])} ({issues.get('_elapsed', '?')}s)"
        )
        for o in issues["_outputs"]:
            print(f"    out: {o}")
    for c in real_console:
        print(f"    [{c['type']}] {c['text'][:240]}")
    for nid, err in (issues.get("api_errors") or {}).items():
        print(f"    api[{nid}] {str(err)[:240]}")
    for lint in issues.get("static_lints", []):
        print(
            f"    lint[{lint.get('nid')}] {lint.get('kind')}: {lint.get('detail', '')[:200]}"
        )
    for err in issues.get("runtime_errors", []):
        if isinstance(err, dict):
            print(
                f"    runtime[{err.get('nid')}] {err.get('exception_type', '')}: "
                f"{(err.get('message') or '')[:200]}"
            )
    for m in issues["missing_nodes"]:
        print(
            f"    missing : id={m.get('id')} type={m.get('type')} title={m.get('title')}"
        )
    for t in issues["toasts"][:5]:
        print(f"    toast   : {t[:240]}")


# ---------------------------------------------------------------- Main


async def run(args) -> int:
    skip_frontend = _is_cloud(args.comfy) or args.api_only
    for i in range(1, args.max_iter + 1):
        print(f"\n=== Iteration {i}/{args.max_iter} ===")
        issues = await inspect(
            args.workflow,
            args.comfy,
            args.headless,
            api_key=args.api_key,
            skip_frontend=skip_frontend,
            execute=args.execute,
            execute_timeout=args.execute_timeout,
        )
        report(issues)

        if not has_real_issues(issues):
            print(f"\n  Workflow loads cleanly after {i} iteration(s).")
            return 0

        if args.inspect_only:
            print("\n[--inspect-only] not applying fixes.")
            return 1

        applied, changes = auto_fix(args.workflow, issues)
        if not applied:
            print("\n  Issues remain but no auto-fix matched. Stopping.")
            print(
                "  Update NODE_REPLACEMENTS / UNION_TYPE_MAP / heuristics in this script."
            )
            return 1
        print(f"  applied {len(changes)} fix(es):")
        for c in changes:
            print(f"    - {c}")

    print("\n  Max iterations exhausted with issues remaining.")
    return 1


def main():
    import os

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("workflow")
    ap.add_argument("--max-iter", type=int, default=5)
    ap.add_argument("--comfy", default="http://localhost:8188/")
    ap.add_argument(
        "--api-key",
        default=os.environ.get("COMFY_API_KEY"),
        help="X-API-Key for Comfy Cloud (defaults to $COMFY_API_KEY)",
    )
    ap.add_argument(
        "--api-only",
        action="store_true",
        help="Skip Playwright frontend check; rely solely on /api/prompt validation",
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="After validation, actually run the workflow and poll /history for "
        "runtime errors. Costs GPU time. Useful to catch errors that only "
        "surface at execution (e.g. Flux ControlNet missing VAE).",
    )
    ap.add_argument(
        "--execute-timeout",
        type=int,
        default=1800,
        help="Seconds to wait for execution before timing out (default 1800).",
    )
    ap.add_argument("--inspect-only", action="store_true")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--headless", dest="headless", action="store_true", default=True)
    g.add_argument("--headed", dest="headless", action="store_false")
    args = ap.parse_args()

    if _is_cloud(args.comfy) and not args.api_key:
        print(
            "Comfy Cloud requires --api-key (or $COMFY_API_KEY in env)",
            file=sys.stderr,
        )
        return 2

    if not Path(args.workflow).exists():
        print(f"workflow file not found: {args.workflow}", file=sys.stderr)
        return 2

    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
