"""Strata nodes: start -> run one prompt -> stop, using an already configured Strata folder.

Strata (https://github.com/Niko1221/Strata) runs Qwen3.8-Flash-Next on one GPU plus
system RAM and serves an OpenAI-compatible API. Only ONE instance can be resident at a
time (loading it pins tens of GB), so the node owns the whole lifecycle:

    1. read <strata_dir>/strata-<model>.json   (engine exe + args, cwd, tokenizer, port,
                                                sampling, reasoning_effort, vision)
    2. read <strata_dir>/run-<model>.bat       (the exact start command setup wrote)
    3. if something already listens on that port, stop it gracefully first
    4. spawn the server, poll /health until "loaded": true
    5. POST /v1/chat/completions with the prompt (picture parts optional, in socket order)
    6. stop the server: SIGTERM/terminate -> the server sends QUIT to its engine child
       and frees VRAM/RAM; force-kill only if it refuses to end

Nodes:
    StrataRun      prompt + media references in, text + reasoning + run metadata out
    StrataInspect  a folder path in, the parsed config / start command out
"""

from __future__ import annotations

from comfy_api.latest import ComfyExtension, io

import glob
import hashlib
import json
import os
import re
import urllib.error
import shutil
import subprocess
import threading
import time
import urllib.request

MAX_BODY = 64 * 1024 * 1024

try:
    import psutil          # the process list; ComfyUI ships it, but do not require it
except ImportError:
    psutil = None


def _load_config(strata_dir: str) -> dict:
    """The engine config setup wrote for this folder: strata-<model>.json."""
    root = os.path.abspath(strata_dir.strip().strip('"'))
    if not os.path.isdir(root):
        raise ValueError(f"not a folder: {root}")
    found = [p for p in glob.glob(os.path.join(root, "strata-*.json"))
             if not p.endswith(".deps.json")]
    if not found:
        raise ValueError(
            "no strata-<model>.json next to this folder - it is not a configured "
            "Strata install (run START-HERE.bat --yes ... first)")
    # the config whose name matches a run-<model>.bat wins when several are installed
    def tag_of(path: str) -> str:
        base = os.path.basename(path)
        if base.startswith("strata-"):
            return base[len("strata"):-len(".json")].lstrip("-")
        return base[len("run-"):].removesuffix(".bat").removesuffix(".sh")

    bats = {tag_of(p) for p in glob.glob(os.path.join(root, "run-*.bat"))}
    chosen = next((p for p in found if tag_of(p) in bats), found[0])
    with open(chosen, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["_path"] = chosen
    cfg["_root"] = root
    return cfg


def _read_run_bat(cfg: dict) -> tuple[list[str], list[str]]:
    """The start command setup wrote for this model, as (python_exe, server_args).

    run-<model>.bat is the source of truth for which interpreter and which flags Strata
    expects; we reuse it verbatim and drop only --open (no browser from a graph node).
    """
    root = cfg["_root"]
    name = os.path.basename(cfg["_path"])
    tag = name[len("strata"):-len(".json")].lstrip("-")
    bat = os.path.join(root, f"run-{tag}.bat")
    if not os.path.isfile(bat):
        hits = glob.glob(os.path.join(root, "run-*.bat"))
        bat = hits[0] if hits else ""
    py, args = "", []
    if bat:
        with open(bat, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                low = line.lower()
                if "server.py" not in low:
                    continue
                toks = re.findall(r'"([^"]+)"', line)
                if not toks:
                    continue
                i = next((k for k, t in enumerate(toks) if t.endswith("server.py")), None)
                if i is None:
                    continue
                py = toks[i - 1] if i >= 1 else ""
                args = [t for t in toks[i + 1:] if t.lower() not in ("--open",)]
                break
    if not py:
        cand = os.path.join(root, ".venv", "Scripts", "python.exe")
        py = cand if os.path.isfile(cand) else (shutil.which("python") or "")
    server_py = os.path.join(root, "serve", "server.py")
    if not os.path.isfile(server_py):
        raise ValueError(f"serve/server.py missing under {root}")
    return [py, server_py], args


def _health(port: int) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except OSError:
        return None


def _pid_on_port(port: int) -> int:
    """The PID listening on 127.0.0.1:port, from netstat (Windows has no lsof)."""
    out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True,
                        timeout=20).stdout
    for line in out.splitlines():
        m = re.search(r"127\.0\.0\.1:%d\s+\S+\s+LISTENING\s+(\d+)" % port, line)
        if m:
            return int(m.group(1))
    return 0


def _proc_rows() -> list[tuple[int, int, str, str]]:
    """Every live process as (pid, parent pid, command line, working folder).

    psutil when the host has it (ComfyUI ships it), tasklist otherwise. The working
    folder is kept separately because the launcher chain (cmd.exe running START-HERE.bat,
    setup.py) often has only bare names in its command line.
    """
    if psutil is not None:
        try:
            return [(p.info["pid"], p.info["ppid"],
                     " ".join(p.info["cmdline"] or []), str(p.info.get("cwd") or ""))
                    for p in psutil.process_iter(["pid", "ppid", "cmdline", "cwd"])]
        except Exception:            # any psutil hiccup: fall back to tasklist
            pass
    rows = []
    out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True,
                         timeout=30).stdout
    for line in out.splitlines():
        cols = [c.strip('"') for c in line.split('","')]
        if len(cols) >= 4 and cols[1].isdigit():
            rows.append((int(cols[1]), 0, "", ""))
    return rows


def _alive(pid: int) -> bool:
    """Is a PID still in the process list? psutil first, tasklist as fallback."""
    if psutil is not None:
        try:
            return psutil.pid_exists(pid)
        except Exception:            # unknown: assume it is still around
            return True
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                         capture_output=True, text=True, timeout=30).stdout
    for line in out.splitlines():
        cols = [c.strip('"') for c in line.split('","')]
        if len(cols) >= 2 and cols[1].isdigit() and int(cols[1]) == pid:
            return True
    return False

def _strata_pids(root: str, skip: set[int] | None = None) -> list[int]:
    """Pids of live processes whose command line or folder belongs to this Strata install.

    That is the server, its engine child, the vision helper and the launcher chain
    (cmd.exe running START-HERE.bat / run-<model>.bat, setup.py). `skip` hides the caller's
    own process tree - the node passes its interpreter here before spawning, since the path
    appears in that very command line.
    """
    key = os.path.abspath(root).lower()
    skip = _subtree_pids(os.getpid()) if skip is None else set(skip)
    out: list[int] = []
    for pid, _ppid, cmd, cwd in _proc_rows():
        if pid in skip or pid == os.getpid():
            continue
        cmd, cwd = cmd.lower(), cwd.lower()
        inside = bool(cwd) and cwd.startswith(key)
        if not (inside or key in cmd):
            continue
        # something that only mentions the folder in passing (an editor, a shell) is
        # not a Strata process; leave it alone so nothing unrelated ever gets killed
        if not (inside or any(m in cmd for m in ("server.py", "setup.py", ".bat", ".cmd"))):
            continue
        out.append(pid)
    return sorted(set(out))


def _parents_first(pids: list[int]) -> list[int]:
    """Order pids so a launcher is killed before the server it starts.

    Killing the child first lets cmd.exe running run-<model>.bat fall through to the next
    line and start a fresh server, so the top of every chain has to go first.
    """
    if psutil is None:
        return pids
    parent = {}
    try:
        for p in psutil.process_iter(["pid", "ppid"]):
            parent[p.info["pid"]] = p.info["ppid"]
    except (psutil.Error, OSError):
        return pids
    def depth(pid: int) -> int:
        d = 0
        seen = set()
        while parent.get(pid, 0) in set(pids) - {pid} and pid not in seen:
            seen.add(pid)
            pid = parent[pid]
            d += 1
        return d
    return sorted(pids, key=depth)


def _subtree_pids(pid: int) -> set[int]:
    """The pid plus every process below it in the tree.

    server.py re-execs itself (the venv interpreter hands over to another python), so the
    process that really listens on the port is a CHILD of what Popen returned - a plain
    comparison against proc.pid would call our own server "foreign".
    """
    out = {pid}
    if psutil is None:
        return out
    try:
        stack = [psutil.Process(pid)]
        while stack:
            for child in stack.pop().children():
                if child.pid not in out:
                    out.add(child.pid)
                    stack.append(child)
    except (psutil.Error, OSError, KeyError, AttributeError):
        pass
    return out


def _kill_pid(pid: int, timeout: float = 240.0) -> bool:
    """Terminate a process together with its whole subtree, escalating until it is gone.

    /T matters: the engine (strata.exe, tens of GB pinned) is a CHILD of server.py, so a
    plain kill leaves it behind. The chain is walked from the launcher down, because
    killing only the socket owner lets the batch/launcher parent start it up again.
    """
    for flags in (["/T"], ["/F", "/T"], ["/F"]):
        try:
            subprocess.run(["taskkill", "/PID", str(pid)] + flags, capture_output=True,
                           text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            pass
        deadline = time.time() + timeout
        while time.time() < deadline and _alive(pid):
            time.sleep(2.0)
        if not _alive(pid):
            return True
    return False


def _stop_existing(port: int, root: str = "", timeout: float = 420.0) -> bool:
    """Free the GPU first: only one Strata instance can be resident at a time.

    The whole launcher chain is stopped, not just the socket owner: killing only that
    python process lets the parent (cmd.exe running run-<model>.bat / START-HERE.bat)
    start a fresh server on the same port again, which then answers every /health.
    Raises when the port stays busy - in that case nothing is loaded.
    """
    pid = _pid_on_port(port)
    targets = set(_strata_pids(root)) if root else set()
    if pid and pid != os.getpid():
        targets.add(pid)
    if not targets:
        return False
    print(f"[strata] port {port} already in use (pids {sorted(targets)}); stopping them first",
          flush=True)
    _post(port, "/unload")            # ask the engine to free its tens of GB gracefully
    deadline = time.time() + timeout
    while True:
        left = _parents_first(sorted(t for t in targets | set(_strata_pids(root)) if _alive(t)))
        busy = _pid_on_port(port)
        if not left and not busy:
            return True
        if time.time() > deadline:
            raise ValueError(f"the server on port {port} would not stop (still pid "
                             f"{busy or left}) - close its window first")
        for victim in left:
            _kill_pid(victim, timeout=min(45.0, max(10.0, deadline - time.time())))
        time.sleep(2.0)


def _post(port: int, path: str) -> None:
    """Ask the running server to unload the model (QUIT to the engine, memory freed)."""
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                     data=b"{}", headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=600).read()   # noqa: S31
    except (urllib.error.URLError, OSError):
        pass


def _spawn(cfg: dict, port: int) -> subprocess.Popen:
    """Start the server, keeping its stdout drained so the pipe never fills up."""
    (py, server_py), extra = _read_run_bat(cfg)
    cfg_path = cfg["_path"]
    cmd = [py, server_py, "--engine", "strata", "--config", cfg_path, "--port", str(port)]
    # keep whatever else setup put in the run script (api key, gpu choice, ...),
    # skipping the flags we already set ourselves so argparse cannot see them twice
    skip_value = {"--engine", "--config", "--port", "--gpu"}
    i = 0
    while i < len(extra):
        flag = extra[i]
        if flag.lower() in skip_value or flag.lower() == "--open":
            i += 2 if flag.lower() in skip_value else 1
        else:
            cmd.append(flag)
            i += 1
    if "--gpu" not in {x.lower() for x in cmd} and cfg.get("gpu") is not None:
        cmd += ["--gpu", str(cfg["gpu"])]
    log = os.path.join(cfg["_root"], f"_comfynode-{port}.log")
    print("[strata] start command:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=cfg["_root"], stdout=subprocess.PIPE,
                            stderr=open(log, "a", encoding="utf-8", errors="replace"),
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    threading.Thread(target=_drain, args=(proc,), name="strata-stdout").start()
    return proc


def _drain(proc: subprocess.Popen) -> None:
    """Keep the child's stdout pipe empty; a full pipe would block the server."""
    try:
        for line in proc.stdout:
            if line.strip():
                print(f"[strata] {line.rstrip()}", flush=True)
    except (OSError, ValueError):
        pass


def _wait_loaded(proc: subprocess.Popen, port: int, timeout: float) -> dict:
    """Poll /health until the engine reports loaded; fail fast if the process dies."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise ValueError(f"the server exited (code {proc.returncode}) before loading - "
                             "see _comfynode-<port>.log")
        owner = _pid_on_port(port)
        if owner and owner not in _subtree_pids(proc.pid):
            # a server we do not control answered on this port: prompting it would be wrong,
            # and waiting for our own copy to load would fight over the same GPU forever
            raise ValueError(f"another process (pid {owner}) owns port {port} - "
                             "the preliminary cleanup did not free it")
        h = _health(port)
        if h and h.get("loaded"):
            return h
        time.sleep(2.0)
    raise ValueError(f"the model was not loaded within {timeout:.0f} s "
                     "(first load can take 1-3 minutes; raise startup_timeout)")


def _chat(port: int, messages: list, max_tokens: int, effort: str,
            timeout: float, api_key: str = "", sampling: dict | None = None) -> tuple[str, str, dict]:
    """One non-streamed /v1/chat/completions call. Returns (content, reasoning, usage)."""
    body = {"model": "strata", "messages": messages, "max_tokens": int(max_tokens),
            "reasoning_effort": effort, "stream": False}
    # the server reads these straight off the request and merges them over the values from
    # strata-<model>.json (serve/server.py: sampling = {**defaults, **req_values}), so only
    # the fields the node was given are sent
    body.update(sampling or {})
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:      # noqa: S31
        data = json.loads(r.read(MAX_BODY).decode("utf-8", "replace"))
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    return (msg.get("content") or "", msg.get("reasoning_content") or "",
            data.get("usage") or {"finish_reason": choice.get("finish_reason")})


def _stop_server(proc: subprocess.Popen, port: int, timeout: float, root: str = "") -> str:
    """Unload the model first (memory back), then end the process and its engine child.

    Ending server.py alone is not enough on Windows: strata.exe runs as its CHILD and can
    keep tens of GB pinned, so every process still belonging to this Strata folder is
    terminated through the process list as well.
    """
    if proc.poll() is not None:
        return "already ended"
    _post(port, "/unload")            # the engine frees its tens of GB itself
    try:
        proc.terminate()          # graceful: lets server.py send QUIT to the engine
    except (OSError, AttributeError):
        pass
    deadline = time.time() + timeout
    while time.time() < deadline and proc.poll() is None:
        time.sleep(1.0)
    if proc.poll() is None:       # stubborn: the engine kept tens of GB pinned
        try:
            proc.kill()
        except (OSError, AttributeError):
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
    # whatever is still left under this folder (engine, launcher chain) gets killed too
    for pid in _parents_first(_strata_pids(root, set())) if root else []:
        _kill_pid(pid, timeout=min(60.0, max(10.0, deadline - time.time())))
    left = [pid for pid in _strata_pids(root, set()) if _alive(pid)] if root else []
    if proc.poll() is not None and not left:
        return "stopped"
    if proc.poll() is not None:
        return f"stopped, {len(left)} related process(es) left"
    return "killed, still running"


def _image_part(image) -> dict:
    """One ComfyUI IMAGE (a single H,W,C frame) to Strata's OpenAI image_url part.

    A batch goes through `_picture_parts` instead, which makes one part per frame so the
    model sees every picture separately, like the reference layout does."""
    import base64
    import io
    import numpy as np
    from PIL import Image

    t = image[0] if getattr(image, "ndim", 3) == 4 else image
    a = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    if a.ndim == 3 and a.shape[0] in (1, 3, 4) and a.shape[-1] not in (1, 3, 4):
        a = np.moveaxis(a, 0, -1)          # CHW -> HWC
    if a.dtype in (np.float32, np.float64):
        a = (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.shape[-1] == 4:
        a = a[..., :3]
    out = io.BytesIO()
    Image.fromarray(a.astype(np.uint8)).save(out, format="PNG")
    return {"type": "image_url",
            "image_url": {"url": "data:image/png;base64,"
                           + base64.b64encode(out.getvalue()).decode()}}


def _picture_parts(images) -> list[dict]:
    """Connected pictures to ordered OpenAI image parts.

    Each connected input may itself be a batch (B,H,W,C): every frame becomes its own part,
    so the server renders one vision block per picture in exactly this order."""
    out: list[dict] = []
    for image in images:
        if image is None:
            continue
        if getattr(image, "ndim", 3) == 4:      # a batch of frames -> one part each
            out.extend(_image_part(frame) for frame in image)
        else:
            out.append(_image_part(image))
    return out


# The result cache lives here rather than on the node object: new-API nodes are class-based,
# so there is no instance to hang state on. One entry at a time, like the old node kept.
_RESULT_CACHE: dict = {}


def _media_digest(pictures) -> tuple:
    """A small fingerprint of connected pictures for the result cache.

    Same shape plus a content hash per picture, so a cached answer is only reused when the very
    same tensors come in - counting sockets alone would reuse it after the workflow swapped one
    picture for another."""
    import numpy as np

    out = []
    for image in pictures:
        if image is None:
            continue
        a = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
        a = np.ascontiguousarray(a)
        out.append((a.shape, a.dtype.str, hashlib.sha1(a.tobytes()).hexdigest()))
    return tuple(out)


class StrataRun(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StrataRun",
            display_name="Strata Run (start / prompt / stop)",
            description=("Owns one port: a server already listening there is found in the process "
                         "list and terminated with its whole launcher chain (model unloaded first), "
                         "the engine is loaded from scratch on the same port, the prompt runs, then "
                         "this server is stopped and VRAM/RAM are freed. Nothing is loaded when the "
                         "port could not be freed first - the run fails instead of stacking a second "
                         "model in memory."),
            category="LLM/Strata",
            inputs=[
                io.String.Input("folder", multiline=False,
                                tooltip="The configured Strata folder (the one holding "
                                       "strata-<model>.json)."),
                io.Int.Input("port", default=8080, min=1, max=65535,
                             tooltip="The port to own. Any Strata server already listening here is "
                                     "stopped first, then the node starts its own on this same port."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True,
                                tooltip="The prompt to execute."),
                io.String.Input("system", multiline=True, default="", dynamic_prompts=True,
                                tooltip="Optional system message."),
                io.Int.Input("max_tokens", default=4096, min=1, max=131072),
                # a Combo is what draws the dropdown in both UIs; 'config' keeps the config's own effort
                io.Combo.Input("thinking", options=["config", "none", "low", "medium", "high"],
                               default="config",
                               tooltip="Reasoning effort; 'config' keeps whatever the Strata config chose."),
                io.Int.Input("startup_timeout", default=900, min=30, max=7200,
                             tooltip="Seconds to wait for the model to load (first load is slowest)."),
                io.Int.Input("prompt_timeout", default=1800, min=60, max=7200),
                # sampling: -1 means "keep whatever strata-<model>.json chose" (the field is then not sent)
                io.Float.Input("temperature", default=1.0, min=-1.0, max=2.0, step=0.05,
                               tooltip="Sampling temperature (config default 1.0); -1 keeps the config value."),
                io.Float.Input("top_p", default=0.95, min=-1.0, max=1.0, step=0.01,
                               tooltip="Nucleus sampling (config default 0.95); -1 keeps the config value."),
                io.Int.Input("top_k", default=20, min=-1, max=64,
                             tooltip="Top-k candidates, engine caps at 64 (config default 20); "
                                     "-1 keeps the config value."),
                io.Float.Input("min_p", default=0.0, min=-1.0, max=1.0, step=0.01,
                               tooltip="Min-p sampling (config default 0.0); -1 keeps the config value."),
                io.Float.Input("presence_penalty", default=0.0, min=-1.0, max=2.0, step=0.05,
                               tooltip="Penalise tokens already present (config default 0.0); "
                                       "-1 keeps the config value."),
                io.Float.Input("frequency_penalty", default=0.0, min=-1.0, max=2.0, step=0.05,
                               tooltip="Penalise tokens by how often they appear (config default 0.0); "
                                       "-1 keeps the config value."),
                io.Float.Input("repetition_penalty", default=1.05, min=-1.0, max=2.0, step=0.05,
                               tooltip="Penalise any token seen in the last window (config default 1.05); "
                                       "-1 keeps the config value."),
                io.Int.Input("penalty_last_n", default=512, min=-1, max=8192,
                             tooltip="How many past tokens the penalties look at (default 512; the config "
                                     "uses 64); -1 keeps the config value."),
                io.Boolean.Input("unload_after", default=True,
                                 tooltip="Stop the server and free VRAM/RAM when the prompt finishes."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffff,
                             tooltip="Non-zero reuses the cached result instead of loading the model again; "
                                     "it is also the engine's sampling seed."),
                io.Boolean.Input("bypass", default=False,
                                 tooltip="Pass the prompt straight through to the result: Strata is not read, "
                                         "started, stopped or touched at all."),
                # Reference pictures mirror LLMTextProcessorVL counts and order so a workflow can feed
                # the same connections to both nodes; Strata reads image parts only.
                io.Autogrow.Input("ref_images", optional=True,
                                  template=io.Autogrow.TemplatePrefix(
                                      input=io.Image.Input("ref_image"), prefix="ref_image_", min=0, max=9)),
                io.Image.Input("first_frame", optional=True,
                               tooltip="First frame; becomes the next Picture only when also_ref_first_frame "
                                       "is set."),
                io.Image.Input("last_frame", optional=True,
                               tooltip="Last frame keyframe pin. Never sent as a Picture to Strata."),
                io.Boolean.Input("also_ref_first_frame", default=False, advanced=True,
                                 tooltip="Also expose first_frame as the next Picture reference."),
            ],
            outputs=[
                io.String.Output(display_name="result", tooltip="The assistant text."),
                io.String.Output(display_name="reasoning",
                                 tooltip="reasoning_content when the model emits it, otherwise empty."),
                io.String.Output(display_name="run info",
                                 tooltip="JSON: model, port, seconds, unload state, token usage."),
            ],
        )

    @classmethod
    def _ordered_pictures(cls, ref_images: dict, first_frame, also_ref_first_frame: bool):
        # Same order rules as LLMTextProcessorVL._ordered_media, pictures only: socket index first,
        # then first_frame appended last when also_ref_first_frame asks for it. last_frame is a
        # keyframe pin in the reference layout and never reaches the model, so it stays unused here.
        def _by_index(mapping):
            items = [(int(key.rsplit("_", 1)[-1]), value)
                     for key, value in mapping.items() if value is not None]
            return [value for _, value in sorted(items, key=lambda pair: pair[0])]

        pictures = _by_index(ref_images)
        if also_ref_first_frame and first_frame is not None:
            # only its first frame joins the Pictures, like the VL node's first_frame[:1]
            pictures.append(first_frame[:1] if getattr(first_frame, "ndim", 3) == 4 else first_frame)
        return pictures

    @classmethod
    def _run(cls, folder, prompt, system, pictures, max_tokens, effort,
             startup_timeout, prompt_timeout, unload, seed, port,
             temperature=-1.0, top_p=-1.0, top_k=-1, min_p=-1.0,
             presence_penalty=-1.0, frequency_penalty=-1.0, repetition_penalty=-1.0,
             penalty_last_n=-1, bypass=False):
        if bypass:
            # the switch is on: the prompt is the result, nothing of Strata is even read
            return (prompt or "", "", json.dumps({"bypass": True, "seconds": 0.0}, indent=2))
        if not (folder or "").strip():
            raise ValueError("the Strata folder path is empty")
        cfg = _load_config(folder)
        want = int(port or 0) or int(cfg.get("port") or 8080)
        if want != int(cfg.get("port") or want):
            print(f"[strata] using port {want} instead of the config's "
                  f"{cfg.get('port')}", flush=True)
        port = want
        effort = (effort or "").strip().lower() or str(cfg.get("reasoning_effort") or "low")
        if effort not in ("none", "low", "medium", "high"):
            raise ValueError(f"thinking must be none/low/medium/high, got {effort!r}")
        messages = []
        if (system or "").strip():
            messages.append({"role": "system", "content": system.strip()})
        parts = _picture_parts(pictures)
        if parts:
            if not cfg.get("vision"):
                raise ValueError("this Strata install has no vision configured")
            messages.append({"role": "user",
                             "content": parts + [{"type": "text",
                                                  "text": (prompt or "").strip()}]})
        else:
            messages.append({"role": "user", "content": (prompt or "").strip()})
        if not (prompt or "").strip() and not parts:
            raise ValueError("the prompt is empty")
        # -1 in a sampling widget means "leave the config value alone": the field is not sent,
        # so serve/server.py keeps strata-<model>.json's own number for it
        want_sampling = {}
        for key, value in (("temperature", temperature), ("top_p", top_p), ("top_k", top_k),
                           ("min_p", min_p), ("presence_penalty", presence_penalty),
                           ("frequency_penalty", frequency_penalty),
                           ("repetition_penalty", repetition_penalty),
                           ("penalty_last_n", penalty_last_n)):
            if value is not None and float(value) >= 0:
                want_sampling[key] = int(value) if key in ("top_k", "penalty_last_n") else float(value)
        if seed:
            want_sampling["seed"] = int(seed)
        stopped = _stop_existing(port, cfg["_root"])
        t0 = time.time()
        proc = _spawn(cfg, port)
        try:
            info = _wait_loaded(proc, port, float(startup_timeout))
            api_key = str(info.get("api_key") or "")
            content, reasoning, usage = _chat(port, messages, int(max_tokens), effort,
                                              float(prompt_timeout), api_key, want_sampling)
        finally:
            state = (_stop_server(proc, port, 180.0, cfg["_root"]) if unload
                     else "left running")
        meta = {"model": info.get("model"), "port": port, "loaded": True,
            "seconds": round(time.time() - t0, 1), "unload": state, "usage": usage,
            "pictures": len(parts), "stopped_existing_server": stopped}
        return (content, reasoning, json.dumps(meta, indent=2))

    @classmethod
    def execute(cls, folder: str, port: int, prompt: str, system: str = "",
              max_tokens: int = 4096, thinking: str = "config",
              startup_timeout: int = 900, prompt_timeout: int = 1800,
              temperature: float = 1.0, top_p: float = 0.95, top_k: int = 20, min_p: float = 0.0,
              presence_penalty: float = 0.0, frequency_penalty: float = 0.0,
              repetition_penalty: float = 1.05, penalty_last_n: int = 512,
              unload_after: bool = True, seed: int = 0, bypass: bool = False,
              ref_images: dict | None = None, first_frame=None, last_frame=None,
              also_ref_first_frame: bool = False):
        # new-API nodes are called by keyword, so this signature no longer has to shadow the
        # schema order; media references come in as one autogrow dict plus the frame pins
        pictures = cls._ordered_pictures(ref_images or {}, first_frame, also_ref_first_frame)
        # only a non-zero seed can reuse a result, so the pictures are fingerprinted just then -
        # hashing them every time would copy every tensor to the CPU on each execution
        media = _media_digest(pictures) if seed else None
        key = (folder, prompt, system, max_tokens, thinking, media, port,
               startup_timeout, prompt_timeout, unload_after, seed, bypass,
               temperature, top_p, top_k, min_p, presence_penalty, frequency_penalty,
               repetition_penalty, penalty_last_n)
        if seed and key in _RESULT_CACHE:
            return io.NodeOutput(*_RESULT_CACHE[key])
        out = cls._run(folder, prompt, system, pictures, max_tokens,
                        "" if thinking == "config" else thinking, startup_timeout,
                        prompt_timeout, unload_after, seed, port, temperature, top_p, top_k,
                        min_p, presence_penalty, frequency_penalty, repetition_penalty,
                       penalty_last_n, bypass)
        _RESULT_CACHE.clear()
        _RESULT_CACHE[key] = out
        return io.NodeOutput(*out)


class StrataInspect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="StrataInspect",
            display_name="Strata Config",
            description="A folder path in, the parsed config and exact start command out. "
                        "No GPU work: nothing is started or stopped.",
            category="LLM/Strata",
            inputs=[
                io.String.Input("folder", multiline=False,
                                tooltip="The configured Strata folder."),
            ],
            outputs=[
                io.String.Output(display_name="config info"),
                io.String.Output(display_name="start command"),
            ],
        )

    @classmethod
    def execute(cls, folder: str):
        cfg = _load_config(folder)
        (py, server_py), extra = _read_run_bat(cfg)
        cmd = " ".join([py, server_py, "--engine", "strata",
                       "--config", cfg["_path"], "--port", str(cfg.get("port"))] + extra)
        h = _health(int(cfg.get("port") or 8080))
        info = {"config": cfg["_path"], "model": cfg.get("model_name"),
                "port": cfg.get("port"), "max_context": cfg.get("max_context"),
                "vision": bool(cfg.get("vision")), "reasoning_effort":
                cfg.get("reasoning_effort"), "sampling": cfg.get("sampling"),
                "engine_exe": cfg.get("exe"), "start_command": cmd,
                "server_running": bool(h and h.get("loaded"))}
        return io.NodeOutput(json.dumps(info, indent=2), cmd)


class StrataExtension(ComfyExtension):
    async def get_node_list(self):
        return [StrataRun, StrataInspect]
