"""Strata nodes: start -> run one prompt -> stop, using an already configured Strata folder.

Strata (https://github.com/Niko1221/Strata) runs Qwen3.8-Flash-Next on one GPU plus
system RAM and serves an OpenAI-compatible API. Only ONE instance can be resident at a
time (loading it pins tens of GB), so the node owns the whole lifecycle:

    1. read <strata_dir>/strata-<model>.json   (engine exe + args, cwd, tokenizer, port,
                                                sampling, reasoning_effort, vision)
    2. read <strata_dir>/run-<model>.bat       (the exact start command setup wrote)
    3. if something already listens on that port, stop it gracefully first
    4. spawn the server, poll /health until "loaded": true
    5. POST /v1/chat/completions with the prompt (image part optional)
    6. stop the server: SIGTERM/terminate -> the server sends QUIT to its engine child
       and frees VRAM/RAM; force-kill only if it refuses to end

Nodes:
    StrataRun      prompt in, text + reasoning + run metadata out
    StrataInspect  a folder path in, the parsed config / start command out
"""

from __future__ import annotations

import glob
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
          timeout: float, api_key: str = "") -> tuple[str, str, dict]:
    """One non-streamed /v1/chat/completions call. Returns (content, reasoning, usage)."""
    body = {"model": "strata", "messages": messages, "max_tokens": int(max_tokens),
            "reasoning_effort": effort, "stream": False}
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
    """A ComfyUI IMAGE tensor to Strata's OpenAI image_url part."""
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


class StrataRun:
    def __init__(self):
        self.cache = {}

    def INPUT_TYPES(s=None):
        return {
            "required": {
                "folder": ("STRING", {"multiline": False,
                                      "tooltip": "The configured Strata folder (the one "
                                                 "holding strata-<model>.json)."}),
                                "port": ("INT", {"default": 8080, "min": 1, "max": 65535,
                                                  "tooltip": "The port to own. Any Strata server already "
                                                             "listening here is stopped first, then the node "
                                                             "starts its own on this same port."}),
                "prompt": ("STRING", {"multiline": True, "tooltip": "The prompt to execute."}),
                "system": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "Optional system message."}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 131072}),
                # a list as the type is what makes the classic UI draw a dropdown (same
                # as KSampler's sampler_name/scheduler); "choices" on a STRING is ignored
                "thinking": (["config", "none", "low", "medium", "high"],
                             {"default": "config",
                              "tooltip": "Reasoning effort; 'config' keeps whatever the "
                                         "Strata config chose."}),
                "startup_timeout": ("INT", {"default": 900, "min": 30, "max": 7200,
                                            "tooltip": "Seconds to wait for the model to load "
                                                       "(first load is slowest)."}),
                "prompt_timeout": ("INT", {"default": 1800, "min": 60, "max": 7200}),
                "unload_after": ("BOOLEAN", {"default": True,
                                             "tooltip": "Stop the server and free VRAM/RAM when "
                                                        "the prompt finishes."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff,
                                 "tooltip": "Non-zero reuses the cached result instead of "
                                            "loading the model again."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional image for the prompt (needs vision "
                                               "enabled in the Strata config)."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("result", "reasoning", "run info")
    FUNCTION = "execute"
    DESCRIPTION = ("Owns one port: a server already listening there is found in the process "
                   "list and terminated with its whole launcher chain (model unloaded first), "
                   "the engine is loaded from scratch on the same port, the prompt runs, then "
                   "this server is stopped and VRAM/RAM are freed. Nothing is loaded when the "
                   "port could not be freed first - the run fails instead of stacking a second "
                   "model in memory.")
    CATEGORY = "LLM/Strata"
    SIZE_HINTS = ("large",)

    def _run(self, folder, prompt, system, image, max_tokens, effort,
             startup_timeout, prompt_timeout, unload, seed, port):
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
        part = _image_part(image) if image is not None else None
        if part:
            if not cfg.get("vision"):
                raise ValueError("this Strata install has no vision configured")
            messages.append({"role": "user",
                             "content": [part, {"type": "text",
                                                "text": (prompt or "").strip()}]})
        else:
            messages.append({"role": "user", "content": (prompt or "").strip()})
        if not (prompt or "").strip() and not part:
            raise ValueError("the prompt is empty")
        stopped = _stop_existing(port, cfg["_root"])
        t0 = time.time()
        proc = _spawn(cfg, port)
        try:
            info = _wait_loaded(proc, port, float(startup_timeout))
            api_key = str(info.get("api_key") or "")
            content, reasoning, usage = _chat(port, messages, int(max_tokens), effort,
                                              float(prompt_timeout), api_key)
        finally:
            state = (_stop_server(proc, port, 180.0, cfg["_root"]) if unload
                     else "left running")
        meta = {"model": info.get("model"), "port": port, "loaded": True,
            "seconds": round(time.time() - t0, 1), "unload": state, "usage": usage,
            "stopped_existing_server": stopped}
        return (content, reasoning, json.dumps(meta, indent=2))

    def execute(self, folder: str, port: int, prompt: str, system: str = "",
                max_tokens: int = 4096, thinking: str = "config",
                startup_timeout: int = 900, prompt_timeout: int = 1800,
                unload_after: bool = True, seed: int = 0, image=None):
        key = (folder, prompt, system, max_tokens, thinking, image is not None,
               port, startup_timeout, prompt_timeout, unload_after, seed)
        if seed and key in self.cache:
            return self.cache[key]
        out = self._run(folder, prompt, system, image, max_tokens,
                        "" if thinking == "config" else thinking,
                        startup_timeout, prompt_timeout, unload_after, seed, port)
        self.cache = {key: out}
        return out


class StrataInspect:
    def INPUT_TYPES(s=None):
        return {"required": {"folder": ("STRING", {"multiline": False,
                                                   "tooltip": "The configured Strata folder."})}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("config info", "start command")
    FUNCTION = "execute"
    CATEGORY = "LLM/Strata"

    def execute(self, folder: str):
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
        return (json.dumps(info, indent=2), cmd)


NODE_CLASS_MAPPINGS = {
    "StrataRun": StrataRun,
    "StrataInspect": StrataInspect,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "StrataRun": "Strata Run (start / prompt / stop)",
    "StrataInspect": "Strata Config",
}
