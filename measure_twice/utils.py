"""Small shared helpers: logging, shell commands and working directories.

Paths are configurable through environment variables so the GPU stages run the same way on a workstation, a cluster or
a hosted notebook:
  MT_WORK  scratch space for downloads, model checkpoints and temporary files   (default: ./work)
  MT_OUT   where per-scene outputs and logs are written                         (default: ./outputs)
On Kaggle the defaults switch to /kaggle/temp and /kaggle/working/out automatically.
"""
import os, subprocess, time

_ON_KAGGLE = os.path.isdir("/kaggle/working")
WORK = os.environ.get("MT_WORK", "/kaggle/temp" if _ON_KAGGLE and os.path.isdir("/kaggle/temp") else os.path.abspath("work"))
OUT = os.environ.get("MT_OUT", "/kaggle/working/out" if _ON_KAGGLE else os.path.abspath("outputs"))
os.makedirs(WORK, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
_LOG = None


def log(*a):
    """Print a timestamped line and append it to OUT/steps.log."""
    global _LOG
    msg = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(msg, flush=True)
    if _LOG is None:
        _LOG = open(os.path.join(OUT, "steps.log"), "a")
    _LOG.write(msg + "\n"); _LOG.flush()


def sh(cmd, check=True, cwd=None, tail=3000, env=None):
    """Run a shell command and log the tail of its output. Returns (returncode, output)."""
    t = time.time()
    log("$", cmd if len(cmd) < 400 else cmd[:400] + " ...")
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, env=dict(os.environ, **(env or {})))
    out = (r.stdout or "") + (r.stderr or "")
    log(f"  rc={r.returncode} ({time.time() - t:.1f}s)\n" + out[-tail:])
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed rc={r.returncode}: {cmd}")
    return r.returncode, out


def hf_token():
    """Hugging Face token for gated checkpoints: the HF_TOKEN environment variable, if set."""
    return os.environ.get("HF_TOKEN") or None
