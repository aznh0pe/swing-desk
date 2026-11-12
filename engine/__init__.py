"""Swing Desk engine.

Every scheduled run starts on a fresh cloud machine, and the set of Python packages on those machines varies
(on Nov 11, 2026 the daily review's machine had no pyarrow, which the research step needs to read the parquet
price history). So the engine makes sure its own packages are present before anything else runs: importing
`engine` installs whatever is missing, once, as routine setup. A machine that cannot install them (no network)
fails loudly at the import instead, which the runs report as a problem.
"""
import importlib
import importlib.util as _ilu
import os as _os
import subprocess as _sp
import sys as _sys
import tempfile as _tf

# import name -> pip package
REQUIRED = {"pandas": "pandas", "numpy": "numpy", "sklearn": "scikit-learn", "pyarrow": "pyarrow"}
_FAILED_MARK = _os.path.join(_tf.gettempdir(), ".swing_desk_deps_failed")


def missing_packages() -> list[str]:
    return [pkg for mod, pkg in REQUIRED.items() if _ilu.find_spec(mod) is None]


def ensure_deps(log=None) -> list[str]:
    """Install any package the engine needs that this machine lacks; returns the packages installed. Tries once
    per machine when an install fails, so a machine without network doesn't retry in every process."""
    missing = missing_packages()
    if not missing or _os.environ.get("SWING_DESK_NO_AUTO_INSTALL") or _os.path.exists(_FAILED_MARK):
        return []
    base = [_sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check"]
    for extra in (["--break-system-packages"], []):
        try:
            r = _sp.run(base + extra + missing, capture_output=True, text=True, timeout=300)
        except Exception:  # pip missing or timed out: try the next form, then give up
            continue
        if r.returncode == 0:
            break
    importlib.invalidate_caches()
    still = missing_packages()
    done = [p for p in missing if p not in still]
    msg = (f"engine setup: installed {', '.join(done)}, which this fresh machine lacked. This is routine setup, not a "
           "run problem." if done else "")
    if still:
        try:
            open(_FAILED_MARK, "w").write(" ".join(still))
        except OSError:
            pass
        msg = (msg + " " if msg else "") + f"engine setup: could not install {', '.join(still)} (pip failed); imports will fail."
    (log or (lambda m: print(m, file=_sys.stderr)))(msg)
    return done


ensure_deps()
