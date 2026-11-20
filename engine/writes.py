"""Hand a run's database writes to ArtifactData exactly as the engine made them.

Nov 19, 2026: the daily review sent a research test's writes with a file name it had typed itself, a timestamp
1520 where the engine had written 1519; the save was refused and had to be resent. Ids and file names carry the
minute of the test, so nothing in them may be retyped. Every writes file is checked here (each file exists, and its
name matches its doc id) and printed one chunk per line, ready to pass to ArtifactData "batch" as printed.

    python3 -m engine.writes show <writes.json> [<writes.json> ...]   check, then print each chunk on one line
    python3 -m engine.writes check <writes.json> [...]                check only (exit 1 and say what is wrong)
"""
import json
import pathlib
import sys


def load(path) -> list:
    data = json.loads(pathlib.Path(path).read_text())
    if data and isinstance(data[0], dict):  # a single chunk written flat
        data = [data]
    return [c for c in data if c]


def problems(chunks: list) -> list:
    out = []
    for i, ch in enumerate(chunks, 1):
        for w in ch:
            fp = w.get("file_path")
            if fp is None:
                continue
            p = pathlib.Path(fp)
            if not p.is_file():
                out.append(f"chunk {i}: {w.get('collection')}/{w.get('doc_id')}: file {fp} does not exist")
            elif w.get("doc_id") and p.stem != str(w["doc_id"]):
                out.append(f"chunk {i}: doc_id {w['doc_id']} does not match its file {p.name}")
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2 or argv[0] not in ("show", "check"):
        print(__doc__)
        return 2
    cmd, files = argv[0], argv[1:]
    bad, shown = [], []
    for f in files:
        if not pathlib.Path(f).is_file():
            bad.append(f"{f} does not exist (nothing to write from it)")
            continue
        ch = load(f)
        bad += [f"{f}: {p}" for p in problems(ch)]
        shown.append((f, ch))
    if bad:
        print("WRITES NOT READY: " + " | ".join(bad))
        return 1
    if cmd == "check":
        print("WRITES OK: " + ", ".join(f"{f} ({sum(len(c) for c in ch)} writes)" for f, ch in shown))
        return 0
    for f, ch in shown:
        for i, c in enumerate(ch, 1):
            print(f"CHUNK {i}/{len(ch)} of {f} ({len(c)} writes), pass to ArtifactData batch exactly as printed:")
            print(json.dumps(c, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
