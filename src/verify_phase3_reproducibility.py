from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_paths(project: Path) -> list[Path]:
    paths = [project / "phase3_audit" / "line_samples.jsonl"]
    paths.extend(sorted((project / "results").glob("phase3_*.csv")))
    paths.append(project / "results" / "phase3_summary.json")
    paths.extend(sorted((project / "phase3_masked_images").glob("*.png")))
    paths.extend(sorted((project / "phase3_debug_overlays").glob("*.png")))
    paths.extend(sorted((project / "examples" / "phase3_debug_overlays").glob("*")))
    return [path for path in paths if path.is_file()]


def snapshot(project: Path) -> dict[str, str]:
    return {str(path.relative_to(project)): sha256(path) for path in artifact_paths(project)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--mode", choices=["snapshot", "compare"], required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    snapshot_path = project / "phase3_audit" / "reproducibility_snapshot.json"
    result_path = project / "results" / "phase3_reproducibility.json"
    current = snapshot(project)
    if args.mode == "snapshot":
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"mode": "snapshot", "artifact_count": len(current), "snapshot": str(snapshot_path)}))
        return
    expected = json.loads(snapshot_path.read_text(encoding="utf-8"))
    missing = sorted(set(expected) - set(current))
    extra = sorted(set(current) - set(expected))
    mismatched = sorted(path for path in set(expected) & set(current) if expected[path] != current[path])
    result = {
        "classification": "deterministic_cpu_pipeline",
        "verdict": "REPRODUCIBLE" if not missing and not extra and not mismatched else "NOT_REPRODUCIBLE",
        "exact_hash_match": not missing and not extra and not mismatched,
        "artifact_count_before": len(expected),
        "artifact_count_after": len(current),
        "missing": missing,
        "extra": extra,
        "mismatched": mismatched,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    if result["verdict"] != "REPRODUCIBLE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
