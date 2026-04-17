import json
import sys
from pathlib import Path

CATEGORIES_FILE = Path("difficulty_categories_all.json")

DIFFICULTY_MAPPING = {
    "<15 min fix": "easy",
    "15 min - 1 hour": "medium",
    "1-4 hours": "hard",
    ">4 hours": "very_hard",
}


def build_categories() -> None:
    from datasets import load_dataset

    data = load_dataset("SWE-bench/SWE-bench_Full", split="test")

    categories: dict[str, list[str]] = {v: [] for v in DIFFICULTY_MAPPING.values()}

    for item in data:
        bucket = DIFFICULTY_MAPPING.get(item["difficulty"])
        if bucket:
            categories[bucket].append(item["instance_id"])

    CATEGORIES_FILE.write_text(json.dumps(categories, indent=2))

    for bucket, ids in categories.items():
        print(f"{bucket}: {len(ids)} instances")
    print(f"\nSaved to {CATEGORIES_FILE}")


def classify_solved(solved: list[str]) -> None:
    if not CATEGORIES_FILE.exists():
        print(f"{CATEGORIES_FILE} not found. Run `python test.py build` first.")
        sys.exit(1)

    categories: dict[str, list[str]] = json.loads(CATEGORIES_FILE.read_text())

    id_to_bucket: dict[str, str] = {}
    for bucket, ids in categories.items():
        for iid in ids:
            id_to_bucket[iid] = bucket

    grouped: dict[str, list[str]] = {v: [] for v in DIFFICULTY_MAPPING.values()}
    unknown: list[str] = []

    for iid in solved:
        bucket = id_to_bucket.get(iid)
        if bucket:
            grouped[bucket].append(iid)
        else:
            unknown.append(iid)

    print("=== Solved issues by difficulty ===\n")
    for bucket in DIFFICULTY_MAPPING.values():
        ids = grouped[bucket]
        print(f"[{bucket}]  ({len(ids)} solved)")
        for iid in ids:
            print(f"  {iid}")
        print()

    if unknown:
        print(f"[unknown / not in dataset]  ({len(unknown)})")
        for iid in unknown:
            print(f"  {iid}")


def classify_resolved_from_result_json(
    result_json: "str | Path | dict | list",
    output_file: "str | Path | None" = None,
) -> dict:
    if not CATEGORIES_FILE.exists():
        print(f"{CATEGORIES_FILE} not found. Run `python difficulty_categorization.py build` first.")
        sys.exit(1)

    if isinstance(result_json, (str, Path)):
        data = json.loads(Path(result_json).read_text())
    else:
        data = result_json

    resolved_ids: list[str] = []

    if isinstance(data, dict):

        for instance_id, record in data.items():
            if not isinstance(record, dict):
                continue

            if record.get("resolved", True):
                resolved_ids.append(instance_id)
    else:
        for record in data:
            if not isinstance(record, dict):
                continue
            instance_id = record.get("instance_id")
            if not instance_id:
                continue
            if record.get("resolved", True):
                resolved_ids.append(instance_id)

    categories: dict[str, list[str]] = json.loads(CATEGORIES_FILE.read_text())
    id_to_bucket: dict[str, str] = {
        iid: bucket for bucket, ids in categories.items() for iid in ids
    }

    classified: dict[str, list[str]] = {v: [] for v in DIFFICULTY_MAPPING.values()}
    unknown: list[str] = []

    for iid in resolved_ids:
        bucket = id_to_bucket.get(iid)
        if bucket:
            classified[bucket].append(iid)
        else:
            unknown.append(iid)

    result: dict = {**classified, "unknown": unknown}

    if output_file:
        Path(output_file).write_text(json.dumps(result, indent=2))
        print(f"Classified results saved to {output_file}")

    return result


if __name__ == "__main__":
    build_categories()
    # result = classify_resolved_from_result_json("claude4.5_verified.json", output_file="classified.json")
  
