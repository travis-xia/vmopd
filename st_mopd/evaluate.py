#!/usr/bin/env python3
"""Build and score ST-MOPD evaluation files.

This script intentionally keeps model inference separate from metric scoring:

1. build: create SWIFT-compatible eval jsonl files from local annotations.
2. score: read a SWIFT infer result jsonl and compute task metrics.
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from st_mopd.build_dataset import (
        DEFAULT_ACTIVITYNET_ROOT,
        DEFAULT_CHARADES_ROOT,
        DEFAULT_GOT_ROOT,
        limited,
        make_spatial_rows,
        make_temporal_rows,
        read_json,
        shuffled,
        write_jsonl,
    )
    from st_mopd.reward_plugin import (
        FORMAT_RE,
        box_iou,
        parse_spatial_boxes,
        parse_temporal_segment,
        temporal_iou,
        trajectory_average_iou,
    )
except ModuleNotFoundError:
    from build_dataset import (  # type: ignore
        DEFAULT_ACTIVITYNET_ROOT,
        DEFAULT_CHARADES_ROOT,
        DEFAULT_GOT_ROOT,
        limited,
        make_spatial_rows,
        make_temporal_rows,
        read_json,
        shuffled,
        write_jsonl,
    )
    from reward_plugin import (  # type: ignore
        FORMAT_RE,
        box_iou,
        parse_spatial_boxes,
        parse_temporal_segment,
        temporal_iou,
        trajectory_average_iou,
    )


DEFAULT_THRESHOLDS = (0.3, 0.5, 0.7)
RESPONSE_FIELDS = ("response", "completion", "prediction", "pred", "output", "generated_text")
REFERENCE_FIELDS = ("labels", "solution", "reference", "ground_truth")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build ST-MOPD eval jsonl files from annotations.")
    build.add_argument("--annotation-root", type=Path, default=Path("Videochat-R1/annotations"))
    build.add_argument("--output-dir", type=Path, default=Path("data/st_mopd_eval"))
    build.add_argument("--charades-anno-file", type=Path, default=None)
    build.add_argument("--activitynet-anno-file", type=Path, default=None)
    build.add_argument("--got-anno-file", type=Path, default=None)
    build.add_argument("--charades-video-root", default=DEFAULT_CHARADES_ROOT)
    build.add_argument("--activitynet-video-root", default=DEFAULT_ACTIVITYNET_ROOT)
    build.add_argument("--got-root", default=DEFAULT_GOT_ROOT)
    build.add_argument("--charades-split", default="val")
    build.add_argument("--activitynet-split", default="val")
    build.add_argument("--got-split", default="val")
    build.add_argument(
        "--temporal-sources",
        nargs="+",
        default=["charades", "activitynet"],
        choices=["charades", "activitynet"],
    )
    build.add_argument("--charades-ext", default=".mp4")
    build.add_argument("--activitynet-ext", default=".mp4")
    build.add_argument("--max-temporal-samples", type=int, default=None)
    build.add_argument("--max-spatial-samples", type=int, default=None)
    build.add_argument("--max-samples-per-source", type=int, default=None)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--strict-media", action="store_true")
    build.add_argument("--pretty", action="store_true")
    build.set_defaults(func=build_eval_files)

    score = subparsers.add_parser("score", help="Score ST-MOPD predictions.")
    score.add_argument("--predictions", type=Path, nargs="+", required=True)
    score.add_argument("--gold-jsonl", type=Path, default=None)
    score.add_argument("--output", type=Path, default=None)
    score.add_argument("--per-sample-output", type=Path, default=None)
    score.add_argument("--response-field", default=None)
    score.add_argument("--id-field", default="sample_id")
    score.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    score.add_argument("--strict", action="store_true")
    score.add_argument("--pretty", action="store_true")
    score.set_defaults(func=score_predictions)
    return parser


def build_eval_files(args: argparse.Namespace) -> Dict[str, Any]:
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    temporal_rows: List[Dict[str, Any]] = []
    per_source_counts: Dict[str, int] = {}
    if "charades" in args.temporal_sources:
        path = args.charades_anno_file or (
            args.annotation_root / "Charades" / "charades_annotation" / f"{args.charades_split}.json"
        )
        rows = make_temporal_rows(
            "charades_sta",
            read_json(path),
            Path(args.charades_video_root),
            args.charades_ext,
            args.strict_media,
        )
        rows = limited(rows, args.max_samples_per_source, rng)
        per_source_counts["charades_sta"] = len(rows)
        temporal_rows.extend(rows)

    if "activitynet" in args.temporal_sources:
        path = args.activitynet_anno_file or (
            args.annotation_root / "ActivityNet" / "activitynet_annotation" / f"{args.activitynet_split}.json"
        )
        rows = make_temporal_rows(
            "activitynet_grounding",
            read_json(path),
            Path(args.activitynet_video_root),
            args.activitynet_ext,
            args.strict_media,
        )
        rows = limited(rows, args.max_samples_per_source, rng)
        per_source_counts["activitynet_grounding"] = len(rows)
        temporal_rows.extend(rows)

    got_path = args.got_anno_file or (args.annotation_root / "Got" / f"got_{args.got_split}.json")
    spatial_rows = make_spatial_rows(read_json(got_path), Path(args.got_root), args.strict_media)

    temporal_rows = limited(temporal_rows, args.max_temporal_samples, rng)
    spatial_rows = limited(spatial_rows, args.max_spatial_samples, rng)
    mixed_rows = shuffled(temporal_rows + spatial_rows, rng)

    outputs = {
        "temporal_eval": temporal_rows,
        "spatial_eval": spatial_rows,
        "mixed_eval": mixed_rows,
    }
    counts = {name: write_jsonl(args.output_dir / f"{name}.jsonl", rows) for name, rows in outputs.items()}
    manifest = {
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "splits": {
            "charades": args.charades_split,
            "activitynet": args.activitynet_split,
            "got": args.got_split,
        },
        "temporal_sources": args.temporal_sources,
        "per_source_counts": per_source_counts,
        "counts": counts,
        "paths": {name: str(args.output_dir / f"{name}.jsonl") for name in outputs},
        "notes": [
            "Eval files keep assistant labels so `swift infer` can remove them and store labels.",
            "Metric scoring uses the retained gt_start/gt_end and gt_boxes fields.",
        ],
    }
    write_json(args.output_dir / "manifest.json", manifest, pretty=args.pretty)
    print(json.dumps({"counts": counts, "paths": manifest["paths"]}, ensure_ascii=False, indent=2))
    return manifest


def write_json(path: Path, data: Any, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2 if pretty else None)
        f.write("\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object.")
            rows.append(value)
    return rows


def read_prediction_file(path: Path, id_field: str) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)

    data = read_json(path)
    if isinstance(data, list):
        return [dict(item) for item in data]
    if isinstance(data, dict):
        if isinstance(data.get("predictions"), list):
            return [dict(item) for item in data["predictions"]]
        rows = []
        for key, value in data.items():
            if isinstance(value, dict):
                row = dict(value)
                row.setdefault(id_field, key)
            else:
                row = {id_field: key, "response": value}
            rows.append(row)
        return rows
    raise ValueError(f"Unsupported prediction file format: {path}")


def load_rows(paths: Sequence[Path], id_field: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        rows.extend(read_prediction_file(path, id_field))
    return rows


def merge_gold(rows: List[Dict[str, Any]], gold_path: Optional[Path], id_field: str) -> List[Dict[str, Any]]:
    if gold_path is None:
        return rows
    gold_rows = read_jsonl(gold_path)
    gold_by_id = {row.get(id_field): row for row in gold_rows if row.get(id_field) is not None}
    # Older swift infer outputs may omit IDs after preprocessing removes metadata.
    positional_fallback = len(rows) == len(gold_rows) and all(not row.get(id_field) for row in rows)
    merged = []
    for index, row in enumerate(rows):
        sample_id = row.get(id_field)
        gold = gold_by_id.get(sample_id)
        if gold is None and positional_fallback:
            gold = gold_rows[index]
        if gold is None:
            gold = {}
        merged_row = dict(gold)
        # Do not let metadata stripped by preprocessing (`None`) erase gold fields.
        merged_row.update({key: value for key, value in row.items() if value is not None})
        merged.append(merged_row)
    return merged


def nested_get(data: Dict[str, Any], path: Sequence[Any]) -> Any:
    value: Any = data
    for part in path:
        if isinstance(part, int):
            if not isinstance(value, list) or part >= len(value):
                return None
            value = value[part]
        else:
            if not isinstance(value, dict):
                return None
            value = value.get(part)
        if value is None:
            return None
    return value


def extract_response(row: Dict[str, Any], response_field: Optional[str]) -> str:
    if response_field:
        value = row.get(response_field)
        return "" if value is None else str(value)

    for field in RESPONSE_FIELDS:
        if row.get(field) is not None:
            return str(row[field])

    choice_content = nested_get(row, ("choices", 0, "message", "content"))
    if choice_content is not None:
        return str(choice_content)

    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return str(message.get("content", ""))
    return ""


def extract_reference(row: Dict[str, Any]) -> str:
    for field in REFERENCE_FIELDS:
        value = row.get(field)
        if value is not None:
            return str(value)
    return ""


def infer_data_type(row: Dict[str, Any]) -> Optional[str]:
    data_type = row.get("data_type")
    if data_type in {"temporal", "spatial"}:
        return str(data_type)
    if row.get("gt_start") is not None and row.get("gt_end") is not None:
        return "temporal"
    if row.get("gt_boxes") is not None or row.get("gt") is not None:
        return "spatial"
    # Swift infer keeps the original assistant target in `labels`.
    reference = extract_reference(row)
    if parse_spatial_boxes(reference):
        return "spatial"
    if parse_temporal_segment(reference) is not None:
        return "temporal"
    return None


def parse_gt_boxes(row: Dict[str, Any]) -> List[List[float]]:
    value = row.get("gt_boxes", row.get("gt", []))
    boxes: List[List[float]] = []
    if not isinstance(value, list):
        return boxes
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            continue
        try:
            box = [float(v) for v in item]
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(v) for v in box):
            boxes.append(box)
    return boxes


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def score_temporal(row: Dict[str, Any], response: str) -> Dict[str, Any]:
    if row.get("gt_start") is not None and row.get("gt_end") is not None:
        gt = (float(row["gt_start"]), float(row["gt_end"]))
    else:
        gt = parse_temporal_segment(extract_reference(row))
        if gt is None:
            return {"skip_reason": "missing_temporal_gt"}
    pred = parse_temporal_segment(response)
    iou = 0.0 if pred is None else temporal_iou(pred, gt)
    return {
        "valid_prediction": pred is not None,
        "task_iou": iou,
        "temporal_iou": iou,
        "pred_start": None if pred is None else pred[0],
        "pred_end": None if pred is None else pred[1],
        "gt_start": gt[0],
        "gt_end": gt[1],
    }


def per_frame_ious(pred_boxes: Sequence[Sequence[float]], gt_boxes: Sequence[Sequence[float]]) -> List[float]:
    ious = []
    for idx, gt_box in enumerate(gt_boxes):
        if idx >= len(pred_boxes):
            ious.append(0.0)
        else:
            ious.append(box_iou(pred_boxes[idx], gt_box))
    return ious


def score_spatial(row: Dict[str, Any], response: str) -> Dict[str, Any]:
    gt_boxes = parse_gt_boxes(row)
    if not gt_boxes:
        gt_boxes = parse_spatial_boxes(extract_reference(row))
    if not gt_boxes:
        return {"skip_reason": "missing_spatial_gt"}
    pred_boxes = parse_spatial_boxes(response)
    frame_ious = per_frame_ious(pred_boxes, gt_boxes)
    traj_iou = trajectory_average_iou(pred_boxes, gt_boxes)
    return {
        "valid_prediction": len(pred_boxes) > 0,
        "task_iou": traj_iou,
        "trajectory_iou": traj_iou,
        "frame_iou": mean(frame_ious),
        "pred_box_count": len(pred_boxes),
        "gt_box_count": len(gt_boxes),
    }


def score_row(row: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    response = extract_response(row, args.response_field)
    data_type = infer_data_type(row)
    base = {
        "sample_id": row.get(args.id_field),
        "data_type": data_type,
        "source": row.get("source"),
        "format_ok": bool(FORMAT_RE.match(response or "")),
        "response": response,
    }
    if data_type == "temporal":
        base.update(score_temporal(row, response))
    elif data_type == "spatial":
        base.update(score_spatial(row, response))
    else:
        base["skip_reason"] = "unknown_data_type"
    return base


def summarize(samples: List[Dict[str, Any]], thresholds: Sequence[float]) -> Dict[str, Any]:
    evaluated = [item for item in samples if "skip_reason" not in item]
    skipped = [item for item in samples if "skip_reason" in item]
    summary: Dict[str, Any] = {
        "count": len(evaluated),
        "skipped": len(skipped),
    }
    if not evaluated:
        return summary

    task_ious = [float(item.get("task_iou", 0.0)) for item in evaluated]
    summary.update({
        "format_acc": mean(1.0 if item.get("format_ok") else 0.0 for item in evaluated),
        "valid_acc": mean(1.0 if item.get("valid_prediction") else 0.0 for item in evaluated),
        "mean_iou": mean(task_ious),
    })
    for threshold in thresholds:
        summary[f"r@{threshold:g}"] = mean(1.0 if value >= threshold else 0.0 for value in task_ious)

    temporal_items = [item for item in evaluated if item.get("data_type") == "temporal"]
    if temporal_items:
        summary["temporal_mean_iou"] = mean(float(item.get("temporal_iou", 0.0)) for item in temporal_items)
    spatial_items = [item for item in evaluated if item.get("data_type") == "spatial"]
    if spatial_items:
        summary["spatial_trajectory_iou"] = mean(float(item.get("trajectory_iou", 0.0)) for item in spatial_items)
        summary["spatial_frame_iou"] = mean(float(item.get("frame_iou", 0.0)) for item in spatial_items)
    return summary


def grouped_summary(samples: List[Dict[str, Any]], key: str, thresholds: Sequence[float]) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in samples:
        value = item.get(key)
        groups[str(value if value is not None else "unknown")].append(item)
    return {name: summarize(items, thresholds) for name, items in sorted(groups.items())}


def score_predictions(args: argparse.Namespace) -> Dict[str, Any]:
    rows = load_rows(args.predictions, args.id_field)
    rows = merge_gold(rows, args.gold_jsonl, args.id_field)
    samples = [score_row(row, args) for row in rows]

    skipped_reasons = Counter(item.get("skip_reason") for item in samples if item.get("skip_reason"))
    if args.strict and skipped_reasons:
        raise ValueError(f"Skipped samples in strict mode: {dict(skipped_reasons)}")

    report = {
        "overall": summarize(samples, args.thresholds),
        "by_data_type": grouped_summary(samples, "data_type", args.thresholds),
        "by_source": grouped_summary(samples, "source", args.thresholds),
        "skipped_reasons": dict(skipped_reasons),
        "prediction_files": [str(path) for path in args.predictions],
        "gold_jsonl": None if args.gold_jsonl is None else str(args.gold_jsonl),
    }

    if args.per_sample_output:
        args.per_sample_output.parent.mkdir(parents=True, exist_ok=True)
        with args.per_sample_output.open("w", encoding="utf-8") as f:
            for item in samples:
                f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

    if args.output:
        write_json(args.output, report, pretty=args.pretty)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
