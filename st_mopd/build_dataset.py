#!/usr/bin/env python3
"""Build ST-MOPD SFT and GRPO jsonl datasets.

The first experiment uses two capability domains:
- temporal: text query to temporal segment.
- spatial: GoT-style video tracking to normalized box trajectory.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


RL_SYSTEM_PROMPT = (
    "A conversation between user and assistant. The user provides a visual input "
    "and asks a question. The assistant MUST first think about the reasoning "
    "process in the mind and then provide the answer. The reasoning process and "
    "answer are enclosed within <think> </think> and <answer> </answer> tags, "
    "respectively."
)

DEFAULT_CHARADES_ROOT = (
    "/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/"
    "hf_download/VideoAuto-R1-Data/CharadesSTA/Charades_v1_480"
)
DEFAULT_ACTIVITYNET_ROOT = (
    "/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/"
    "TVG-R1/video-r1/data/GroundedVLLM/activitynet/videos"
)
DEFAULT_GOT_ROOT = (
    "/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/"
    "hf_download/GoT-10k"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-root", type=Path, default=Path("Videochat-R1/annotations"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/st_mopd"))
    parser.add_argument("--charades-anno-file", type=Path, default=None)
    parser.add_argument("--activitynet-anno-file", type=Path, default=None)
    parser.add_argument("--got-anno-file", type=Path, default=None)
    parser.add_argument("--charades-video-root", default=DEFAULT_CHARADES_ROOT)
    parser.add_argument("--activitynet-video-root", default=DEFAULT_ACTIVITYNET_ROOT)
    parser.add_argument("--got-root", default=DEFAULT_GOT_ROOT)
    parser.add_argument("--charades-split", default="train")
    parser.add_argument("--activitynet-split", default="train")
    parser.add_argument("--got-split", default="train")
    parser.add_argument("--temporal-sources", nargs="+", default=["charades", "activitynet"],
                        choices=["charades", "activitynet"])
    parser.add_argument("--charades-ext", default=".mp4")
    parser.add_argument("--activitynet-ext", default=".mp4")
    parser.add_argument("--max-temporal-samples", type=int, default=None)
    parser.add_argument("--max-spatial-samples", type=int, default=None)
    parser.add_argument("--max-samples-per-source", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-media", action="store_true",
                        help="Fail if any referenced video/frame path is missing.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print manifest json.")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def format_seconds(value: float) -> str:
    return f"{float(value):.2f}"


def round_box(box: Sequence[float]) -> List[float]:
    return [round(float(v), 3) for v in box]


def format_box(box: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(v):.3f}" for v in box) + "]"


def format_boxes(boxes: Sequence[Sequence[float]]) -> str:
    rounded = [round_box(box) for box in boxes]
    return json.dumps(rounded, ensure_ascii=False, separators=(", ", ": "))


def temporal_prompt(sentence: str) -> str:
    return (
        "<video>\n"
        f'To accurately pinpoint the event "{sentence}" in the video, determine '
        "the precise time period of the event.\n\n"
        "Provide the start and end times (in seconds, precise to two decimal places) "
        'in the format "start time to end time" within the <answer> </answer> tags. '
        'For example: "12.54 to 17.83".'
    )


def spatial_prompt(object_name: str, initial_box: Sequence[float]) -> str:
    return (
        "<video>\n"
        f'Track the "{object_name}" in the video based on its initial coordinates '
        f'"{format_box(initial_box)}". The output should be a list containing eight '
        "sublists. Each sublist includes four normalized coordinates [x0, y0, x1, y1] "
        "representing the bounding box of the object at specific time intervals.\n\n"
        "Provide the final trajectory within the <answer> </answer> tags."
    )


def sft_messages(user_content: str, answer: str) -> List[Dict[str, Any]]:
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer, "loss": True},
    ]


def rl_messages(user_content: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": RL_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def media_exists(media: Any) -> bool:
    if isinstance(media, list):
        return all(media_exists(item) for item in media)
    return Path(media).exists()


def require_media(row: Dict[str, Any], strict: bool) -> None:
    if not strict:
        return
    for video in row.get("videos", []):
        if not media_exists(video):
            raise FileNotFoundError(f"Missing media for sample {row.get('sample_id')}: {video}")


def make_temporal_rows(
    source: str,
    annotation: Dict[str, Any],
    video_root: Path,
    video_ext: str,
    strict_media: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for video_id, item in annotation.items():
        duration = float(item["duration"])
        timestamps = item.get("timestamps", [])
        sentences = item.get("sentences", [])
        video_path = str(video_root / f"{video_id}{video_ext}")
        for idx, (timestamp, sentence) in enumerate(zip(timestamps, sentences)):
            start, end = float(timestamp[0]), float(timestamp[1])
            answer = f"<answer>{format_seconds(start)} to {format_seconds(end)}</answer>"
            prompt = temporal_prompt(str(sentence).strip())
            row = {
                "data_type": "temporal",
                "source": source,
                "sample_id": f"{source}:{video_id}:{idx}",
                "video_id": video_id,
                "duration": duration,
                "query": str(sentence).strip(),
                "gt_start": round(start, 4),
                "gt_end": round(end, 4),
                "solution": answer,
                "messages": sft_messages(prompt, answer),
                "videos": [video_path],
            }
            require_media(row, strict_media)
            rows.append(row)
    return rows


def select_got_frame_indices(num_frames: int, num_boxes: int = 8) -> List[int]:
    if num_frames <= 0:
        return []
    if num_boxes <= 1:
        return [0]
    if num_frames <= num_boxes:
        indices = list(range(num_frames))
        indices.extend([num_frames - 1] * (num_boxes - len(indices)))
        return indices

    middle_count = max(num_boxes - 2, 0)
    if middle_count == 0:
        return [0, num_frames - 1]
    middle_start = 1
    middle_end = max(num_frames - 2, middle_start)
    if middle_count == 1:
        middle = [middle_start]
    else:
        step = (middle_end - middle_start) / (middle_count - 1)
        middle = [round(middle_start + i * step) for i in range(middle_count)]
    return [0] + middle + [num_frames - 1]


def normalize_got_sequence_path(got_root: Path, relative_path: str) -> Path:
    rel = Path(relative_path)
    parts = rel.parts
    if got_root.name.lower() in {"got-10k", "got10k"} and parts and parts[0].lower() == "got-10k":
        rel = Path(*parts[1:])
    return got_root / rel


def list_sequence_frames(sequence_dir: Path) -> List[Path]:
    if not sequence_dir.exists():
        return []
    frames: List[Path] = []
    for suffix in ("*.jpg", "*.jpeg", "*.png"):
        frames.extend(sequence_dir.glob(suffix))
    return sorted(frames)


def fallback_got_frames(sequence_dir: Path, count: int) -> List[Path]:
    return [sequence_dir / f"{idx:08d}.jpg" for idx in range(1, count + 1)]


def make_spatial_rows(annotation: List[Dict[str, Any]], got_root: Path, strict_media: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, item in enumerate(annotation):
        boxes = [round_box(box) for box in item["gt"]]
        initial_box = boxes[0]
        sequence_dir = normalize_got_sequence_path(got_root, item["path"])
        all_frames = list_sequence_frames(sequence_dir)
        media_missing = not all_frames
        if media_missing:
            sampled_frames = fallback_got_frames(sequence_dir, len(boxes))
        else:
            indices = select_got_frame_indices(len(all_frames), len(boxes))
            sampled_frames = [all_frames[i] for i in indices]
        frame_paths = [str(path) for path in sampled_frames]
        answer = f"<answer>{format_boxes(boxes)}</answer>"
        prompt = spatial_prompt(str(item["object"]).strip(), initial_box)
        row = {
            "data_type": "spatial",
            "source": "got10k",
            "sample_id": f"got10k:{index:06d}",
            "sequence_path": str(sequence_dir),
            "object": str(item["object"]).strip(),
            "initial_box": initial_box,
            "gt_boxes": boxes,
            "solution": answer,
            "media_missing": media_missing,
            "messages": sft_messages(prompt, answer),
            "videos": [frame_paths],
        }
        require_media(row, strict_media)
        rows.append(row)
    return rows


def to_rl_row(row: Dict[str, Any]) -> Dict[str, Any]:
    rl_row = dict(row)
    user_message = row["messages"][0]
    rl_row["messages"] = rl_messages(user_message["content"])
    return rl_row


def limited(rows: List[Dict[str, Any]], limit: Optional[int], rng: random.Random) -> List[Dict[str, Any]]:
    if limit is None or limit >= len(rows):
        return rows
    sampled = rows[:]
    rng.shuffle(sampled)
    return sampled[:limit]


def shuffled(rows: List[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    copied = rows[:]
    rng.shuffle(copied)
    return copied


def build(args: argparse.Namespace) -> Dict[str, Any]:
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    temporal_rows: List[Dict[str, Any]] = []
    per_source_counts: Dict[str, int] = {}
    if "charades" in args.temporal_sources:
        path = args.charades_anno_file or (
            args.annotation_root / "Charades" / "charades_annotation" / f"{args.charades_split}.json")
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
            args.annotation_root / "ActivityNet" / "activitynet_annotation" / f"{args.activitynet_split}.json")
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
    mixed_sft = shuffled(temporal_rows + spatial_rows, rng)
    mixed_rl = [to_rl_row(row) for row in mixed_sft]

    outputs = {
        "temporal_sft": temporal_rows,
        "spatial_sft": spatial_rows,
        "mixed_sft": mixed_sft,
        "temporal_rl": [to_rl_row(row) for row in temporal_rows],
        "spatial_rl": [to_rl_row(row) for row in spatial_rows],
        "mixed_rl": mixed_rl,
    }

    counts: Dict[str, int] = {}
    for name, rows in outputs.items():
        counts[name] = write_jsonl(args.output_dir / f"{name}.jsonl", rows)

    manifest = {
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "rl_system_prompt": RL_SYSTEM_PROMPT,
        "temporal_sources": args.temporal_sources,
        "per_source_counts": per_source_counts,
        "counts": counts,
        "paths": {name: str(args.output_dir / f"{name}.jsonl") for name in outputs},
        "notes": [
            "SFT files include assistant answers and no system prompt.",
            "RL files include the shared RL system prompt and no assistant answer.",
            "Spatial rows use videos=[list_of_8_frame_paths]. Build on the cluster for real GoT frame paths.",
        ],
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        indent = 2 if args.pretty else None
        json.dump(manifest, f, ensure_ascii=False, indent=indent)
        f.write("\n")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build(args)
    print(json.dumps({"counts": manifest["counts"], "paths": manifest["paths"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
