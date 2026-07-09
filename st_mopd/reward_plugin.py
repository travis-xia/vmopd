"""Reward functions for ST-MOPD GRPO baselines."""

import ast
import json
import math
import re
from typing import Any, List, Optional, Sequence, Tuple

try:
    from swift.rewards import ORM, orms
except ModuleNotFoundError:
    class ORM:
        def __init__(self, *args, **kwargs):
            pass

    orms = {}


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
FORMAT_RE = re.compile(r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.DOTALL | re.IGNORECASE)
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


def extract_answer(text: str) -> Optional[str]:
    match = ANSWER_RE.search(text or "")
    if not match:
        return None
    return match.group(1).strip()


def extract_numbers(text: str) -> List[float]:
    values: List[float] = []
    for item in NUMBER_RE.findall(text or ""):
        try:
            value = float(item)
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def parse_temporal_segment(text: str) -> Optional[Tuple[float, float]]:
    answer = extract_answer(text)
    if answer is None:
        return None
    numbers = extract_numbers(answer)
    if len(numbers) < 2:
        return None
    start, end = numbers[0], numbers[1]
    if start < 0 or end <= start:
        return None
    return start, end


def temporal_iou(pred: Tuple[float, float], gt: Tuple[float, float]) -> float:
    pred_start, pred_end = pred
    gt_start, gt_end = gt
    inter = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    if union <= 0:
        return 0.0
    return inter / union


def parse_box_list_from_obj(obj: Any) -> List[List[float]]:
    if not isinstance(obj, list):
        return []
    boxes: List[List[float]] = []
    for item in obj:
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            continue
        try:
            box = [float(v) for v in item]
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(v) for v in box):
            boxes.append(box)
    return boxes


def parse_spatial_boxes(text: str) -> List[List[float]]:
    answer = extract_answer(text)
    if answer is None:
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(answer)
        except Exception:
            continue
        boxes = parse_box_list_from_obj(parsed)
        if boxes:
            return boxes
    numbers = extract_numbers(answer)
    boxes = []
    for idx in range(0, len(numbers) - 3, 4):
        boxes.append(numbers[idx:idx + 4])
    return boxes


def box_iou(pred: Sequence[float], gt: Sequence[float]) -> float:
    px0, py0, px1, py1 = [float(v) for v in pred]
    gx0, gy0, gx1, gy1 = [float(v) for v in gt]
    if px1 <= px0 or py1 <= py0 or gx1 <= gx0 or gy1 <= gy0:
        return 0.0
    inter_w = max(0.0, min(px1, gx1) - max(px0, gx0))
    inter_h = max(0.0, min(py1, gy1) - max(py0, gy0))
    inter = inter_w * inter_h
    pred_area = (px1 - px0) * (py1 - py0)
    gt_area = (gx1 - gx0) * (gy1 - gy0)
    union = pred_area + gt_area - inter
    if union <= 0:
        return 0.0
    return inter / union


def trajectory_average_iou(pred_boxes: Sequence[Sequence[float]], gt_boxes: Sequence[Sequence[float]]) -> float:
    if not gt_boxes:
        return 0.0
    total = 0.0
    for idx, gt_box in enumerate(gt_boxes):
        if idx >= len(pred_boxes):
            continue
        total += box_iou(pred_boxes[idx], gt_box)
    return total / len(gt_boxes)


def item_at(values: Any, index: int, default: Any = None) -> Any:
    if values is None:
        return default
    if isinstance(values, list):
        if index >= len(values):
            return default
        return values[index]
    return values


class STMoPDFormat(ORM):
    """Check the shared RL-family think/answer output protocol."""

    def __call__(self, completions, **kwargs) -> List[float]:
        return [1.0 if FORMAT_RE.match(completion or "") else 0.0 for completion in completions]


class STMoPDTemporalIoU(ORM):
    """Temporal segment IoU reward for temporal-domain prompts."""

    def __call__(self, completions, gt_start=None, gt_end=None, **kwargs) -> List[float]:
        rewards: List[float] = []
        for idx, completion in enumerate(completions):
            pred = parse_temporal_segment(completion)
            start = item_at(gt_start, idx)
            end = item_at(gt_end, idx)
            if pred is None or start is None or end is None:
                rewards.append(0.0)
                continue
            rewards.append(temporal_iou(pred, (float(start), float(end))))
        return rewards


class STMoPDSpatialIoU(ORM):
    """Average trajectory IoU reward for spatial-domain GoT prompts."""

    def __call__(self, completions, gt_boxes=None, **kwargs) -> List[float]:
        rewards: List[float] = []
        for idx, completion in enumerate(completions):
            pred_boxes = parse_spatial_boxes(completion)
            gold_boxes = item_at(gt_boxes, idx, [])
            rewards.append(trajectory_average_iou(pred_boxes, gold_boxes))
        return rewards


class STMoPDTaskIoU(ORM):
    """Route mixed-domain samples to their matching task reward."""

    temporal = STMoPDTemporalIoU()
    spatial = STMoPDSpatialIoU()

    def __call__(self, completions, data_type=None, gt_start=None, gt_end=None, gt_boxes=None, **kwargs) -> List[float]:
        rewards: List[float] = []
        for idx, completion in enumerate(completions):
            domain = item_at(data_type, idx, "")
            if domain == "temporal":
                pred = parse_temporal_segment(completion)
                start = item_at(gt_start, idx)
                end = item_at(gt_end, idx)
                reward = 0.0 if pred is None or start is None or end is None else temporal_iou(
                    pred, (float(start), float(end)))
            elif domain == "spatial":
                pred_boxes = parse_spatial_boxes(completion)
                reward = trajectory_average_iou(pred_boxes, item_at(gt_boxes, idx, []))
            else:
                reward = 0.0
            rewards.append(reward)
        return rewards


orms["st_mopd_format"] = STMoPDFormat
orms["st_mopd_temporal_iou"] = STMoPDTemporalIoU
orms["st_mopd_spatial_iou"] = STMoPDSpatialIoU
orms["st_mopd_task_iou"] = STMoPDTaskIoU
