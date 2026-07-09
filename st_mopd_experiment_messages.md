# ST-MOPD 实验中的 messages 结构

这里先统一一件事：SFT 和 RL-family 实验承担的角色不同，因此 messages 不必强行一致。SFT 是独立对照组，目标是让模型学会任务格式和可验证答案；它不需要 `<think>`，也不需要额外 system prompt。RL、Mixed-RL 和 MOPD 才是主要比较对象，它们应该共享同一个 system prompt，把跨任务的思考格式协议放在 system 里，而不是在每个 user prompt 里重复。

当前第一版只保留两个能力域：`temporal` 和 `spatial`。GoT 在我们的使用方式里直接归入 `spatial`，因为它虽然输入是视频帧序列、输出是跨帧轨迹，但 reward 验证的仍然是目标在视觉空间中的框位置是否正确。路由时统一使用 `spatial`。

推荐的 RL-family system prompt 是：

```python
RL_SYSTEM_PROMPT = (
    "A conversation between user and assistant. The user provides a visual input "
    "and asks a question. The assistant MUST first think about the reasoning "
    "process in the mind and then provide the answer. The reasoning process and "
    "answer are enclosed within <think> </think> and <answer> </answer> tags, "
    "respectively."
)
```

这样设计的好处是，`<think>` 是所有 RL 能力域共享的输出协议，而不是某个任务 prompt 的一部分。Temporal 和 Spatial 的 user prompt 只负责描述任务和最终答案格式；格式纪律由 system prompt 统一约束。Separate RL、Mixed-RL 和 MOPD 必须使用完全相同的 system prompt，否则 prompt 差异会混进优化范式差异里。

## 两个能力域的基本输出

Temporal Teacher 学时间定位，输入是视频和事件描述，输出一个时间段：

```python
assistant_content = "<answer>14.20 to 22.50</answer>"
```

Spatial Teacher 在第一版里学 GoT tracking。输入是视频或 8 帧序列、对象名和初始框，输出 8 个归一化框：

```python
assistant_content = "<answer>[[0.58, 0.50, 0.64, 0.60], [0.58, 0.50, 0.64, 0.60], [0.53, 0.48, 0.57, 0.56], [0.70, 0.38, 0.76, 0.47], [0.56, 0.50, 0.61, 0.59], [0.49, 0.59, 0.54, 0.69], [0.34, 0.51, 0.38, 0.62], [0.32, 0.48, 0.33, 0.56]]</answer>"
```

这两个输出字段就是两个能力域的边界。Temporal 的 reward 看 temporal IoU，Spatial 的 reward 看 trajectory average IoU / GoT overlap / per-frame box IoU。是否写 `<think>` 不是能力域定义的核心；核心是最终可验证字段。

后续如果加入 COCO / RefCOCO，它们应该作为 `spatial` 下的静态 grounding 子任务。任务子类型可以不同，MOPD 路由的 capability domain 仍然是 `spatial`。

## Separate SFT

Separate SFT 是分别训练 Temporal 和 Spatial 的 SFT checkpoint，作为后续 RL 的初始化或对照。这里不需要 `<think>`，因为旧数据集没有真实思考链。SFT 的目标应该是学会任务格式和可验证答案，不应该监督模型编造 reasoning。

Temporal SFT:

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    'To accurately pinpoint the event "person takes a drink from the cup" '
                    "in the video, determine the precise time period of the event.\n\n"
                    'Provide the start and end times (in seconds, precise to two decimal places) '
                    'in the format "start time to end time" within the <answer> </answer> tags. '
                    'For example: "12.54 to 17.83".'
                ),
            },
            {
                "type": "video",
                "video": "/path/to/video.mp4",
                "total_pixels": 3584 * 28 * 28,
                "min_pixels": 16 * 28 * 28,
            },
        ],
    },
    {
        "role": "assistant",
        "content": "<answer>14.20 to 22.50</answer>",
    },
]
```

Spatial SFT:

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    'Track the "motorcycle wheel" in the video based on its initial coordinates '
                    '"[0.58, 0.50, 0.64, 0.60]". The output should be a list containing '
                    "eight sublists. Each sublist includes four normalized coordinates "
                    "[x0, y0, x1, y1] representing the bounding box of the object at "
                    "specific time intervals.\n\n"
                    "Provide your answer within the <answer> </answer> tags."
                ),
            },
            {
                "type": "video",
                "video": [
                    "/path/to/frame_000001.jpg",
                    "/path/to/frame_000020.jpg",
                    "/path/to/frame_000040.jpg",
                    "/path/to/frame_000060.jpg",
                    "/path/to/frame_000080.jpg",
                    "/path/to/frame_000100.jpg",
                    "/path/to/frame_000120.jpg",
                    "/path/to/frame_000140.jpg",
                ],
                "total_pixels": 3584 * 28 * 28,
                "min_pixels": 16 * 28 * 28,
            },
        ],
    },
    {
        "role": "assistant",
        "content": "<answer>[[0.58, 0.50, 0.64, 0.60], [0.58, 0.50, 0.64, 0.60], [0.53, 0.48, 0.57, 0.56], [0.70, 0.38, 0.76, 0.47], [0.56, 0.50, 0.61, 0.59], [0.49, 0.59, 0.54, 0.69], [0.34, 0.51, 0.38, 0.62], [0.32, 0.48, 0.33, 0.56]]</answer>",
    },
]
```

实现上建议只对 assistant 部分算 SFT loss。VideoChat-R1 的 SFT collator 直接把整段 chat template 克隆成 labels，只 mask 了 pad token 和 video token；这个能跑，但不是最干净的 assistant-only SFT。

## Separate RL

Separate RL 是分别训练两个能力教师：Temporal RL Teacher 和 Spatial RL Teacher。和 SFT 不同，RL 阶段没有 assistant label；模型在线采样 completion，reward 从 completion 中解析 `<answer>`。

RL-family 实验统一使用 system prompt 要求 `<think>` 和 `<answer>`。user prompt 不再写 “Output your thought process”，否则同一个结构约束会在 system 和 user 中重复。reward 不应该把自然语言思考当成能力本身；`<think>` 只提供统一的生成结构，真正的任务 reward 仍然来自 `<answer>`。

Temporal RL messages:

```python
messages = [
    {
        "role": "system",
        "content": RL_SYSTEM_PROMPT,
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    'To accurately pinpoint the event "person takes a drink from the cup" '
                    "in the video, determine the precise time period of the event.\n\n"
                    'Provide the start and end times (in seconds, precise to two decimal places) '
                    'in the format "start time to end time" within the <answer> </answer> tags. '
                    'For example: "12.54 to 17.83".'
                ),
            },
            {
                "type": "video",
                "video": "/path/to/video.mp4",
                "total_pixels": 3584 * 28 * 28,
                "min_pixels": 16 * 28 * 28,
            },
        ],
    }
]
```

Temporal reward:

```python
reward = format_reward(completion) + temporal_iou_reward(completion, gt_start, gt_end)
```

Spatial RL messages:

```python
messages = [
    {
        "role": "system",
        "content": RL_SYSTEM_PROMPT,
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    'Track the "motorcycle wheel" in the video based on its initial coordinates '
                    '"[0.58, 0.50, 0.64, 0.60]". The output should be a list containing '
                    "eight sublists. Each sublist includes four normalized coordinates "
                    "[x0, y0, x1, y1] representing the bounding box of the object at "
                    "specific time intervals.\n\n"
                    "Provide the final trajectory within the <answer> </answer> tags."
                ),
            },
            {
                "type": "video",
                "video": [
                    "/path/to/frame_000001.jpg",
                    "/path/to/frame_000020.jpg",
                    "/path/to/frame_000040.jpg",
                    "/path/to/frame_000060.jpg",
                    "/path/to/frame_000080.jpg",
                    "/path/to/frame_000100.jpg",
                    "/path/to/frame_000120.jpg",
                    "/path/to/frame_000140.jpg",
                ],
                "total_pixels": 3584 * 28 * 28,
                "min_pixels": 16 * 28 * 28,
            },
        ],
    }
]
```

Spatial reward:

```python
reward = format_reward(completion) + trajectory_average_iou_reward(completion, gt_boxes)
```

## Mixed-RL

Mixed-RL 是把 Temporal 和 Spatial 数据混在一起，训练一个总模型。它不是 MOPD，因为没有冻结教师，也没有 teacher log-prob 蒸馏；每条样本仍然只使用自己的 reward。

messages 本身和 Separate RL 完全相同，差别只在 dataloader 和 reward router：

```python
if example["data_type"] == "temporal":
    messages = build_temporal_rl_messages(example)
    reward = temporal_format_reward + temporal_iou_reward
elif example["data_type"] == "spatial":
    messages = build_spatial_rl_messages(example)
    reward = spatial_format_reward + trajectory_average_iou_reward
```

Mixed-RL 的核心控制变量是采样比例，而不是 prompt。第一版应该固定每个能力域的 sample budget 或 token budget；否则视频长度、帧数和视觉 token 预算会混进能力整合效果里。

## MOPD

MOPD 的 prompt messages 也和 Separate RL 保持一致，包括同一个 `RL_SYSTEM_PROMPT`。区别不在输入格式，而在优化信号：学生先根据 system + user prompt 自己生成 completion，然后把同一条学生轨迹路由到对应冻结教师，教师在这条轨迹上提供 token-level log-prob。

学生生成时看到的是 prompt-only messages，即 system + user，没有 assistant label：

```python
student_prompt_messages = build_temporal_rl_messages(example)
student_completion = student.generate(student_prompt_messages)
```

教师打分时看的是同一个 prompt 加学生已经生成的 assistant 内容。注意这里不是让教师重新生成答案，而是 teacher forcing / prefill，用教师计算学生轨迹上每个 token 的 log-prob。

```python
teacher_prefill_messages = [
    *student_prompt_messages,
    {
        "role": "assistant",
        "content": student_completion,
    },
]

teacher_logprobs = temporal_teacher.prefill_logprobs(teacher_prefill_messages)
student_logprobs = student.logprobs(teacher_prefill_messages)
loss = reverse_kl_on_student_trajectory(student_logprobs, teacher_logprobs)
```

路由规则仍然由能力域决定：

```python
if example["data_type"] == "temporal":
    teacher = temporal_rl_teacher
elif example["data_type"] == "spatial":
    teacher = spatial_rl_teacher
```

这也是为什么 MOPD 里的教师不是数据集教师。`Charades` 和 `ActivityNet` 只是 Temporal domain 的数据来源；`GoT` 是当前 Spatial domain 的视频轨迹数据来源。MOPD 的路由单位应该是 capability domain，而不是 dataset name。

## 四类实验的差异

```text
Separate SFT:
  prompt + ground-truth assistant answer
  trains one checkpoint per capability domain

Separate RL:
  system + user prompt only
  model samples completion online
  reward comes from that domain's verifier
  trains one RL teacher per capability domain

Mixed-RL:
  same system + user prompt messages as Separate RL
  temporal and spatial data mixed in one dataloader
  one student/model optimized by routed rewards
  no teacher distillation

MOPD:
  same system + user prompt messages as Separate RL
  student samples completion online
  completion is routed to the matching frozen capability teacher
  loss uses token-level teacher log-prob on the student's own trajectory
```

所以从 messages 角度看，Separate RL、Mixed-RL 和 MOPD 应该尽量完全一致：同一个 system prompt，同一套 task user prompt，同一套 answer schema。这样最后比较的是优化范式，而不是 prompt 工程。SFT 可以使用更短的 no-think target，因为它只是格式和答案空间的 warmup，不承担能力相互作用的主要结论。
