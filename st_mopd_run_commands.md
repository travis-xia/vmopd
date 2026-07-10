# ST-MOPD 第一阶段运行命令

当前只准备 SFT 对照、Temporal/Spatial teacher RL、Mixed-RL。MOPD 蒸馏下一步再接。

## 1. 构建数据集

建议在集群上构建，因为 GoT 需要从真实序列目录枚举 8 帧。

```bash
bash examples/train/st_mopd/build_dataset.sh
```

输出：

```text
data/st_mopd/temporal_sft.jsonl
data/st_mopd/spatial_sft.jsonl
data/st_mopd/mixed_sft.jsonl
data/st_mopd/temporal_rl.jsonl
data/st_mopd/spatial_rl.jsonl
data/st_mopd/mixed_rl.jsonl
data/st_mopd/manifest.json
```

小样本冒烟：

```bash
OUTPUT_DIR=data/st_mopd_smoke bash examples/train/st_mopd/build_dataset.sh \
  --max-samples-per-source 4 \
  --max-spatial-samples 4 \
  --pretty
```

如果路径和默认集群路径不同：

```bash
CHARADES_VIDEO_ROOT=/path/to/Charades_v1_480 \
ACTIVITYNET_VIDEO_ROOT=/path/to/activitynet/videos \
GOT_ROOT=/path/to/GoT-10k \
OUTPUT_DIR=data/st_mopd \
bash examples/train/st_mopd/build_dataset.sh
```

如果 ActivityNet annotation 使用集群上的 `val_1.json`：

```bash
ACTIVITYNET_ANNO_FILE=/path/to/activitynet/captions/val_1.json \
bash examples/train/st_mopd/build_dataset.sh
```

## 2. SFT 对照组

默认使用 `mixed_sft.jsonl`，这是独立对照组；后面的 teacher RL 和 Mixed-RL 不从这个 SFT checkpoint 继续。
默认模型目录是集群本地路径：

```text
/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/Qwen2.5-VL-7B-Instruct
```

```bash
bash examples/train/st_mopd/sft.sh
```

只做某个域的 SFT：

```bash
DATASET=data/st_mopd/temporal_sft.jsonl OUTPUT_DIR=output/st_mopd/sft_temporal \
bash examples/train/st_mopd/sft.sh

DATASET=data/st_mopd/spatial_sft.jsonl OUTPUT_DIR=output/st_mopd/sft_spatial \
bash examples/train/st_mopd/sft.sh
```

## 3. 训练单域 RL teachers

Temporal teacher：

```bash
DOMAIN=temporal bash examples/train/st_mopd/teacher_rl.sh
```

Spatial teacher：

```bash
DOMAIN=spatial bash examples/train/st_mopd/teacher_rl.sh
```

这两个脚本默认都从集群本地 `Qwen2.5-VL-7B-Instruct` 目录启动，不加载 SFT checkpoint。

## 4. Mixed-RL baseline

```bash
bash examples/train/st_mopd/mixed_rl.sh
```

Mixed-RL 使用 `mixed_rl.jsonl`，每条样本仍只走自己的 reward：temporal 样本走 temporal IoU，spatial 样本走 trajectory average IoU。
RL 脚本中格式奖励和答案奖励等权：`--reward_weights 1.0 1.0`。

## 常用覆盖项

```bash
MODEL=/path/to/base_or_instruct_model \
LOCAL_MODEL_DIR=/path/to/default_base_or_instruct_model \
NPROC_PER_NODE=8 \
DEEPSPEED=zero3 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
bash examples/train/st_mopd/mixed_rl.sh
```

所有训练脚本默认都是全参训练：

```bash
DEEPSPEED=zero3 bash examples/train/st_mopd/mixed_rl.sh
```

默认脚本已设置：

```text
HF_HUB_DISABLE_SSL_VERIFICATION=1
TRANSFORMERS_OFFLINE=1
HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
CURL_CA_BUNDLE=
REQUESTS_CA_BUNDLE=
```

训练命令也显式传入 `--check_model false`，避免离线集群上触发 ModelScope 的本地模型最新性检查。

## 5. Evaluation

annotation 里已有 val/test split。默认评测用 Charades/ActivityNet/GoT 的 `val`：

```bash
bash examples/train/st_mopd/evaluate.sh
```

脚本会依次生成：

```text
data/st_mopd_eval/temporal_eval.jsonl
data/st_mopd_eval/spatial_eval.jsonl
data/st_mopd_eval/mixed_eval.jsonl
output/st_mopd/eval/<run>/predictions.jsonl
output/st_mopd/eval/<run>/metrics.json
output/st_mopd/eval/<run>/per_sample.jsonl
```

也可以分步跑：

```bash
ACTION=build bash examples/train/st_mopd/evaluate.sh
ACTION=infer MODEL=/path/to/checkpoint bash examples/train/st_mopd/evaluate.sh
ACTION=score RESULT_PATH=output/st_mopd/eval/<run>/predictions.jsonl bash examples/train/st_mopd/evaluate.sh
```

如果希望三步写到同一个固定目录，可以设置同一个 `EVAL_RUN_NAME` 或直接设置 `OUTPUT_DIR`。

如果要评 test split：

```bash
CHARADES_SPLIT=charades_test ACTIVITYNET_SPLIT=test GOT_SPLIT=val \
bash examples/train/st_mopd/evaluate.sh
```

打分脚本也可以单独用于已有推理结果：

```bash
python3 st_mopd/evaluate.py score \
  --predictions output/st_mopd/eval/<run>/predictions.jsonl \
  --output output/st_mopd/eval/<run>/metrics.json \
  --per-sample-output output/st_mopd/eval/<run>/per_sample.jsonl \
  --pretty
```
