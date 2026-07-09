# 用 MOPD 研究视频 Temporal 与 Spatial 能力的整合

这个工作的第一版不需要把所有视频能力都放进来。QA、caption、event abstraction 以后可以补；当前先看最核心、最可验证、也最容易和 VideoChat-R1 接起来的两类能力：**时间定位 Temporal** 和 **空间定位 Spatial**。

这里需要统一命名：我们现在使用 GoT 的方式也是一种 **Spatial**。虽然 GoT 的样本是视频序列，输出是跨帧框轨迹，但优化目标仍然是对象在视觉空间中的位置是否正确。因此第一版直接把 GoT 纳入 `Spatial Teacher`。

对应的数据源先定为：

```text
Temporal: Charades-STA / ActivityNet-Grounding
Spatial: GoT-10k
```

后续如果加入 COCO / RefCOCO，它们也应该作为 Spatial domain 的静态 grounding 子任务，而不是新增一个和 GoT 平行的能力域。论文里也不能把它们写成 **Charades Teacher** 或 **GoT Teacher**。那样会变成数据集拼接。这里真正要研究的是：**不同可验证奖励塑造出的 Temporal 与 Spatial 能力，能否通过 MOPD 在同一个学生模型中比 Mixed-RL 更稳定地整合。**

更准确的表述是：**奖励函数定义能力域，能力域训练出教师，MOPD 用教师在学生自身轨迹上的 token-level 信号来整合能力。**

## 教师能力域，而不是数据集教师

- **Temporal Teacher** 专门学时间定位。它由 Charades-STA 和 ActivityNet-Grounding 这类文本到时间片段的数据塑造，优化目标是 temporal IoU、R@0.5、R@0.7 这类时间重叠指标。这个教师学到的不是某个数据集的风格，而是“给定语言查询，在视频里找到相关时间段”的能力。它更像一个视频证据检索器，负责把长视频压缩成关键片段。

- **Spatial Teacher** 专门学视觉空间定位。第一版里它由 GoT-10k 这类 tracking 数据塑造，输入是视频帧序列、对象名和初始框，输出是若干采样时刻上的归一化框轨迹，优化目标是 trajectory average IoU、average overlap、success rate 或逐帧 box IoU。这里的 Spatial 不是只指静态图片框定位，而是指对象在画面空间中的位置判断；GoT 只是这个空间能力在视频序列上的形态。

这两个教师都属于视频时空理解的底层能力，但边界不同。**Temporal Teacher 关注事件何时发生，Spatial Teacher 关注目标在画面中在哪里，以及这个位置如何跨帧保持。** 对第一版来说，这个二分更干净，因为我们当前没有引入 COCO / RefCOCO 作为独立静态空间训练源。

## 为什么 MOPD 适合这个问题

VideoChat-R1 已经证明，时空奖励可以通过 RL 显著增强视频 MLLM；MOPD 提供的不是另一个更花哨的训练名字，而是一种更干净的能力整合方式。普通 Mixed-RL 把 Temporal 和 Spatial 样本放在同一个 RL 过程里，最后只能看到“混合训练涨了或没涨”。**MOPD 先把不同奖励函数分别训练成同源教师，再在学生自身轨迹上同步蒸馏这些教师，因此可以把“能力生产”和“能力整合”拆开。**

这点对视频尤其重要。时间定位可能帮助 Spatial tracking，因为它减少了模型需要搜索的帧范围；Spatial tracking 也可能帮助时间定位，因为很多动作边界由对象状态和空间关系变化决定。MOPD 的价值就在于，它让这些关系可以通过 teacher source 的加入、去除和加权被测出来，而不是停留在直觉上。

因此方法部分不应该只写“我们有两个数据集”。强化学习论文会认真写奖励函数设计，因为奖励函数定义了模型被优化成什么行为；这篇 MOPD 化的视频论文也应该认真写 **teacher capability domain**。每个教师的边界都由三件事共同决定：**数据形态、输出格式、奖励函数**。

```text
Temporal domain = temporal segment data + timestamp output + temporal IoU reward
Spatial domain = tracking frame data + trajectory box output + tracking overlap reward
```

这样写的好处是，教师划分不是主观命名，而是由可验证优化目标自然诱导出来的。MOPD 继承的也不是数据集标签，而是这些奖励函数塑造出的策略偏好。

## 第一版最值得看的相互作用

第一版不要急着引入 QA、Event Teacher 或静态图片 grounding。更有价值的是先把 Temporal 与 Spatial 这两个视频底层能力的整合关系讲清楚。

- **Temporal to Spatial** 是最自然的正迁移假设。时间定位把视频切到相关片段，Spatial tracking 在更短、更相关的片段里更容易保持目标。如果这个迁移成立，提升应该主要出现在长视频、目标短暂出现、背景干扰强的样本上，而不是所有 GoT 样本平均上涨。这说明 Temporal Teacher 提供的是搜索空间压缩，而不是泛泛的视觉增强。

- **Spatial to Temporal** 可能更有洞见。很多事件边界不是语言匹配出来的，而是对象状态变化决定的。例如“拿起杯子”的开始和结束，取决于手、杯子、桌面的空间关系如何随时间改变。如果加入 Spatial Teacher 后，Temporal 的 R@0.7 或 mIoU 提升比 R@0.3 更明显，那说明 spatial tracking 能力主要帮助边界精修，而不是粗粒度召回。

- **Spatial 单独训练的泛化边界** 也值得看。GoT 训练通常给定初始框，任务重点是维持目标身份和位置连续性，而不是从语言表达中找对象。因此它未必自然提升 RefCOCO 式语言条件框定位。这个点可以作为后续扩展静态 grounding 子任务时的动机，但不需要在第一版里提前拆成第三个教师。

这些不是最终要强行证明的结论，而是第一版实验的假设地图。真正的结果可能有正迁移，也可能有负迁移；只要控制实验干净，负结果同样有解释力。

## 实验组织

训练仍然遵循原始 MOPD：学生从共享 SFT 初始化，在混合 batch 中采样不同能力域的 prompt；每条样本只路由到自己的能力教师；教师在学生自身生成轨迹上提供 token 级蒸馏信号。不要把 Temporal 和 Spatial 输出串成一个统一推理链，也不要要求一个样本同时输出时间段和轨迹框。

关键不是 full model 最终分数，而是在同样数据和 prompt 约束下，MOPD 是否比 Mixed-RL 更好地整合这两个视频能力。第一版可以围绕三类实验展开：

1. **Separate RL Teachers**：分别训练 Temporal RL Teacher 和 Spatial RL Teacher，确认单域 RL 教师本身有效。
2. **Mixed-RL baseline**：把 Temporal 和 Spatial 数据混在一起，用 reward router 训练一个模型，作为最直接基线。
3. **MOPD**：使用同样的 Temporal / Spatial prompt 和采样比例，但把学生轨迹路由到冻结的对应能力教师，用 teacher log-prob 做在策略蒸馏。

如果资源允许，可以再做两个分析实验：

1. **Single-source MOPD**：只蒸馏一个能力教师，然后同时评估 Temporal 和 Spatial，看它自然迁移到哪里。
2. **Source-ratio scaling**：保持两类教师同时存在，只改变某一类 source 的 batch/token 占比，观察 target 指标是否随之单调变化。

这里要特别控制数据量和 token 量。GoT 的样本会带来多帧视觉 token，Charades / ActivityNet 的视频长度和采样方式也会影响训练成本。如果只按原始样本数采样，结果可能变成模态 token 预算效应，而不是能力整合效应。更合理的是按能力域固定 sample budget 或 token budget，再单独做 ratio scaling。

## 论文主线

这篇工作的核心不是提出一个“更强的视频大模型”，而是提出一个可控的视频能力整合实验：

**我们根据可验证奖励函数将视频能力划分为 Temporal 和 Spatial 两个能力域，其中 GoT tracking 被视为 Spatial 的视频轨迹形态；随后分别训练同源领域教师，并用 MOPD 在统一学生模型中整合这两个能力，检验它是否优于 Mixed-RL。**

如果最后 full MOPD 模型涨分很多，那是好结果；但即使涨幅中等，只要能回答 Temporal 与 Spatial 哪个方向的迁移更稳定、帮助发生在哪类样本上、提升主要体现在粗召回还是边界精修上，这个工作仍然成立。它比简单地说“多教师蒸馏提升视频理解”更像一个视频理解研究问题：时间提供检索，空间提供对象位置和跨帧身份连续性，而 MOPD 提供观察这些能力如何在共享学生参数中整合的实验工具。
