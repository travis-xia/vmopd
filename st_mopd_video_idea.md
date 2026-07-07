# 用 MOPD 研究视频时空能力之间的相互作用

这个工作的初步尝试不需要把所有视频能力都放进来。QA、caption、event abstraction 以后可以补；**第一版只看最核心、最可验证、也最容易和 VideoChat-R1 接起来的时空能力：时间定位、静态空间定位、时空目标跟踪。**

对应的数据源可以先定为：

```text
Temporal: Charades / ActivityNet
Static Spatial: COCO / RefCOCO
Spatiotemporal Tracking: GoT
```

但论文里不能把它们写成 **Charades Teacher、COCO Teacher、GoT Teacher**。那样会变成数据集拼接。这里真正要研究的不是“多个数据集能不能混着蒸馏”，而是**不同可验证奖励塑造出的时空能力，能否在同一个学生模型里发生可解释的正迁移或干扰**。

更准确的表述是：**奖励函数定义能力域，能力域训练出教师，MOPD 用这些教师观察时空能力之间的相互作用。**

## 教师能力域，而不是数据集教师

- **Temporal Teacher** 专门学时间定位。它由 Charades-STA 和 ActivityNet-Grounding 这类文本到时间片段的数据塑造，优化目标是 temporal IoU、R@0.5、R@0.7 这类时间重叠指标。这个教师学到的不是某个数据集的风格，而是“给定语言查询，在视频里找到相关时间段”的能力。它更像一个视频证据检索器，负责把长视频压缩成关键片段。

- **Static Spatial Teacher** 专门学框定位。它由 COCO、RefCOCO 系列数据塑造，优化目标是 box IoU、指代表达 grounding accuracy 或检测/定位正确率。这个教师本身不理解时间，但它提供对象级空间锚点：目标是什么、在哪里、语言表达指向哪个区域。它可以被看作视频空间能力的静态来源。

- **Spatiotemporal Tracking Teacher** 专门学目标在时间中的连续存在。它由 GoT 这类 tracking 数据塑造，优化目标是轨迹重叠、average overlap、success rate 或逐帧 box IoU。它和 Static Spatial Teacher 的区别很关键：Static Spatial 解决的是单帧里“在哪”，Spatiotemporal Tracking 解决的是跨帧后“还在哪里”。它学到的是对象恒常性、运动连续性、遮挡后的再识别，以及目标状态随时间变化的边界。

这三个教师都属于时空能力，但不是同一种能力。**Temporal Teacher 关注事件何时发生，Static Spatial Teacher 关注对象在画面中的位置，Spatiotemporal Tracking Teacher 关注同一个对象如何随时间延续。** 它们共同构成一个最小但清晰的时空能力三角。

## 为什么 MOPD 适合这个问题

VideoChat-R1 已经证明，时空奖励可以通过 RL 显著增强视频 MLLM；MOPD 提供的不是另一个更花哨的训练名字，而是一种更干净的观察方式。普通 joint RL 把所有奖励混在一起，最后只能看到“混合训练涨了”。**MOPD 先把不同奖励函数分别训练成同源教师，再在学生自身轨迹上同步蒸馏这些教师，因此可以把“能力生产”和“能力整合”拆开。**

这点对视频尤其重要。视频能力之间的关系不是简单的层级结构。时间定位可能帮助 tracking，因为它减少了模型需要搜索的帧范围；tracking 也可能帮助时间定位，因为很多动作边界由对象状态变化决定；静态框定位可能帮助 tracking 的起点和再识别，但未必帮助事件级时间判断。MOPD 的价值就在于，它让这些关系可以通过 teacher source 的加入、去除和加权被测出来，而不是停留在直觉上。

因此方法部分不应该只写“我们有三个 teacher”。强化学习论文会认真写奖励函数设计，因为奖励函数定义了模型被优化成什么行为；这篇 MOPD 化的视频论文也应该认真写 **teacher capability domain**。每个教师的边界都由三件事共同决定：**数据形态、输出格式、奖励函数**。

```text
Temporal domain = temporal segment data + timestamp output + temporal IoU reward
Static spatial domain = image grounding data + box output + box IoU reward
Spatiotemporal domain = tracking data + trajectory output + tracking overlap reward
```

这样写的好处是，教师划分不是主观命名，而是由可验证优化目标自然诱导出来的。MOPD 继承的也不是数据集标签，而是这些奖励函数塑造出的策略偏好。

## 第一版最值得看的相互作用

第一版不要急着引入 QA 或 Event Teacher。更有价值的是先把底层时空关系讲清楚：时间、空间和轨迹到底谁在帮助谁。

- **Temporal to Spatiotemporal** 是最自然的正迁移假设。时间定位把视频切到相关片段，tracking 在更短、更相关的片段里更容易保持目标。如果这个迁移成立，提升应该主要出现在长视频、目标短暂出现、背景干扰强的样本上，而不是所有 tracking 样本平均上涨。这说明 Temporal Teacher 提供的是搜索空间压缩，而不是泛泛的视觉增强。

- **Spatiotemporal to Temporal** 可能更有洞见。很多事件边界不是语言匹配出来的，而是对象状态变化决定的。例如“拿起杯子”的开始和结束，取决于手、杯子、桌面的空间关系如何随时间改变。如果加入 Spatiotemporal Teacher 后，Temporal 的 R@0.7 或 mIoU 提升比 R@0.3 更明显，那说明 tracking 能力主要帮助边界精修，而不是粗粒度召回。

- **Static Spatial to Spatiotemporal** 应该是强相关但有限的迁移。静态框定位能帮助模型建立对象概念和空间锚点，对 tracking 的初始化、遮挡恢复、相似物区分可能有帮助。但如果只有静态空间能力，没有时间连续性，它很可能只能提升短片段或低遮挡样本，对长程 tracking 的帮助有限。这个差异正好可以证明 Static Spatial 和 Spatiotemporal Tracking 不该被混成一个 Spatial Teacher。

- **Spatiotemporal to Static Spatial** 则可能较弱。tracking 训练通常给定初始框，任务重点是维持目标身份，而不是从语言表达中找对象。因此它未必能反向提升 RefCOCO 式的语言条件框定位。如果结果确实如此，也很有价值：跨帧保持对象和单帧语言 grounding 是相邻但不等价的能力。

- **Temporal to Static Spatial** 也不一定强。时间定位学的是事件片段级匹配，不要求输出框；它可能让模型更会看相关帧，但不一定让模型更会画框。反过来，Static Spatial to Temporal 可能在高 IoU 时间指标上有帮助，因为对象状态变化可以提供边界线索，但这种帮助应该弱于 Spatiotemporal Tracking。

这些不是最终要强行证明的结论，而是第一版实验的假设地图。真正的结果可能有正迁移，也可能有负迁移；只要控制实验干净，负结果同样有解释力。

## 实验组织

训练仍然遵循原始 MOPD：学生从共享 SFT 初始化，在混合 batch 中采样不同能力域的 prompt；每条样本只路由到自己的能力教师；教师在学生自身生成轨迹上提供 token 级蒸馏信号。不要把三种输出格式串成一个统一推理链，也不要要求一个样本同时输出时间段、框和轨迹。

关键不是 full model 最终分数，而是 source capability 对 target evaluation 的边际影响。第一版可以围绕三类实验展开：

1. **Single-source MOPD**：只蒸馏一个能力教师，然后评估三个能力域，看它自然迁移到哪里。
2. **Leave-one-source-out MOPD**：从 full temporal-spatial-spatiotemporal MOPD 中去掉一个教师，看哪些目标指标下降。
3. **Source-ratio scaling**：保持三类教师同时存在，只改变某一类 source 的 batch/token 占比，观察 target 指标是否随之单调变化。

这里要特别控制数据量。**COCO/RefCOCO 的规模远大于 Charades、ActivityNet 和 GoT**，如果按原始数据规模采样，Spatial Teacher 很容易主导训练，最后看到的是数据量效应，不是能力迁移。更合理的是按能力域固定 sample budget 或 token budget，再单独做 ratio scaling。

## 论文主线

这篇工作的核心不是提出一个“更强的视频大模型”，而是提出一个可控的时空能力分析框架：

**我们根据可验证奖励函数将视频时空能力划分为 Temporal、Static Spatial 和 Spatiotemporal Tracking 三个能力域，分别训练同源领域教师，并用 MOPD 在统一学生模型中测量这些能力之间的方向性迁移关系。**

如果最后 full MOPD 模型涨分很多，那是好结果；但即使涨幅中等，只要能回答哪些能力帮助哪些能力、帮助发生在哪类样本上、提升主要体现在粗召回还是边界精修上，这个工作仍然成立。它比简单地说“多教师蒸馏提升视频理解”更像一个视频理解研究问题：时间提供检索，空间提供对象锚点，轨迹提供跨帧身份连续性，而 MOPD 提供观察这些能力如何在共享学生参数中相互作用的实验工具。
