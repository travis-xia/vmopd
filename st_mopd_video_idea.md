# 用 MOPD 研究视频能力之间的促进关系

直接把 MOPD 搬到 VideoChat-R1 上，容易变成一个普通的“多任务 teacher 合并”工作：时间定位一个 teacher，目标跟踪一个 teacher，QA 一个 teacher，最后蒸到一个学生里。这当然能做，但研究问题不够尖锐。视频领域更有意思的不是“多个能力能否被合并”，而是这些能力之间是否存在可解释的促进关系。

我觉得更好的表述是：**把 MOPD 当成分析视频能力迁移关系的工具，而不是把它当成最终模型结构的卖点**。最终模型仍然应该根据不同题目输出不同格式：时间定位题输出 `<clue>` 或时间段，跟踪题输出 `<track>`，QA 题输出 `<answer>`，caption 题输出事件描述。我们不要求一个样本同时输出所有结构化字段，也不把 `<event> -> <clue> -> <track> -> <answer>` 固定成推理链。真正要研究的是，在 MOPD 的同步混合训练中，某一类能力教师带来的梯度更新会怎样影响其他能力的评测表现。

## 能力教师，而不是数据集教师

这里的 teacher 不应该简单命名为 Charades Teacher、GoT Teacher、NExTGQA Teacher。那样还是数据集视角，论文味道会变成数据集拼接。更自然的是把它们抽象成视频能力：

Temporal Teacher 学的是时间定位能力。它知道一个事件大概发生在哪一段，擅长从视频中找到与查询相关的时间片段。

Spatial Teacher 学的是空间和轨迹能力。它知道目标在哪里、如何移动、何时出现或消失，擅长捕捉对象级状态变化。

QA Teacher 学的是问题回答能力。它不一定显式输出时间或轨迹，但它最终要把视频证据压缩成一个答案。

Event Teacher 学的是事件抽象能力。它把局部视觉变化组织成“发生了什么”，介于低层时空证据和高层语义答案之间。

这个划分的关键不是让学生在每个样本里同时模仿四个 teacher，而是提供一组可控的能力来源。MOPD 合并阶段仍然遵循原始机制：从混合数据中采样一个 batch，每条样本按照自己的能力域路由到对应 teacher，所有样本共同更新同一个 student。能力之间的促进关系不是通过显式串联 teacher 得到的，而是通过共享 student 参数后产生的跨域泛化和干扰来体现。

## 从奖励函数设计到教师能力域划分

这个点可以成为方法部分的主线。强化学习论文通常要写奖励函数设计，因为奖励函数定义了模型被优化成什么样的行为；MOPD 类论文也应该有一个对应部分，只不过它不再只写单个 reward，而是写 **teacher capability domains**：每个 teacher 代表什么能力域，这个能力域由哪些数据、输出形式和奖励函数共同塑形。

换句话说，教师不是凭空划分出来的。Temporal Teacher 之所以是时间教师，是因为它由时间定位任务和 temporal IoU reward 训练出来；Spatial Teacher 之所以是空间教师，是因为它由框、轨迹和 spatial IoU / tracking reward 训练出来；QA Teacher 由 answer accuracy reward 训练出来；Event Teacher 则可能由 caption recall、event entailment 或事件覆盖率 reward 训练出来。

这样写有两个好处。第一，它避免 teacher 划分显得主观。每个能力域都有清楚的优化目标，reward function 就是能力边界的定义。第二，它把 VideoChat-R1 和 MOPD 自然接起来：VideoChat-R1 的方法部分强调不同视频任务的奖励函数，MOPD 化之后的方法部分则强调这些奖励函数如何诱导出不同能力教师，以及这些教师之间如何发生迁移。

一个更准确的方法表述可以是：

```text
Reward functions define capability domains.
Capability-specific RL produces domain teachers.
MOPD is used to probe and transfer these teacher-induced capabilities.
```

因此，这个工作不应该只说“我们有四个 teacher”，而应该说“我们根据视频任务中可验证奖励的语义，将视频能力划分为 temporal、spatial、event 和 answer 四个能力域，并用对应奖励训练同源教师”。这比按数据集划分更有原则，也更像一个方法设计。

## MOPD 合并仍然是同步混合训练

一个重要原则是：**任务输出格式保持原样**。

时间定位样本仍然只要求模型输出时间段。目标跟踪样本仍然只要求模型输出轨迹或框。QA 样本仍然只要求模型输出答案。Caption/Event 样本仍然只要求模型输出描述或事件。不同能力之间的关系不通过“把所有标签串在一个输出里”来体现，而通过训练信号的路由、蒸馏权重、样本选择和交叉评估来体现。

这里不能把“Temporal-to-Spatial”理解成训练时把 temporal teacher 接到 spatial 样本上。原始 MOPD 不是这种跨 teacher 路由。更准确的理解是：在同一个 mixed-batch MOPD 训练中，Temporal 样本经 Temporal Teacher 蒸馏产生的参数更新，是否会提升 Spatial 评测；Spatial 样本经 Spatial Teacher 蒸馏产生的参数更新，是否会提升 Temporal 评测。这种关系发生在共享学生参数里，而不是发生在单条样本的输出结构里。

因此，所谓“方向”不是训练顺序，也不是推理链方向，而是 **source capability 对 target evaluation 的边际影响**。source 是被加入、去掉或加权的能力域；target 是最终评测的能力域。比如我们说 Temporal helps Spatial，意思是：在控制其他能力域训练不变的情况下，加入或增强 Temporal Teacher 信号后，Spatial tracking/grounding 指标变好。

## 真正值得研究的关系

第一个关系是 **time helps space**。时间定位可能像一个视频检索器，让模型更容易学会关注事件发生的关键片段。这个结论不需要训练时显式让 temporal teacher 监督 tracking 样本；只需要观察在 mixed-batch MOPD 中加入 Temporal 能力域后，Spatial 评测是否提升。如果提升主要发生在长视频、稀疏事件、背景干扰强的样本上，就说明时间能力提供的是搜索空间压缩，而不是普遍的视觉增强。

第二个关系是 **space refines time**。很多事件边界不是靠语言匹配出来的，而是靠对象状态变化决定的。例如“拿起杯子”的边界在手接触杯子和杯子离桌之间，“放下书”的边界在书重新接触桌面时。如果在 mixed-batch MOPD 中加入 Spatial 能力域后，时间定位的 R@0.7 或 mIoU 提升更明显，而 R@0.3 提升有限，就说明空间能力主要帮助边界精修，而不是粗粒度片段召回。

第三个关系是 **event bridges evidence and answer**。事件能力可能不是 caption 任务独有的能力，而是视频 QA 的中间层。很多 QA 错误不是因为模型完全没看到目标，而是没有把目标运动组织成正确事件。如果 Event Teacher 能提升 QA，尤其是 why/how 类型问题，而对纯定位任务帮助较小，就可以说明事件抽象主要服务于语义推理。

第四个关系是 **QA may not teach grounding automatically**。QA teacher 可能很强，但它未必能反向提升时间或空间能力。模型可以靠语言先验和全局视觉印象答对很多题，却没有学到精确定位。如果 QA-to-Temporal 或 QA-to-Spatial 迁移很弱，这本身就是一个有价值的结论：答案正确不等于证据可靠。

## 实验应该围绕能力贡献，而不是只看最终融合

最关键的实验不是 full model 分数，而是能力贡献矩阵。把能力作为行和列，行表示被控制的 source capability，列表示评估的 target capability：

```text
              eval Temporal   eval Spatial   eval Event   eval QA
source Temporal      -             ?            ?          ?
source Spatial       ?             -            ?          ?
source Event         ?             ?            -          ?
source QA            ?             ?            ?          -
```

这张矩阵比“多任务混合训练涨了多少”更重要。它能回答哪些能力是基础能力，哪些能力是下游能力，哪些能力带来正迁移，哪些能力带来干扰。

具体实验仍然是 MOPD 风格的 mixed-batch 训练，不是串行训练。可以做三类控制：

1. **Single-source MOPD**：只加入一个能力域的数据和 teacher，然后评估所有能力，看它自然迁移到哪里。
2. **Leave-one-source-out MOPD**：从 full mixed-batch MOPD 中去掉一个能力域，看哪些评测项下降，由此估计该能力域的边际贡献。
3. **Source-ratio scaling**：保持所有能力域同时训练，只改变某个 source capability 的 batch 占比，看 target capability 是否随之单调变化。

如果去掉 Temporal 后 Spatial 指标明显下降，而去掉 Spatial 后 Temporal 的粗召回几乎不变、但高 IoU 边界指标下降，就可以得到更细的结论：时间能力更影响空间任务中的证据召回，空间能力更影响时间任务中的边界精修。如果 Event source 对 QA 明显有帮助，但 QA source 对 Event 或 Grounding 帮助弱，则说明事件抽象更像问答的前置中间能力，而不是问答能力的自然副产品。

## 这个方向的核心贡献

这个工作不应该被包装成“我们提出一个更强的最终学生模型”，而应该被包装成：

**我们用多教师在策略蒸馏构造了一个可控框架，系统分析视频 MLLM 中时间、空间、事件和问答能力之间的方向性促进关系。**

MOPD 的作用是让不同 teacher 的能力可以在学生自身轨迹上以较稠密、较稳定的方式同步注入。视频领域的创新不是 MOPD 本身，而是把 teacher 从“任务专家”重新定义成“能力探针”，再通过 mixed-batch ablation、leave-one-out 和 source-ratio scaling 去测量能力之间的边际促进关系。

这样最后即使 full model 只带来中等幅度提升，工作也仍然有价值，因为它能给出更深入的结论：时间能力更像召回模块，空间能力更像判别和边界精修模块，事件能力连接时空证据和语义答案，而 QA 能力本身并不保证 grounding 能力。这比简单报告一个 ST-MOPD 最终分数更像视频理解研究。
