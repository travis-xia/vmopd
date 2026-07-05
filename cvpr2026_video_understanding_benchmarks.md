# CVPR 2026 Video Understanding Benchmark Papers

数据源：[CVPR 2026 Open Access Repository, day=all](https://openaccess.thecvf.com/CVPR2026?day=all)

检索日期：2026-06-28

筛选过程：

- CVF `day=all` 页面共解析到 4068 篇论文。
- 标题包含 `bench/benchmark` 的论文共 162 篇。
- 在此基础上，用 `video / long video / temporal / spatio-temporal / action / activity / gesture / egocentric / audio-visual / sound / deepfake / forgery / RVOS / tracking / segmentation / QA / question-answering / dataset / evaluation` 做补漏。
- 逐篇查看候选论文摘要，保留提出 benchmark、dataset、evaluation suite，且任务属于视频理解、视频问答、长视频推理、时序/动作/行为理解、音视频理解、视频 grounding/segmentation/tracking 或视频伪造检测的论文。
- 排除：纯视频生成质量评估、视频编辑/压缩/重建/去模糊、SfM/pose/NVS 为主且不以语义理解为核心的 benchmark。

## 主清单

### VideoQA / long-video reasoning / MLLM video understanding

- [MovieRecapsQA: A Multimodal Open-Ended Video Question-Answering Benchmark](https://openaccess.thecvf.com/content/CVPR2026/html/Shaar_MovieRecapsQA_A_Multimodal_Open-Ended_Video_Question-Answering_Benchmark_CVPR_2026_paper.html)：开放式多模态 VideoQA benchmark。基于电影 recap 视频，包含 8.2K 问题，强调视觉和对白联合推理，以及 reference-free 评测。
- [AV-Reasoner: Improving and Benchmarking Clue-Grounded Audio-Visual Counting for MLLMs](https://openaccess.thecvf.com/content/CVPR2026/html/Lu_AV-Reasoner_Improving_and_Benchmarking_Clue-Grounded_Audio-Visual_Counting_for_MLLMs_CVPR_2026_paper.html)：提出 CG-AV-Counting，面向长视频音视频计数、线索 grounding 和 MLLM 推理评测。
- [MA-Bench: Towards Fine-grained Micro-Action Understanding](https://openaccess.thecvf.com/content/CVPR2026/html/Li_MA-Bench_Towards_Fine-grained_Micro-Action_Understanding_CVPR_2026_paper.html)：细粒度 micro-action 理解 benchmark，包含 1K 视频和 12K QA，评测感知、关系理解和解释推理。
- [CURVE: A Benchmark for Cultural and Multilingual Long Video Reasoning](https://openaccess.thecvf.com/content/CVPR2026/html/Singh_CURVE_A_Benchmark_for_Cultural_and_Multilingual_Long_Video_Reasoning_CVPR_2026_paper.html)：多文化、多语言长视频推理 benchmark，用于缓解现有长视频评测的西方中心和英语偏置。
- [HanDyVQA: A Video QA Benchmark for Fine-Grained Hand-Object Interaction Dynamics](https://openaccess.thecvf.com/content/CVPR2026/html/Tateno_HanDyVQA_A_Video_QA_Benchmark_for_Fine-Grained_Hand-Object_Interaction_Dynamics_CVPR_2026_paper.html)：面向手物交互动态的细粒度 VideoQA，强调操作、对象状态变化和时空因果。
- [ELV-Halluc: Benchmarking Semantic Aggregation Hallucinations in Video Understanding](https://openaccess.thecvf.com/content/CVPR2026/html/Lu_ELV-Halluc_Benchmarking_Semantic_Aggregation_Hallucinations_in_Video_Understanding_CVPR_2026_paper.html)：评测 Video-MLLM 从帧级语义聚合到事件级解释时产生的 hallucination。
- [HumanVBench: Probing Human-Centric Video Understanding in MLLMs with Automatically Synthesized Benchmarks](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_HumanVBench_Probing_Human-Centric_Video_Understanding_in_MLLMs_with_Automatically_Synthesized_CVPR_2026_paper.html)：人中心视频理解 benchmark，覆盖情绪、行为、跨模态对齐等维度。
- [VirtueBench: Evaluating Trustworthiness under Uncertainty in Long Video Understanding](https://openaccess.thecvf.com/content/CVPR2026/html/Yu_VirtueBench_Evaluating_Trustworthiness_under_Uncertainty_in_Long_Video_Understanding_CVPR_2026_paper.html)：长视频理解中的不确定性与可信拒答评测，区分 answerable / unanswerable 场景。
- [SurgCoT: Advancing Spatiotemporal Reasoning in Surgical Videos through a Chain-of-Thought Benchmark](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_SurgCoT_Advancing_Spatiotemporal_Reasoning_in_Surgical_Videos_through_a_Chain-of-Thought_CVPR_2026_paper.html)：手术视频 CoT 推理 benchmark，覆盖因果动作排序、cue-action 对齐、异常开始追踪等。
- [VRR-QA: Visual Relational Reasoning in Videos Beyond Explicit Cues](https://openaccess.thecvf.com/content/CVPR2026/html/Swetha_VRR-QA_Visual_Relational_Reasoning_in_Videos_Beyond_Explicit_Cues_CVPR_2026_paper.html)：视频关系推理 QA，要求模型推断画面中未直接显式呈现的关系和上下文。
- [CaST-Bench: Benchmarking Causal Chain-Grounded Spatio-Temporal Reasoning for Video Question Answering](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_CaST-Bench_Benchmarking_Causal_Chain-Grounded_Spatio-Temporal_Reasoning_for_Video_Question_Answering_CVPR_2026_paper.html)：因果链 grounding 的时空 VideoQA benchmark，带 temporal segments 和 bounding-box tracks 证据标注。
- [Seeing the Scene Matters: Revealing Forgetting in Video Understanding Models with a Scene-Aware Long-Video Benchmark](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Seeing_the_Scene_Matters_Revealing_Forgetting_in_Video_Understanding_Models_CVPR_2026_paper.html)：提出 SceneBench，评测长视频场景级上下文记忆和 scene-level question answering。
- [HERBench: A Benchmark for Multi-Evidence Integration in Video Question Answering](https://openaccess.thecvf.com/content/CVPR2026/html/Ami_HERBench_A_Benchmark_for_Multi-Evidence_Integration_in_Video_Question_Answering_CVPR_2026_paper.html)：多证据整合 VideoQA benchmark，每个问题需要跨至少三个不重叠视频片段整合信息。
- [Flat-Pack Bench: Evaluating Spatio-Temporal Understanding in Large Vision-Language Models through Furniture Assembly](https://openaccess.thecvf.com/content/CVPR2026/html/Chetan_Flat-Pack_Bench_Evaluating_Spatio-Temporal_Understanding_in_Large_Vision-Language_Models_through_CVPR_2026_paper.html)：通过家具组装任务评测细粒度时空理解，包括步骤排序、状态定位、部件配合和 tracking。
- [FPS-Bench: A Benchmark for High Frame-Rate Video Understanding](https://openaccess.thecvf.com/content/CVPR2026/html/Choudhury_FPS-Bench_A_Benchmark_for_High_Frame-Rate_Video_Understanding_CVPR_2026_paper.html)：高帧率视频理解 benchmark，引入 `minFPS` 指标，评测低 FPS 输入下模型遗漏快速事件的问题。
- [UniVBench: Towards Unified Evaluation for Video Foundation Models](https://openaccess.thecvf.com/content/CVPR2026/html/Wei_UniVBench_Towards_Unified_Evaluation_for_Video_Foundation_Models_CVPR_2026_paper.html)：统一评测 video foundation models 的 benchmark，包含 video understanding，也包含 generation/editing/reconstruction，因此属于混合型相关项。

### Egocentric / assistive / embodied video understanding

- [Ego-Grounding for Personalized Question-Answering in Egocentric Videos](https://openaccess.thecvf.com/content/CVPR2026/html/Xiao_Ego-Grounding_for_Personalized_Question-Answering_in_Egocentric_Videos_CVPR_2026_paper.html)：提出 MyEgo，评测 egocentric VideoQA 中对 camera wearer 的理解、记忆和推理。
- [Do You See What I Am Pointing At? Gesture-Based Egocentric Video Question Answering](https://openaccess.thecvf.com/content/CVPR2026/html/Choi_Do_You_See_What_I_Am_Pointing_At_Gesture-Based_Egocentric_CVPR_2026_paper.html)：提出 EgoPointVQA，面向指向手势 grounding 的第一视角视频问答。
- [Ego2Web: A Web Agent Benchmark Grounded in Egocentric Videos](https://openaccess.thecvf.com/content/CVPR2026/html/Yu_Ego2Web_A_Web_Agent_Benchmark_Grounded_in_Egocentric_Videos_CVPR_2026_paper.html)：把第一视角视频感知和网页任务执行结合，评测 agent 是否能看懂现实环境并完成线上任务。
- [LifeEval: A Multimodal Benchmark for Assistive AI in Egocentric Daily Life Tasks](https://openaccess.thecvf.com/content/CVPR2026/html/Gao_LifeEval_A_Multimodal_Benchmark_for_Assistive_AI_in_Egocentric_Daily_CVPR_2026_paper.html)：第一视角日常生活 assistive AI benchmark，强调实时、任务导向的人机协作。

### Action / gesture / behavior / temporal localization

- [DarkAct: A RGB-Thermal Dataset and Fusion Framework for Multimodal Low-Light Action Recognition](https://openaccess.thecvf.com/content/CVPR2026/html/Tan_DarkAct_A_RGB-Thermal_Dataset_and_Fusion_Framework_for_Multimodal_Low-Light_CVPR_2026_paper.html)：低光 RGB-Thermal 行为识别视频数据集和 benchmark，包含 12,778 对 RGB-thermal 视频。
- [VideoNet: A Large-Scale Dataset for Domain-Specific Action Recognition](https://openaccess.thecvf.com/content/CVPR2026/html/Yadav_VideoNet_A_Large-Scale_Dataset_for_Domain-Specific_Action_Recognition_CVPR_2026_paper.html)：大规模 domain-specific action recognition benchmark，覆盖 38 个领域、1,087 类动作。
- [OpenMarcie: Dataset for Multimodal Action Recognition in Industrial Environments](https://openaccess.thecvf.com/content/CVPR2026/html/Bello_OpenMarcie_Dataset_for_Multimodal_Action_Recognition_in_Industrial_Environments_CVPR_2026_paper.html)：工业环境多模态动作识别数据集，包含 egocentric/exocentric 视频和多种传感器，benchmark 三类 human activity recognition 任务。
- [MooCap: A Multi-View Benchmark for Cow-Object-Human Interaction and Behavior Dynamics](https://openaccess.thecvf.com/content/CVPR2026/html/Noronha_MooCap_A_Multi-View_Benchmark_for_Cow-Object-Human_Interaction_and_Behavior_Dynamics_CVPR_2026_paper.html)：多视角动物-对象-人交互与行为动态 benchmark，包含同步多相机视频和细粒度行为标注。
- [OmniVTG: A Large-Scale Dataset and Training Paradigm for Open-World Video Temporal Grounding](https://openaccess.thecvf.com/content/CVPR2026/html/Zheng_OmniVTG_A_Large-Scale_Dataset_and_Training_Paradigm_for_Open-World_Video_CVPR_2026_paper.html)：开放世界 video temporal grounding 数据集，面向文本查询定位视频片段。
- [OmniGround: A Comprehensive Spatio-Temporal Grounding Benchmark for Real-World Complex Scenarios](https://openaccess.thecvf.com/content/CVPR2026/html/Gao_OmniGround_A_Comprehensive_Spatio-Temporal_Grounding_Benchmark_for_Real-World_Complex_Scenarios_CVPR_2026_paper.html)：真实复杂场景下的 spatio-temporal video grounding benchmark，覆盖多对象类别和复杂自然语言查询。
- [Long-RVOS: A Comprehensive Benchmark for Long-term Referring Video Object Segmentation](https://openaccess.thecvf.com/content/CVPR2026/html/Liang_Long-RVOS_A_Comprehensive_Benchmark_for_Long-term_Referring_Video_Object_Segmentation_CVPR_2026_paper.html)：长视频 referring video object segmentation benchmark，强调遮挡、消失重现、镜头切换和时空一致性。
- [OMG-Bench: A New Challenging Benchmark for Skeleton-based Online Micro Hand Gesture Recognition](https://openaccess.thecvf.com/content/CVPR2026/html/Chang_OMG-Bench_A_New_Challenging_Benchmark_for_Skeleton-based_Online_Micro_Hand_CVPR_2026_paper.html)：基于手部骨架的在线 micro hand gesture recognition benchmark，覆盖细微动作、快速动态和连续执行。
- [SHands: A Multi-View Dataset and Benchmark for Surgical Hand-Gesture and Error Recognition Toward Medical Training](https://openaccess.thecvf.com/content/CVPR2026/html/Ma_SHands_A_Multi-View_Dataset_and_Benchmark_for_Surgical_Hand-Gesture_and_CVPR_2026_paper.html)：手术训练场景下的多视角视频数据集和 benchmark，评测手势识别与错误识别。
- [Breaking Smooth-Motion Assumptions: A UAV Benchmark for Multi-Object Tracking in Complex and Adverse Conditions](https://openaccess.thecvf.com/content/CVPR2026/html/Ye_Breaking_Smooth-Motion_Assumptions_A_UAV_Benchmark_for_Multi-Object_Tracking_in_CVPR_2026_paper.html)：DynUAV，复杂 UAV 视角多目标跟踪 benchmark。偏视频感知/跟踪，不是 VLM 语义推理，但属于视频理解相关基础任务。

### Audio-video / speech-video understanding

- [HAVE-Bench: Hierarchical Audio-Visual Evaluation from Perception to Interaction](https://openaccess.thecvf.com/content/CVPR2026/html/Zhong_HAVE-Bench_Hierarchical_Audio-Visual_Evaluation_from_Perception_to_Interaction_CVPR_2026_paper.html)：分层音视频评测 benchmark，覆盖 perception、reasoning、interaction 三层能力。
- [FAVE: A Structured Benchmark for Fine-Grained Audio-Visual Temporal Evaluation in Multimodal LLMs](https://openaccess.thecvf.com/content/CVPR2026/html/Lu_FAVE_A_Structured_Benchmark_for_Fine-Grained_Audio-Visual_Temporal_Evaluation_in_CVPR_2026_paper.html)：细粒度 audio-visual temporal evaluation，评测音频和视觉流之间的时间关系理解。
- [AMusE: Audio-Visual Benchmark and Alignment Framework for Agentic Multi-Speaker Understanding](https://openaccess.thecvf.com/content/CVPR2026/html/Chowdhury_AMusE_Audio-Visual_Benchmark_and_Alignment_Framework_for_Agentic_Multi-Speaker_Understanding_CVPR_2026_paper.html)：多说话人音视频理解 benchmark，包含 speaker grounding、角色跟踪、对话总结等 agentic 任务。
- [EgoSound: Benchmarking Sound Understanding in Egocentric Videos](https://openaccess.thecvf.com/content/CVPR2026/html/Zhu_EgoSound_Benchmarking_Sound_Understanding_in_Egocentric_Videos_CVPR_2026_paper.html)：第一视角视频中的声音理解 benchmark，覆盖声音感知、空间定位、因果推理和跨模态推理。
- [SVHalluc: Benchmarking Speech-Vision Hallucination in Audio-Visual Large Language Models](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_SVHalluc_Benchmarking_Speech-Vision_Hallucination_in_Audio-Visual_Large_Language_Models_CVPR_2026_paper.html)：评测 audio-visual LLM 中 speech content 与视觉信号不对齐导致的 hallucination，包含语义和时间两个方面。

### Video forensics / deepfake / generated-video detection

- [ActivityForensics: A Comprehensive Benchmark for Localizing Manipulated Activity in Videos](https://openaccess.thecvf.com/content/CVPR2026/html/Bao_ActivityForensics_A_Comprehensive_Benchmark_for_Localizing_Manipulated_Activity_in_Videos_CVPR_2026_paper.html)：活动级视频篡改定位 benchmark，关注改变人类动作导致事件语义扭曲的伪造。
- [AVFakeBench: A Comprehensive Audio-Video Forgery Detection Benchmark for AV-LMMs](https://openaccess.thecvf.com/content/CVPR2026/html/Xia_AVFakeBench_A_Comprehensive_Audio-Video_Forgery_Detection_Benchmark_for_AV-LMMs_CVPR_2026_paper.html)：音视频伪造检测 benchmark，覆盖人类主体和一般主体，多任务评测二分类、类型分类、定位和解释。
- [FVBench: Benchmarking Deepfake Video Detection Capability of Large Multimodal Models](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_FVBench_Benchmarking_Deepfake_Video_Detection_Capability_of_Large_Multimodal_Models_CVPR_2026_paper.html)：面向 LMM 的 deepfake video detection benchmark，包含真实、AI 编辑和全 AI 生成视频。
- [CoCoVideo: The High-Quality Commercial-Model-Based Contrastive Benchmark for AI-Generated Video Detection](https://openaccess.thecvf.com/content/CVPR2026/html/Feng_CoCoVideo_The_High-Quality_Commercial-Model-Based_Contrastive_Benchmark_for_AI-Generated_Video_Detection_CVPR_2026_paper.html)：商业模型生成视频检测 benchmark，提供真实-伪造语义对齐视频对。
- [Omni-Fake: Benchmarking Unified Multimodal Social Media Deepfake Detection](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Omni-Fake_Benchmarking_Unified_Multimodal_Social_Media_Deepfake_Detection_CVPR_2026_paper.html)：社交媒体多模态 deepfake benchmark，覆盖 image、audio、video、audio-video talking head，并支持 detection-localization-explanation。
- [TriDF: Evaluating Perception, Detection, and Hallucination for Interpretable DeepFake Detection](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang-Lin_TriDF_Evaluating_Perception_Detection_and_Hallucination_for_Interpretable_DeepFake_Detection_CVPR_2026_paper.html)：可解释 deepfake 检测 benchmark，覆盖 image、video、audio 三种模态，评测 perception、detection、hallucination。
- [DeepfakeImpact: A Two-Stage Benchmark with Real-World Impact in Deepfake Detection](https://openaccess.thecvf.com/content/CVPR2026/html/Gong_DeepfakeImpact_A_Two-Stage_Benchmark_with_Real-World_Impact_in_Deepfake_Detection_CVPR_2026_paper.html)：面向真实社会影响的 deepfake detection benchmark，引入 Social Misjudgment Impact 评价误判风险。

## 边界但建议关注

- [EgoProx: Evaluating MLLMs on Egocentric 3D Proximity Reasoning Across a Cognitive Hierarchy](https://openaccess.thecvf.com/content/CVPR2026/html/Li_EgoProx_Evaluating_MLLMs_on_Egocentric_3D_Proximity_Reasoning_Across_a_CVPR_2026_paper.html)：egocentric 3D proximity reasoning benchmark，偏具身/空间推理，摘要未明确以视频序列为核心，但和第一视角理解很接近。
- [Ego-1K - A Large-Scale Multiview Video Dataset for Egocentric Vision](https://openaccess.thecvf.com/content/CVPR2026/html/Lee_Ego-1K_-_A_Large-Scale_Multiview_Video_Dataset_for_Egocentric_Vision_CVPR_2026_paper.html)：第一视角多视角视频数据集，支持 dynamic scene understanding，但主要实验偏 3D/4D novel view synthesis 和场景重建。

## 搜到但未收录的典型项

这些论文标题含 `video/bench/evaluation/dataset` 等关键词，但不按“视频理解 benchmark”主清单收录：

- [ORBIT: Benchmarking SfM in the Wild with 360deg Video](https://openaccess.thecvf.com/content/CVPR2026/html/Sabour_ORBIT_Benchmarking_SfM_in_the_Wild_with_360deg_Video_CVPR_2026_paper.html)：主要是 SfM/camera pose estimation。
- [VideoRealBench: A Chain-of-Thought Realism Evaluation Benchmark for Generated Human-Centric Videos](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_VideoRealBench_A_Chain-of-Thought_Realism_Evaluation_Benchmark_for_Generated_Human-Centric_Videos_CVPR_2026_paper.html)：主要是生成视频真实感评测。
- [VGA-Bench: A Unified Benchmark and Multi-Model Framework for Video Aesthetics and Generation Quality Evaluation](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_VGA-Bench_A_Unified_Benchmark_and_Multi-Model_Framework_for_Video_Aesthetics_CVPR_2026_paper.html)：主要是视频美学和生成质量评测。
- [SLVMEval: Synthetic Meta Evaluation Benchmark for Text-to-Long Video Generation](https://openaccess.thecvf.com/content/CVPR2026/html/Matsuda_SLVMEval_Synthetic_Meta_Evaluation_Benchmark_for_Text-to-Long_Video_Generation_CVPR_2026_paper.html)：text-to-long-video generation 评测。
- [SVBench: Evaluation of Video Generation Models on Social Reasoning](https://openaccess.thecvf.com/content/CVPR2026/html/Peng_SVBench_Evaluation_of_Video_Generation_Models_on_Social_Reasoning_CVPR_2026_paper.html)：评测视频生成模型，不是理解模型 benchmark。
- [VABench: A Comprehensive Benchmark for Audio-Video Generation](https://openaccess.thecvf.com/content/CVPR2026/html/Hua_VABench_A_Comprehensive_Benchmark_for_Audio-Video_Generation_CVPR_2026_paper.html)：audio-video generation benchmark。
- [Ref4D-VideoBench: Four-Dimensional Reference-Based Evaluation of Text-to-Video Generative Models](https://openaccess.thecvf.com/content/CVPR2026/html/Wei_Ref4D-VideoBench_Four-Dimensional_Reference-Based_Evaluation_of_Text-to-Video_Generative_Models_CVPR_2026_paper.html)：text-to-video generative model 评测。
- [TiViBench: Benchmarking Think-in-Video Reasoning for Video Generation](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_TiViBench_Benchmarking_Think-in-Video_Reasoning_for_Video_Generation_CVPR_2026_paper.html)：video generation 方向，虽有 reasoning 字样但核心是生成。
- [MotionEdit: Benchmarking and Learning Motion-Centric Image Editing](https://openaccess.thecvf.com/content/CVPR2026/html/Wan_MotionEdit_Benchmarking_and_Learning_Motion-Centric_Image_Editing_CVPR_2026_paper.html)：image editing，不是视频理解。
- [EgoEdit: Dataset, Real-Time Streaming Model, and Benchmark for Egocentric Video Editing](https://openaccess.thecvf.com/content/CVPR2026/html/Li_EgoEdit_Dataset_Real-Time_Streaming_Model_and_Benchmark_for_Egocentric_Video_CVPR_2026_paper.html)：egocentric video editing，不是理解 benchmark。
- [SIMSPINE: A Biomechanics-Aware Simulation Framework for 3D Spine Motion Annotation and Benchmarking](https://openaccess.thecvf.com/content/CVPR2026/html/Khan_SIMSPINE_A_Biomechanics-Aware_Simulation_Framework_for_3D_Spine_Motion_Annotation_CVPR_2026_paper.html)：3D spine motion estimation / biomechanics benchmark，偏 pose/motion annotation。
- [EgoXtreme: A Dataset for Robust Object Pose Estimation in Egocentric Views under Extreme Conditions](https://openaccess.thecvf.com/content/CVPR2026/html/Yoon_EgoXtreme_A_Dataset_for_Robust_Object_Pose_Estimation_in_Egocentric_CVPR_2026_paper.html)：egocentric object pose estimation，偏 6D pose。
