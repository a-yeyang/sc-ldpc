# 投稿去向分析：RL 优化 SC-LDPC 构造（硕士首篇）

> 由调研型 subagent 生成（核对了各刊 scope、页数、近年规范与同类论文真实去向）。
> 指导原则：**宁可瞄准稳一点、干净落地，也别冲高换来 desk-reject 当首次投稿体验。**

## 一句话结论
这是一篇**"内含机器学习方法的有限长编码工程论文"**——不是纯 ML 方法论文，也不是香农意义的信息论
（无门限定理）。诚实的核心结论（"优化边扩展把 FER 降 5–11×、靠消 4 环干掉错误地板；RL 只在
**评估昂贵**时才稳赢随机搜索，且优势随 w 增大"）很有意思，但是**有条件、半负面**——这对 ML 顶会主轨
是毒药、对通信刊物完全 OK。**主目标：IEEE T-COM；快速首胜：IEEE Communications Letters / ISIT / ITW；
ML 社区曝光：AI4NextG workshop。不要投 TWC，也别拿 ML 主轨当首投。**

## 直接回答你的三个顾虑
1. **TWC？不要。** TWC 的 scope **明确写**：不以无线信道（衰落/干扰）为核心的、"纯 AWGN 信道的信息论/
   编码"工作**不在范围内**。你是 AWGN/PAM4 上的编码理论 → **scope desk-reject 风险**。你"更像信息论不像
   无线"的直觉是对的。
2. **AI 顶会（NeurIPS/ICML/ICLR/AAAI 主轨）？首投别去。** 审稿人会看到：方法是标准 REINFORCE（1992）、
   结论有条件/偏负面、benchmark 小众 CPU 规模 → 大概率拒。能进 ML 主轨的（KO Codes/TurboAE/DeepPolar）
   都是**新神经码架构 + 无条件碾压经典 + 资深 ML/IT 团队**。你没有这个，硬造是另一个大项目。
   **但 ML workshop 是对的门**（见下）。
3. **是 AI4Science 吗？算沾边但别靠它。** AI4Science 重心是自然科学发现（AlphaFold 式），通信/编码在边缘。
   对 ML 听众自称 AI4Science 像"贴标签"会减分。**更好的框架是"RL 做组合结构设计"（neural combinatorial
   optimization，你已引 Bello 2016）。**

## 该投哪（按现实排序）
| 刊物/会议 | 类型 | 契合 | 硕士首投现实性 | 建议 |
|---|---|---|---|---|
| **IEEE T-COM** | 期刊 | ★★★★★ | 要求高但对口 | **主旗舰目标**（先加 PEXIT 目标更稳）|
| **IEEE Comm. Letters** | 快报(≤5页) | ★★★★ | 高 | **最佳快速首胜** |
| **ISIT** | 会议(5页) | ★★★★ | 高(~50%+录取) | **强首篇会议论文**（SC-LDPC 常客）|
| **ITW** | 会议(~5页) | ★★★★ | 高；2026 有"Coding for 6G"主题 | 主题极对口 |
| **IEEE JSAIT** | 期刊(特刊) | ★★★★ | 选择性强、看时机 | 有对口特刊就投（RELDEC 期刊版去这）|
| **AI4NextG**(NeurIPS/ICML wksp) | workshop(6页) | ★★★★ | 友好、非存档 | **最佳 ML 社区门**（用下面的重构框架）|
| **ML & Compression**(NeurIPS wksp) | workshop(6页) | ★★★ | 友好、非存档 | IT 味的 ML 曝光 |
| **GLOBECOM/ICC** | 会议(6页) | ★★★ | ~35–40% | 可，曝光友好 |
| **IEEE T-IT** | 期刊 | ★★（缺定理）| 本版本偏低 | 除非补真门限理论 |
| **NeurIPS/ICML 主轨** | 会议 | ★ | 拒稿风险 | 首投**回避** |
| **IEEE T-WC** | 期刊 | ✗(scope) | desk-reject 风险 | **不要投** |

**T-IT vs T-COM vs T-WC 的关键差别**：T-IT 要**定理**（门限/证明/系综分析）；T-COM 要**面向通信链路的
方法+分析，含有限长码设计/译码**——你正落在这；T-WC 要**无线信道（衰落/干扰）为核心**——你不沾。
编码理论里"非定理"的工程结果就该去 **T-COM/Letters/ISIT/ITW**。

## 同类论文真实去向（校准期望）
- **Huang 等《AI Coding：Learning to Construct ECC》→ IEEE T-COM 2020**（华为）：**离你最近的先例，去了
  T-COM 而非 ML 会**——你主目标的最强信号。
- Zhang 等《RL 构造 LDPC》→ **WCSP 2018**（通信会议）。
- KO Codes → **ICML 2021 spotlight**、TurboAE → **NeurIPS 2019**、DeepPolar → **ICML 2024**：进 ML 主轨的
  都是**新神经码 + 碾压经典 + 顶尖 ML-IT 团队**（你不是这个模子）。
- Nachmani《Learning to Decode》→ **Allerton 2016 + IEEE JSTSP 2018**；RELDEC（RL 译码）→ **NeurIPS 2020 +
  JSAIT 2021**（资深团队、且是"译码"这种干净的"学到的策略胜经典"）。
→ **结论：很多强 ML-for-coding 工作就活在 IEEE 体系里，落在那不是安慰奖、是这个领域的主场。**

## 同一份工作、两种框架（数据不变，换叙事）
- **投通信刊（T-COM/Letters/ISIT）→ 主打码、RL 当工具**：标题打"边扩展是 5G NR SC-LDPC 里被忽视的高杠杆
  设计自由度；优化它消错误地板（5–11× FER、R≈0.6 处 ~0.9 dB 门限增益），机理是消 4 环；学到的策略可迁移、
  可解释、自动发现该规则。"**淡化 RL-vs-随机的赛马**（诚实写在一小节即可）。
- **投 ML workshop（AI4NextG）→ 主打方法论发现、码当试验台**：标题打 **"贵评估的组合设计里，RL 何时才比
  随机搜索强？——把单次评估成本当旋钮的受控研究"**。你那个"**RL 优势随单次评估成本单调上升**"恰恰是个
  **可推广、且诚实**的方法论观察，**比'RL 必胜'更有意思**（它刻画了"何时该用这方法"）。**把负面结果当卖点。**

## 硕士首投的现实路线（~12 个月）
0. **现在就挂 arXiv**（最高性价比：给你的"首次把 SC-LDPC 边扩展建成 MDP"打时间戳防被抢；IEEE 允许预印本。
   注意 cs.IT/eess.SP 首次投稿可能需 endorsement，找导师/合作者背书）。
1. **投 Comm. Letters 和/或 下一个 ISIT/ITW 截止**——短、录取率高、审稿人懂编码（看得懂"消 4 环"）。
2. **同时投一个 workshop（AI4NextG）**做 ML 曝光（非存档、不影响以后投 T-COM）。
3. **补 PEXIT/密度演化门限目标 + 一个"同列边↔4环/girth"的小组合引理**，写成完整 **T-COM**。
4. 投 T-COM（或赶上对口特刊就投 JSAIT）。

## 最高杠杆的改进（按对"你"的性价比排序）
1. **把奖励从纯 Monte-Carlo BER 换/加成 DE/PEXIT 门限**——一举三得：补上你最大的诚实短板（"只有蒙卡、
   无门限理论"）、提供一个**昂贵但有原理**的目标（正是你"评估贵→RL 赢"论点最锋利的地方）、把论文推向
   T-IT/JSAIT 的可信度。你已引 Liva–Olmos PEXIT，接上去很自然。
2. **加一点理论框架**：不用完整门限定理，但把"同列边散开"规则与提升 QC 图的 4 环/girth 计数联系起来的
   一个小组合引理，能显著抬高天花板、消解"只是仿真"的质疑。
3. **规模是低杠杆**（对通信刊）：Z≤128、w≤30 对 T-COM/Letters/ISIT **足够**，别为通信投稿去刷 Z=256 GPU。
4. **加一段标准相关性**（绑 3GPP TS 38.212 BG1/BG2 + 6G 边扩展 ESRL），便宜地买"relevance"，不必做硬件。
→ 对通信刊：做 **(1)+(2)**，别做 (3)/(4)。

## 主要来源
TWC scope / T-COM 编委（含"AI for source and channel coding"编辑）/ T-IT scope / Comm. Letters / JSAIT /
ISIT / ITW 2026 / AI4NextG / ML&Compression / AI Coding(T-COM) / KO Codes(ICML) / RELDEC(NeurIPS+JSAIT)。
