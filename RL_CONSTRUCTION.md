# 用强化学习做空间耦合 LDPC 的端到端构造优化

> 研究记录 / 方法说明。配套实现：[rl_construct.py](rl_construct.py)（构造 MDP 环境 +
> 策略梯度 REINFORCE + CEM/随机搜索基线 + 短环分析，纯 NumPy + multiprocessing）、
> [experiments_construct.py](experiments_construct.py)（搜索 / BER 曲线 / 机理分析 /
> 迁移实验与出图）、[tests_construct.py](tests_construct.py)（含两种策略梯度的有限差分校验）。
> 本工作建立在本仓库已有的 5G NR → SC-LDPC 收发链之上（见 [README.md](README.md)），
> 与已有的"**RL 译码**"工作（[RL_SC_LDPC.md](RL_SC_LDPC.md)）相互独立、互为补充：
> 那里学的是**怎么译**一张固定的码，这里学的是**怎么造**这张码。

---

## 0. 一句话

> **空间耦合 LDPC 的"边扩展（edge spreading）"——把 5G NR 基图的系统位边分配到 w+1 个
> 分量矩阵 B_0…B_w——是一个被长期随机化处理的关键构造自由度；我们第一次把这个构造过程
> 建成马尔可夫决策过程（MDP），用策略梯度（REINFORCE）以"真实收发链的误帧率"为奖励，
> 端到端地学出比随机/启发式更好、且能零样本迁移到新码的边扩展策略，并自动发现了
> "同列边要散开（避免 4 环）"这一可解释规则。**

这条思路与学姐说的"BP 译码是马尔可夫过程、可用 RL 优化"一脉相承，但落点不同：我们优化的不是
译码调度，而是**码的构造本身**。在本仓库里，构造的旋钮就是 `SCLDPCCode._edge_spread` 里那行
`rng.integers(0, w+1, size=E)`——一个被随便扔给随机数的设计决策。实测表明：**仅这一个随机选择，
就能让同一工作点的误帧率（FER）相差约 7 倍**（见 §4.1 头部空间探针）。

---

## 1. 文献调研与新颖性

把"机器学习 / 强化学习 + LDPC"按"译码 vs 构造"和"分组码 vs 空间耦合"两条轴梳理：

### ① RL 用于 LDPC【译码】—— 已成熟
- Habib, Beemer, Kliewer，*Learning to Decode: RL for Decoding of Sparse Graph-Based
  Channel Codes*，NeurIPS 2020；其期刊版 **RELDEC**（IEEE TCOM 2023, arXiv 2112.13934）把
  校验节点（簇）调度建成 MDP，用 Q-learning 学序贯调度，并在 5G NR LDPC 上验证。
- 本仓库 [RL_SC_LDPC.md](RL_SC_LDPC.md) 把 SC-LDPC 的**滑窗译码**建成 MDP。
- **共性：都在"如何译码"，码本身是给定的。**

### ② AI / RL 用于【码的构造】—— 有，但不在空间耦合上
- Huang, Zhang, Li, Ge, Wang（华为），*AI Coding: Learning to Construct Error
  Correction Codes*，IEEE TCOM 2020（arXiv 1901.05719）。提出"**构造器–评估器**"
  框架：构造器（可用 **RL** 或遗传算法）提码、评估器回报性能（最小距离 / 门限 / BER），
  迭代优化。**对象是线性分组码 / 极化码，不是 LDPC、更不是 SC-LDPC。** 这是"RL 造码"最
  正统的奠基参考。
- Zhang, Huang, Wang, Wang，*Construction of LDPC Codes Based on Deep RL*，WCSP 2018。
  用策略/价值网络 + 蒙特卡洛树搜索（AlphaZero 式）学 **PEG 式逐边生长**的路径。**对象是
  LDPC 分组码（逐边生长），不是空间耦合 / 边扩展。**
- Tian et al.，*GNN-based Auto-Encoder for Short Linear Block Codes: A DRL Approach*，
  2024（arXiv 2412.02053）。把**校验矩阵生成建成 MDP**，DRL+GNN 联合学码与译码器，奖励含
  girth/BER/最小距离。**对象是短分组码，全文未涉及空间耦合 / 边扩展。**

### ③ SC-LDPC 的【边扩展 / 划分矩阵】优化 —— 有，但都是非学习的搜索
- Mitchell, Lentmaier, Costello，*Spatially Coupled LDPC Codes Constructed from
  Protographs*，IEEE T-IT 2015（arXiv 1407.5366）。**定义了"边扩展规则"**，并明确指出
  *"不同的边扩展会同时影响迭代 BP 译码性能与距离特性"*（其 Table III 直接对比了两种边扩展
  的 BEC 门限与距离增长率）。这是"边扩展是设计自由度、且选择很重要"的权威出处。
- Mitchell, Rosnes（ISIT 2017）、Battaglioni, Baldi, Chiaraluce, Mitchell 等
  （arXiv 1904.07158 / EURASIP 2023）：把 SC-LDPC 设计转化为**"找好的扩展/划分矩阵"**，
  用**贪心**消除有害对象（陷阱集 / 吸收集 / 短环）。
- Esfahanizadeh / Hareedy / Dolecek 一系（arXiv 2203.02052, 2022；2509.21112, 2025）：
  把**划分矩阵**当成核心设计对象，**穷举遍历 + 剪枝**联合优化渐近门限与有限长环计数。其 2025
  论文直言这类离散优化"**当存在更多自由度时，复杂度上会力不从心**"——这正是 RL/策略梯度要补的位。
- Ren et al.，*Edge-Spreading Raptor-Like (ESRL) LDPC for 6G*，2024（arXiv 2410.16875）：
  用"统一图"表示直接构造最优耦合矩阵，**纯图论优化、无 ML**。
- Tanrıkulu, Yıldırım, Hareedy，*MCMC for Efficient Finite-Length LDPC Code Design*，
  2025（arXiv 2504.16071）：**MCMC**（非 RL）在有限长构造空间上采样优化环/陷阱集。

### → 空白点（本课题的新颖性）

| | 分组 LDPC | 空间耦合 LDPC（边扩展） |
|---|---|---|
| 译码：启发式 / 神经 / RL | ✓ / ✓ / ✓ | ✓ / ✓(NWD) / ✓(本仓库 RL_SC_LDPC) |
| 构造：图论/贪心/穷举/MCMC | ✓ | ✓（Mitchell/Battaglioni/Esfahanizadeh/ESRL/MCMC）|
| **构造：强化学习 / MDP** | ✓ (AI Coding, WCSP'18) | **✗ ← 本工作** |

> 经约 8 组不同措辞的检索 + 对最接近论文的全文核对：**没有任何已发表工作把 SC-LDPC 的构造 /
> 边扩展建成 MDP 并用强化学习优化。** 现有 SC-LDPC 构造方法全是贪心 / 穷举 / 图论 / MCMC。
> RL 造码只在**分组码**上做过（AI Coding 2020、WCSP 2018）。本工作把"策略梯度 RL 造码"
> **第一次**带到空间耦合的边扩展上，并直接用**端到端 BER/FER**当奖励。

为什么 SC-LDPC 的构造特别适合 RL：边扩展是一个**天然序贯**的分配过程（逐条边决定去哪个分量），
其质量由协议图 / 提升图的**环结构**主导，而环结构对每条边的分配高度敏感——这正是策略梯度可以
顺着结构梯度爬升、而盲目随机搜索难以高效覆盖（搜索空间 (w+1)^E ≈ 3^40 ≈ 10^19）的地方。

---

## 2. 把"构造"建成 MDP

记分量码（5G NR 基图截断）的系统位边集合为 `{e_0,…,e_{E-1}}`（按行优先的固定顺序），耦合记忆
`w`，则每条系统位边要被分配到分量 `B_0…B_w` 之一（变量位置 t 经 `B_a` 连到校验位置 `t+a`）。
我们按这个固定顺序逐条边地"造码"，把它建成一个 MDP：

- **回合（episode）**：造一整张码——依次给 E 条边各选一个分量，然后**端到端评估**这张码。
- **时间步 t**：给第 `e_t` 条边选分量 `a_t ∈ {0,…,w}`。
- **状态 `s_t`（只与"已造的部分"有关，是真正的序贯 MDP）**：第 `e_t` 条边在**当前部分构造**下的特征——

  | 特征（对每个候选分量 a 各算一份） | 含义 |
  |---|---|
  | `row_frac[a]` | 该边所在**基图行**已分到分量 a 的边的占比（行内负载） |
  | `col_frac[a]` | 该边所在**基图列**已分到分量 a 的边的占比（列内负载） |
  | `glob_frac[a]` | 分量 a 的**全局**负载占比 |
  | `row_empty[a]` | 分量 a 对该行是否还"空着"（0/1，控制行内聚集/散开的尖锐旋钮） |
  | `col_empty[a]` | 分量 a 对该列是否还"空着"（0/1，控制列内聚集/散开） |

  每放一条边都会更新这些计数，从而**改变后续边看到的状态**——这就是序贯性的来源，也让策略能学到
  **结构性规则**（如"把同一列的边散到不同分量"，正是避免短环的关键）。
- **动作 `a_t`**：分量索引 `∈ {0,…,w}`。
- **奖励**：中间步奖励为 0，回合末给一个**端到端终止奖励** `R = -FER`——把这张码放进
  *真实的* `编码 → BPSK+AWGN → 滑窗 BP 译码 → 误帧率` 链路，在训练 SNR 下用一批
  **公共随机数（CRN）**帧测得的 FER。真值只用于"给造好的码打分"（设计阶段），学到的**构造
  （即 assign 向量）本身不含任何真值**，可直接部署——这与所有码设计一样。

最大化 `E[R]` 就是 **REINFORCE**（Williams 1992；Bello 2016 的"神经组合优化"范式）：对因子化
策略 `π_θ(a) = Π_t π(a_t|s_t)`，有
```
∇J = E[ (R − b) · Σ_t ∇ log π(a_t|s_t) ] .
```
其中 `b` 是**批内基线**（batch 内 R 的均值）。再叠加**公共随机数**——一个 batch 里所有候选码都用
*同一批帧*打分——优势 `R_i − b` 的方差极小，于是即便每张码只评很少的帧也能稳定学习。这正是
本方法能在 CPU 上跑起来的关键。

### 2.1 两种策略（都用上面的 REINFORCE）
1. **特征策略（FeaturePolicy，正式方法）**：对上面 `C+5` 个"每分量"特征做线性 softmax，参数只有
   `C=w+1` 个分量偏置 + 5 个共享权重 = **8 个数，与码的规模无关**。因此学到的策略可以**零样本迁移**
   到不同 `Z / L / 码率` 的码上（只要 `w` 相同）。这是真正的序贯构造 MDP。
2. **逐边 logits 策略（PerEdgePolicy，消融）**：每条边一套独立 logits（`E×C` 个参数）。最具表达力，
   但**不能迁移**（每个具体边一行 logits），用作"是否需要泛化结构"的对照。

### 2.2 两个非 RL 基线（同等评估预算）
- **随机搜索（关键基线）**：每步采样一批均匀随机 assign，保留验证 FER 最优者。这是衡量"RL 是否
  真的比盲搜更省样本"的标尺。
- **CEM（交叉熵方法）**：维护逐边类别分布，每步取精英更新分布。一个强力的分布式（非 RL）优化器。

**公平性**：四种方法（RL-特征、RL-逐边、CEM、随机搜索）**共用同一评估器、同一评估预算
（步数×批大小）、同一验证流程**（固定验证帧、最后再用大帧库高精度复测冠军），任何差异都只来自
搜索策略本身。

---

## 3. 算法

纯 NumPy 实现（契合本仓库零依赖风格），多进程并行评估（构造质量评估是唯一的计算瓶颈）：

- **评估器**：给定 assign → 用本仓库的 5G NR SC 编码器逐位置编码 → BPSK+AWGN（CRN）→ 滑窗 BP →
  误码/误帧统计。掩码（打孔/终止/发送位）只与 `L,w,Kb,nb,Z` 有关、**与 assign 无关**，因此可对所有
  候选码施加**完全相同的信息比特与噪声**（CRN），评估器经测试对此**逐比特确定**。
- **策略梯度**：Adam 优化器（纯 NumPy），优势做批内标准化，叠加熵正则鼓励探索。两种策略的
  `∇ log π` 都用**有限差分校验**过（[tests_construct.py](tests_construct.py)）。
- **并行**：用 `multiprocessing.Pool` 把"每张码 × 帧块"切成任务铺满全部 CPU 核；BLAS 线程绑定为 1
  避免过度并发。冠军的高精度复测也按帧切块并行。

---

## 4. 实验设计

### 4.1 为什么这个问题值得做（头部空间）

工作点 `BG2 / ils0 / Z=16 / mp=8`（分量码率≈0.625，耦合后 R≈0.603）、`L=30, w=2`、滑窗 `W=6`、
训练 SNR=2.5 dB。系统位边 `E=40`，构造空间 `3^40 ≈ 10^19`。

用 CRN 探针（同一批帧、只换构造）测 14 个**随机**边扩展的 FER：**0.106 ~ 0.725（约 7 倍跨度）**，
BER 跨度约 8.5 倍——**单这一个随机选择就决定了码好坏的量级**。这说明（a）构造确有巨大头部空间、
（b）2.5 dB 是个"既不饱和、又有清晰排序"的好训练点。

### 4.2 度量与对照
- **学习曲线**（核心 RL 图）：`已找到的最优验证 FER` vs `构造评估次数`，RL（两种）/ CEM / 随机搜索
  同图——直接看**样本效率**。
- **BER-vs-Eb/N0 曲线**：RL 冠军 vs 随机搜索冠军 vs 圆盘式（round-robin）启发式 vs 仓库默认随机
  （seed 0）vs 未耦合 5G NR 分量码——看**端到端瀑布**。
- **机理分析**：各冠军在提升后耦合 Tanner 图上的 **4 环数 / 围长 / 分量负载均衡 / 校验度分布**，并
  与随机构造的统计量对比——解释 RL 到底学到了什么结构。
- **零样本迁移**：把特征策略学到的 8 个权重**直接搬到**新配置（更长块 `Z=24`、更高码率 `mp=6`），
  贪心造码，与新配置下的随机/round-robin 比——看策略是否泛化（这是随机搜索/CEM 给不了的）。

---

## 5. 结果

配置：`BG2/ils0/Z=16/mp=8`、`L=30, w=2`、滑窗 `W=6`、`α=0.8`、训练 SNR=2.5 dB；每种搜索方法
**同等预算** `32 步 × 20 = 640` 次构造评估、每次 36 帧（CRN）。冠军用**独立的 2000 帧**新帧库
高精度复测（消除挑选噪声）。数据见 `results_construct_search.json`，图见三张 `exp_construct_*.svg`。

### 5.1 构造决定成败：优化 vs 不优化（最硬的结论）

@2.5 dB、2000 帧独立复测（FER 越低越好）：

| 构造方式 | FER | BER | 4 环数 | 围长 |
|---|---:|---:|---:|---:|
| **CEM**（交叉熵） | **0.064** | 1.37e-4 | 0 | 6 |
| **RL 逐边策略** | 0.080 | 1.98e-4 | 0 | 6 |
| **RL 特征策略** | 0.092 | 2.67e-4 | 0 | 6 |
| RL 特征策略（确定性贪心） | 0.112 | 3.47e-4 | 0 | 6 |
| 随机搜索（同预算 640） | 0.133 | 2.60e-4 | 0 | 6 |
| **round-robin 朴素均衡** | **0.661** | 1.18e-3 | 944 | 4 |
| **仓库默认随机 seed0** | **0.698** | 1.10e-3 | 928 | 4 |

> **仅优化"边扩展"这一个旋钮，FER 从 0.70 降到 0.06–0.13，约 5–11 倍。** 仓库一直在用的默认随机
> 构造（seed 0）恰恰是最差的一档之一（0.70），朴素的"完美均衡"启发式（round-robin）同样差（0.66）。

**更关键的是错误地板**。BER-vs-Eb/N0 瀑布（`exp_construct_ber.svg`）：

| Eb/N0 | 2.2 | 2.5 | 2.8 | 3.1 | 3.4 |
|---|---:|---:|---:|---:|---:|
| RL 特征 | 5.5e-3 | 3.4e-4 | 1.9e-5 | 5.1e-6 | **0** |
| CEM | 2.7e-3 | 1.4e-4 | 4.7e-6 | **0** | **0** |
| 随机搜索 | 3.9e-3 | 3.7e-4 | 3.7e-5 | 2.7e-6 | **0** |
| seed0 默认 | 6.2e-3 | 1.2e-3 | 3.5e-4 | 1.4e-4 | **6.4e-5(地板)** |
| round-robin | 6.8e-3 | 1.1e-3 | 2.6e-4 | 1.0e-4 | **3.3e-5(地板)** |
| 未耦合分量码 | 2.3e-2 | 9.2e-3 | 3.1e-3 | 1.0e-3 | 1.9e-4 |

→ **优化后的构造没有错误地板**（高 SNR 一路降到 ~1e-6 / 0），而默认/朴素构造在 ~3–6e-5 起翘出现
**地板**。以 BER=1e-3 的瀑布门限衡量：优化构造 **2.30–2.38 dB**、默认/朴素 **2.53–2.54 dB**
（构造增益 ~0.16 dB）、未耦合分量码 **3.11 dB**（耦合增益 ~0.75 dB）。**构造优化的主要红利在低 BER /
错误地板区，门限处红利较小**——这与"短环主要伤地板"的理论一致。

### 5.2 RL / CEM 明显比同预算随机搜索更省样本

学习曲线（`exp_construct_learn.svg`，最优验证 FER vs 评估次数）：

| 评估次数 | ~140 | ~260 | ~380 | ~500 | ~620 |
|---|---:|---:|---:|---:|---:|
| 随机搜索 | 0.09 | 0.09 | 0.09 | 0.09 | **0.09（卡住）** |
| RL 特征 | 0.09 | 0.09 | 0.075 | 0.065 | **0.060** |
| RL 逐边 | 0.09 | 0.075 | 0.07 | 0.07 | **0.035** |
| CEM | 0.09 | 0.075 | 0.05 | 0.05 | **0.050** |

→ **随机搜索约 140 次评估后就卡在 0.09 不再改善**（盲搜的 best-of-k 是阶统计量，长尾极慢）；
RL/CEM 借**结构化策略**持续下探到 0.035–0.06。在 2000 帧独立复测上，RL（0.080/0.092）与
CEM（0.064）都稳定优于随机搜索（0.133），**连不含任何采样运气的确定性贪心规则（0.112）也优于
随机搜索**。

### 5.3 机理：RL 自动发现了"避免 4 环"

所有优化构造都是 **n4=0、围长 6**；默认/朴素构造是 **n4≈930、围长 4**（随机构造参考 n4 均值 526、
范围 [0,1408]）。**正是这 ~930 个 4 环造成了默认/朴素构造的错误地板**，而优化把 4 环清零、地板消失。

特征策略学到的 8 维权重（`results_construct_search.json: theta_feature`）：

| 偏置 B0 | 偏置 B1 | 偏置 B2 | 行占比 | **列占比** | **全局占比** | 行内空 | **列内空** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| +0.33 | +0.28 | −0.12 | −0.49 | **−1.85** | **−2.30** | +0.69 | **+1.61** |

读出来就是一条**可解释规则**：**"把同一列（同一变量基列）的系统位边尽量散到不同分量、并保持全局
负载均衡"**（列占比/全局占比强负、列内空强正）。这恰好打断了"同列两条边落进同一分量 → 在提升后的
耦合图里形成 4 环"的机制——**RL 仅凭'真实收发链的 FER'奖励就自动复现了这条码设计原则，无需任何
人工的环/陷阱集知识**。

### 5.4 学到的规则可零样本迁移

把上面 8 个权重**原封不动搬到新码**、贪心造码（不在新码上做任何搜索）：

| 迁移目标 | 码率 | 零样本学到 FER | 随机(中位) | round-robin |
|---|---:|---:|---:|---:|
| Z=24（更长块，ils=1） | 0.603 | **0.124** | 0.146 | 0.240 |
| mp=6（更高码率） | 0.693 | **0.788** | 0.852 | 0.960 |

→ **零样本的学到规则在两个新配置上都低于"典型随机构造（中位）"与人工启发式（round-robin）**，说明
RL 学到的是**真正的结构性原理**、而非对单一码的过拟合。这是**随机搜索 / CEM 给不了的**（它们产出的是
码专属的分配向量、换码必须重搜）。（注：best-of-12 的随机在 Z=24 上偶能压过单次零样本贪心，属正常——
那是 12 次评估 vs 0 次评估的不对等比较。）

### 5.5 诚实小结

1. **最硬、最稳的结论**：边扩展构造决定成败——优化把 FER 降 ~5–11×、并**消除错误地板**；仓库默认随机
   构造其实是最差档之一。这点在 2000 帧独立复测下毫无悬念。
2. **RL 的真实价值**：(a) 仅凭端到端 FER 奖励**自动发现可解释的"避免 4 环"规则**；(b) **比同预算随机
   搜索显著更省样本**；(c) 特征策略的规则**可零样本迁移**到新码。
3. **诚实边界**：CEM（强进化式基线）在单点 FER 上略优于策略梯度 RL；在深瀑布区，几个"无 4 环"构造
   趋于一致（差异在蒙卡噪声内）——即"是否避开 4 环"是一级因素、可被多种优化器达成，RL 的增量价值在于
   **自动化 + 可解释 + 可迁移**，而非在单点 FER 上碾压一切。这与本仓库 [RL_SC_LDPC.md](RL_SC_LDPC.md)
   一贯的诚实风格一致。

### 5.6 全码率扫描 0.5–0.9：构造增益随码率怎么变

前面 §5.1–5.5 是 R≈0.6 的**深度研究**。这里做**广度扫描**，把 RL 构造优化铺到整个码率区间，
口径与仓库 PAM4/热力图一致：低码率用 BG2、高码率用 BG1；每个码率**自动选工作 SNR**（典型随机
构造落在 FER≈0.3 处）；每码率训练 RL 特征策略 + 同预算随机搜索 + round-robin + 默认 seed0，画
BER 曲线，用 **FER=0.1 的瀑布门限 Eb/N0（越低越好）**做跨码率度量。脚本
[experiments_construct_rate.py](experiments_construct_rate.py)，数据 `results_construct_rate.json`，
图 `exp_construct_rate_threshold.svg`（门限-码率总览）/ `exp_construct_rate_ber_*.svg`（每码率 BER）。

**瀑布门限 Eb/N0 @ FER=0.1（dB，越低越好）：**

| SC 码率 | 基图 | RL 优化 | 随机搜索 | round-robin | 默认 seed0 | **增益(seed0−RL)** |
|---:|:--:|---:|---:|---:|---:|---:|
| 0.479 | BG2 | **2.07** | 2.03 | 2.52 | 2.60 | 0.53 dB |
| 0.603 | BG2 | **2.45** | 2.60 | 3.23 | 3.36 | **0.91 dB** |
| 0.692 | BG1 | **2.69** | 2.70 | 2.73 | 3.01 | 0.32 dB |
| 0.770 | BG1 | **3.12** | 3.14 | 3.23 | 3.37 | 0.25 dB |
| 0.906 | BG1 | **4.55** | 4.61 | 4.66 | 4.67 | 0.12 dB |

**结论：**
1. **RL 优化构造在全部 5 个码率上都优于仓库默认 seed0**，门限增益 **0.12–0.91 dB**，并且
   **在中码率 R≈0.6 处最大（~0.9 dB）、向高码率单调缩小**（R0.9 仅 0.12 dB）。这与
   "阈值饱和填补的是 BP–MAP 门限差、而该差在高码率很小"的理论（见 README §5.3）**完全一致**——
   高码率本就接近 MAP，构造再怎么优化空间也小。
2. **RL vs 同预算随机搜索**：瀑布门限基本持平（RL 在甜点 R0.6 领先 ~0.15 dB）；单点 FER 上 RL 在
   5 个码率里 **4 个更优**（仅 R0.479 这个最大的码因每码率只给了 18 步的精简预算而略逊于随机搜索）。
3. **错误地板**：低/中码率下默认 seed0 有明显地板（R0.6：seed0 在 ~4e-5 起翘，RL 降到 ~1.7e-6），
   **RL 构造把地板消除**；高码率两者地板差异缩小。
4. **机理随码率迁移**：低/中码率由"避免 4 环"主导（与 §5.3 一致）；**高码率下 4 环与性能解耦**
   （R0.7 的 round-robin n4=3264 却不差、R0.9 的 RL n4=1424 仍最优）——说明高码率的主导有害结构
   已不再是 4 环，而 RL 仍能凭端到端奖励找到更优构造。

> 一句话：**构造优化在 0.5–0.9 全程都有用，但"用处多大、靠什么机理"强烈依赖码率**——
> 中码率收益最大（~0.9 dB、靠避 4 环），高码率收益小（BP–MAP 间隙本就小）。

---

## 6. 与最接近工作的区分

| | AI Coding (2020) | Esfahanizadeh/Battaglioni (2019–25) | **本工作** |
|---|---|---|---|
| 码 | 分组 / 极化码 | SC-LDPC | SC-LDPC |
| 优化对象 | 生成/校验矩阵 | 划分/扩展矩阵 | **边扩展（划分）assign** |
| 方法 | RL / 遗传算法 | 贪心 / 穷举+剪枝 / MCMC | **策略梯度 RL（序贯 MDP）** |
| 奖励/目标 | 距离/门限/BER | 渐近门限 + 环计数 | **端到端 FER（真实收发链）** |
| 可迁移性 | — | 否（每码重搜） | **是（特征策略零样本迁移）** |

---

## 7. 局限与后续
- **局限**：阈值饱和是渐近性质，本实验用中等规模（Z=16, L=30）演示有限长下构造优化的增益；奖励是
  单 SNR 工作点的 FER（多 SNR/多码率的元学习留待后续）；只优化系统位边扩展（保留 5G NR 校验结构与
  提升移位以维持高效可编码性）。
- **后续**：动作里加入提升移位的协同优化；线性策略换成小 MLP（若引入 PyTorch）以处理更大状态；
  跨 SNR / 跨码率的元强化学习（对标 AM-RELDEC）；与"RL 译码"（[RL_SC_LDPC.md](RL_SC_LDPC.md)）
  正交结合——**一边学怎么造、一边学怎么译**。

---

## 8. 复现
```bash
python3 tests_construct.py                 # 自检（含策略梯度有限差分校验）
python3 experiments_construct.py all       # 搜索+曲线+机理+迁移+出图（纯 NumPy，约 40 分钟）
python3 experiments_construct.py plot      # 仅从 JSON 重绘
```

---

## 9. 参考文献
1. R. J. Williams, *Simple statistical gradient-following algorithms for connectionist
   reinforcement learning (REINFORCE)*, Machine Learning, 1992.
2. I. Bello, H. Pham, Q. V. Le, M. Norouzi, S. Bengio, *Neural Combinatorial Optimization
   with Reinforcement Learning*, arXiv:1611.09940, 2016.
3. L. Huang, H. Zhang, R. Li, Y. Ge, J. Wang, *AI Coding: Learning to Construct Error
   Correction Codes*, IEEE Trans. Commun. 68(1):26–39, 2020 (arXiv:1901.05719).
4. M. Zhang, Q. Huang, S. Wang, Z. Wang, *Construction of LDPC Codes Based on Deep
   Reinforcement Learning*, IEEE WCSP, 2018.
5. D. G. M. Mitchell, M. Lentmaier, D. J. Costello Jr., *Spatially Coupled LDPC Codes
   Constructed from Protographs*, IEEE Trans. Inf. Theory 61(9):4866–4889, 2015
   (arXiv:1407.5366).
6. D. G. M. Mitchell, E. Rosnes, *Edge Spreading Design of High-Rate Array-Based SC-LDPC
   Codes*, IEEE ISIT, 2017.
7. M. Battaglioni, M. Baldi, F. Chiaraluce, D. G. M. Mitchell et al., *Efficient Search and
   Elimination of Harmful Objects in Optimized QC SC-LDPC Codes*, 2019 (arXiv:1904.07158).
8. H. Esfahanizadeh, A. Hareedy, L. Dolecek et al., *A Unified Spatially-Coupled Code
   Design: Threshold, Cycles, and Locality*, arXiv:2203.02052, 2022; *Adapt or Regress*,
   arXiv:2509.21112, 2025.
9. Y. Ren et al., *Edge-Spreading Raptor-Like LDPC Codes for 6G*, arXiv:2410.16875, 2024.
10. M. Stinner, P. M. Olmos, *Analyzing Finite-Length Protograph-Based SC-LDPC Codes*,
    arXiv:1401.8090, 2014.
11. A. Tanrıkulu, M. Yıldırım, A. Hareedy, *An MCMC Method for Efficient Finite-Length LDPC
    Code Design*, arXiv:2504.16071, 2025.
12. G. Liva, M. Chiani, *Protograph LDPC Codes Design Based on EXIT Analysis*, IEEE
    GLOBECOM, 2007.
13. T. Tian, C. R. Jones, J. D. Villasenor, R. D. Wesel, *Selective Avoidance of Cycles in
    Irregular LDPC Code Construction (ACE)*, IEEE Trans. Commun. 52(8):1242–1247, 2004.
14. S. Habib, A. Beemer, J. Kliewer, *RELDEC: Reinforcement Learning-Based Decoding of
    Moderate Length LDPC Codes*, IEEE Trans. Commun., 2023 (arXiv:2112.13934).
