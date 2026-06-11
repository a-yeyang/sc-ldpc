# IEEE T-COM 投稿蓝图：RL 优化 SC-LDPC 边扩展构造

> 决策（2026-06-11）：以**主线 A（RL 边扩展构造）**为核心打包一篇 **IEEE Transactions on
> Communications（SCI 一区，IF 8.3，电信 12/119）** 投稿。RL 译码（[RL_SC_LDPC.md](RL_SC_LDPC.md)）
> 作为正交工作单列；错误地板/吸收集方向（过夜实验**负面**，见 §6 与 [rl-floor-direction 记忆]）
> 仅作 scoped 局限/future work，不进主线。
> 现有科研叙事见 [RESEARCH_ANALYSIS.md](RESEARCH_ANALYSIS.md)、技术细节见 [RL_CONSTRUCTION.md](RL_CONSTRUCTION.md)。
> 本文件是**投稿执行计划**：论点→证据映射、两个补强件的实现规格、图表清单、审稿风险预案、时间线。

---

## 0. 一句话定位
> **"边扩展是 5G NR SC-LDPC 里一个被长期随机化对待的高杠杆构造自由度。我们把它建成 MDP 用策略梯度
> 端到端优化，把 FER 降 5–11×、消除错误地板（机理是消 4 环，并给出一条把'同列边散开'与 4 环计数
> 联系起来的组合引理），用有限长 PEXIT 分析解释其码率相关性；并刻画出 RL 相对盲搜的样本效率优势
> 随单次评估成本与搜索空间单调上升这一可推广判据。"**

定位 = **"内含 ML 方法的有限长编码工程论文"**，RL 是工具、码是主角。淡化"RL-vs-随机赛马"为一小节诚实呈现。

---

## 1. 标题候选
1. *Learning to Spread Edges: Reinforcement-Learning Construction of Spatially-Coupled LDPC Codes*（短、ML 味，建议主用）
2. *Edge Spreading is an Overlooked High-Leverage Design Freedom in 5G-NR Spatially-Coupled LDPC Codes*（码味、强调"被忽视的旋钮"）
3. *Threshold-Aware Reinforcement Learning for Edge-Spreading Design of SC-LDPC Codes*（强调补的门限目标）

---

## 2. 贡献清单（投稿口径，4 条）
- **C1（最硬）**：边扩展是高杠杆但实践中被随机化——仅优化它，FER 降 **5–11×** 并**消除错误地板**；全码率 0.5–0.9 成立，门限增益 0.12–0.92 dB（R≈0.6 处最大 ~0.9 dB）。
- **C2（机理+理论）**：增益来自消 4 环；学到的 8 维特征策略 = 可解释规则"同列边散开"；**用一条组合引理把'同列共分量'与提升图 4 环计数严格联系**（补强件 ②），并用**有限长 PEXIT 分析**解释为何增益随码率递减（补强件 ①）。
- **C3（方法论判据）**：**RL 相对随机搜索/CEM 的样本效率优势随单次评估成本 × 搜索空间 (w+1)^E 单调上升**——长码贵评估 RL 13/15 胜、w 越大优势越大；廉价代理下打平。给出"何时该用 RL"的判据。
- **C4（工程可用性）**：特征策略**可零样本迁移**到新 Z/L/码率，是随机搜索/CEM 给不了的部署属性。

---

## 3. 章节结构（T-COM 双栏，目标 ~11–14 页）
1. **Introduction** — 边扩展自由度被随机化的现状；4 条贡献；与最接近工作的区分表。
2. **Background** — SC-LDPC、原型图、边扩展规则（B=B_0+…+B_w）、5G NR BG1/BG2 截断、阈值饱和（KRU）、为何构造重要（引 Mitchell-T-IT'15 Table III）。
3. **Construction as an MDP** — 状态/动作/回合末奖励；两种策略（FeaturePolicy 8 维可迁移 / PerEdgePolicy 消融）；REINFORCE + 批内基线 + CRN；有限差分校验。
4. **A Threshold Objective (PEXIT)** — 补强件 ①。渐近门限因阈值饱和≈构造无关（解释 C1 中"瀑布增益随码率递减"），有限长 PEXIT（Stinner-Olmos）才区分构造；作为**无帧、有原理**的奖励/预测器。
5. **A Combinatorial Lemma: Same-Column Co-Assignment ↔ 4-Cycles** — 补强件 ②。证明学到的规则降 4 环。
6. **Baselines & Experimental Setup** — 随机搜索/CEM/round-robin/seed0/未耦合分量码；同评估器、同预算、同验证；并行/CRN 工程。
7. **Results** — 7.1 头部空间探针；7.2 构造决定成败(5–11×+消地板)；7.3 机理+引理数值验证；7.4 全码率扫描；7.5 RL-vs-评估成本判据（长码+大 w）；7.6 零样本迁移；7.7 门限目标结果。
8. **Discussion & Limitations** — 诚实边界（RL 胜出有条件、CEM 强基线、中等规模）；**吸收集代理对地板无效的 scoped 负结果**（§6）作 future work。
9. **Conclusion**。

---

## 4. 论点 → 现有证据映射（已具备，无需重跑）

| 论点 | 数据文件 | 图 | 关键数字 |
|---|---|---|---|
| 头部空间（14 随机构造 FER 跨 7×） | `results/construct/results_construct_search.json` | — | FER 0.106→0.725 |
| C1 5–11× + 消地板 | `results_construct_search.json` | `figures/construct/exp_construct_ber.svg` | seed0/RR FER 0.66–0.70 → 优化 0.06–0.13；地板 3–6e-5 → 0 |
| C2 机理 n4 930→0 / girth 4→6 | `results_construct_search.json`（`theta_feature`、n4） | `exp_construct_learn.svg` | n4: ~930→0；学到权重列占比/全局占比强负 |
| C1 全码率扫描 0.5–0.9 | `results/construct/results_construct_rate.json` | `exp_construct_rate_threshold.svg` / `_ber_0*.svg` / `_gain.svg` | 门限增益 0.12–0.92 dB，R≈0.6 最大 |
| C3 长码贵评估 RL 13/15 | `results/construct/results_big_{A,B,merged}.json` + `results_big_finals.csv` | `exp_big_rl_vs_random.svg` / `exp_big_ber_R*_w*.svg` | RL 13/15 胜随机；w=3 → 5/5 |
| C3 大 w 蒙卡（w≤12） | `results/construct/results_wmc_ALL.json` | `exp_wmc_rl_vs_random.svg` | 随机随 w 变差、RL 常到 0 |
| C3 廉价代理大 w（w≤30）打平 | `results/construct/results_wl.json` | `exp_wl_n4_vs_w.svg` / `_vs_L_w*.svg` | RL≈随机皆 n4=0；seed0 不可靠 |
| C4 零样本迁移（瀑布目标，成功） | `results_construct_search.json`（transfer 段） | `exp_construct_transfer.svg` | Z=24/mp=6 零样本低于随机中位+RR |

> **结论：C1–C4 的实证主体已全部就位。** 投稿增量 = 补强件 ①②（把"只有仿真"抬成"仿真+分析+引理"）+ 重绘出版级图 + 写作。

---

## 5. 两个补强件（这是把论文从"会议级"抬到"T-COM 级"的关键，按杠杆排序）

### 补强件 ①：有限长 PEXIT / 门限目标（最高杠杆，补"无理论"短板）
**目的**：(a) 提供一个**无帧、有原理**的目标/预测器，正面回应审稿人 #1 顾虑"只有蒙卡"；(b) 用阈值饱和**解释** C1 的码率相关性（渐近门限≈构造无关 → 瀑布增益随码率缩小；有限长才区分构造）；(c) 是 C3"评估贵→RL 赢"最锋利的实例。

**诚实前提**：不要承诺"渐近 PEXIT 门限能强区分构造"——阈值饱和恰恰让好构造的渐近门限趋同（这本身是要讲的理论）。**区分构造的是有限长 PEXIT（含终止边界）+ 环结构**。所以 ① 的产出是"解释 + 有原理的预测器"，不是"碾压性的新奖励"。

**实现规格**（接现成 `rl_construct.py` 管线，无新依赖，纯 NumPy）：
1. 新增 `pexit.py`：对由 `assign` 决定的 SC 原型图（带状 B_0..B_w，含 L 与终止边界）实现 **protograph PEXIT**（BIAWGN，J/J^{-1} 函数；Liva-Chiani GLOBECOM'07 + Stinner-Olmos arXiv:1401.8090 有限长版）。返回 (i) 收敛所需 Eb/N0 门限、(ii) 有限长 PEXIT 预测的 BER 代理。
2. 接入：`reward_of(metric, kind="pexit")` → reward = −threshold_ebn0；`train_reinforce(..., reward_kind="pexit")` 直接复用。新增 `eval_pexit(cfg, assigns)` 评估器（替换 `_worker` 蒙卡路径之一）。
3. **验证实验**（写进 §7.7）：(a) PEXIT 门限排序 vs 蒙卡瀑布门限排序的 Spearman（证明代理有效）；(b) 用 PEXIT 当奖励训 RL，冠军再用蒙卡复测，对比 MC-奖励冠军；(c) PEXIT 评估成本 vs 蒙卡成本 → 强化 C3。
4. 工作量估计：~2–3 天写 PEXIT + 1 天接管线 + 集群跑一晚。

### 补强件 ②：组合引理（同列共分量 ↔ 4 环）（中杠杆，消解"只是仿真"）
**目的**：把 §7.3 学到的"同列边散开"规则从经验观察抬成**可证命题**：证明同一基列的两条系统位边共分量会在提升后的 QC Tanner 图里产生可计数的 4 环，散开则消除主导 4 环群。

**引理陈述（草案，待精确化）**：
> 设 SC-LDPC 由 5G NR 基图按 `assign` 边扩展、再以循环移位 `P^{s}` 提升（Z×Z 循环阵）。若同一变量基列的两条系统位边 e_i, e_j 被分到同一分量 B_a 且其校验基行相同，则提升后在该 (变量块, 校验块) 对之间产生 **gcd(...) 个长度为 4 的环**（标准 QC 4 环条件，Fossorier'04 / ACE Tian'04）；把 e_i, e_j 分到不同分量 B_a≠B_b 使二者落入耦合图中**不同的校验位置块**，从而打断该 4 环。由此，"列内最大化分量多样性"的构造使按列计的 4 环计数下界为 0。

**验证**：用现成 `count_4cycles(sc)`（`rl_construct.py:489`）+ `girth(sc)` 数值核对引理预测的 4 环计数 vs 实测（round-robin/seed0 的 ~930 vs 优化后 0），跨 Z∈{16,32,64} 验证标度。
**工作量**：~2–3 天（证明 + 数值核对 + 写作）；难度中等，是标准 QC 环条件的直接应用，不需要新仿真。

> 做 ①+② 即可显著抬高接收概率。**规模刷到 Z=256/GPU 对 T-COM 是低杠杆，不做。**

---

## 6. 必须诚实写进 Limitations 的负结果（来自 2026-06 过夜实验）
- **吸收集代理对地板无效**：`floor_gate.json` 显示地板跨 2.53 个数量级（构造相关 ✓），但 Spearman(a≤6 吸收集计数, 地板)=**−0.13**；`cem_floor`（最小化吸收集）地板 6.1e-7 **差于** `cem_waterfall` 3.4e-8。→ 标准廉价吸收集枚举**预测不了**这类码的有限长地板，最小化它甚至有害。
- **floor 目标下零样本迁移退化**：`transfer.json` 中学到策略退化成"全 0"分配，as_count 是 CEM 的 11.8×、且自身有地板(4.3e-5)而基线无。→ 瀑布目标上成功的迁移，换到 floor 目标失败。
- **写法**：作为一段**有价值的 scoped 负结果 + future work**——"真实地板需昂贵蒙卡目标或**学到的地板代理**（neural surrogate，amortize 不可行的陷阱集枚举）"，自然指向后续（也是潜在 AAAI/CCF-A 的种子，但不在本文）。

---

## 7. 出版级图表清单（T-COM 终稿，~8–10 图）
1. 带状 H_SC 结构 + 边扩展示意（schematic，新画）。
2. 头部空间：14 随机构造 FER 分布（条/箱）。
3. **主图**：BER-vs-Eb/N0 瀑布，6 条构造（优化×3 / seed0 / RR / 未耦合），标出地板（← `exp_construct_ber.svg` 重绘）。
4. 学习曲线：best-val FER vs 评估次数，RL/CEM/随机（← `exp_construct_learn.svg`）。
5. 机理：n4/girth 对比 + 学到权重条形（← 引理 §5 数值核对叠加）。
6. 全码率门限增益 vs 码率（← `exp_construct_rate_threshold.svg` / `_gain.svg`）。
7. **C3 主图**：RL/随机优势比 vs 码率分 w 线 + vs w（← `exp_big_rl_vs_random.svg` + `exp_wmc_rl_vs_random.svg` 合并）。
8. 零样本迁移条形（← `exp_construct_transfer.svg`）。
9. 补强件 ①：PEXIT 门限 vs 蒙卡门限散点（Spearman）+ PEXIT-奖励 RL 收敛。
10.（可选）大 w 廉价代理 n4 vs w（← `exp_wl_n4_vs_w.svg`），收口 C3 判据。

---

## 8. 审稿风险预案（preempt）
| 风险 | 预案 |
|---|---|
| "REINFORCE 1992，方法不新" | 定位为**编码工程**论文；新颖性在"首次 SC-LDPC 边扩展 MDP + 端到端奖励 + 可迁移策略"，非新优化器。引 AI Coding(T-COM'20) 为同型先例。 |
| "RL 没碾压 CEM/随机" | 不主张碾压；**主张 C3 判据 + C4 可迁移**，把"有条件"当成诚实卖点（方法适用边界）。 |
| "只有蒙卡仿真" | 补强件 ①（PEXIT）+ ②（引理）。 |
| "规模小（Z≤128）" | 论证对 T-COM/有限长设计**足够**；规模非本文贡献维度；引"贵评估"正是中等规模也成立的原因。 |
| "新颖性是否被抢" | 已核：2025–2026 SC-LDPC 构造仍是非 RL（结构规避/CLLL/高围长）；RL×边扩展未见。**先挂 arXiv 打时间戳。** |
| 标准相关性 | 一段绑 3GPP TS 38.212 BG1/BG2 + 6G 边扩展(ESRL)，便宜买 relevance。 |

---

## 9. 时间线（现实，~3–4 个月到投稿）
- **W0（现在）**：挂 arXiv（防抢 + IEEE 允许预印本；cs.IT/eess.SP 可能需 endorsement，找导师背书）。整理出版级图 3/4/6/7/8。
- **W1–2**：补强件 ②（引理 + 数值核对）——便宜、确定性高，先做。
- **W2–4**：补强件 ①（`pexit.py` + 接管线 + 集群跑），出图 9。
- **W4–6**：写 §1–6（背景/方法/理论），§7 结果。
- **W6–8**：Discussion/Limitations（含 §6 负结果）、abstract、参考文献、内部 review、查重。
- **W8–10**：终稿、IEEE 模板排版、投 T-COM。
- **并行快胜（可选）**：W1–3 把 C1+C2 压成 ≤5 页投 **IEEE Communications Letters** 或赶 **ISIT/ITW** 截止，拿首胜（与 T-COM 不冲突，结果子集即可）。

---

## 10. 立即可执行的下一步（建议顺序）
1. `arXiv` 预印本（最高性价比，先做）。
2. 补强件 ②（引理）——`rl_construct.py:489 count_4cycles` 已就位，先做数值核对脚本 `lemma_check.py`。
3. 补强件 ①（`pexit.py`）。
4. 出版级重绘（统一 `plotting.py` 风格，T-COM 单栏宽）。
