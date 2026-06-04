# 空间耦合 LDPC（SC-LDPC）——MATLAB 完整复现

> 本目录是 `../`（Python 实现）的**逐模块 MATLAB 复现**：以 3GPP TS 38.212 的
> 5G NR QC-LDPC 为分量码，构造空间耦合 LDPC，在 BPSK + AWGN 下完成
> **发射机编码** 与 **接收机译码**（全图置信传播 + 滑窗译码），并复现
> 蒙特卡洛 BER 仿真与四个参数实验。

算法、构造方法、理论背景与 Python 版**完全一致**，请参阅上级目录的
[`../README.md`](../README.md)。本文件只说明 MATLAB 版的用法、文件对应与实现要点。

环境：MATLAB R2025a（macOS / Apple Silicon），**仅用基础 MATLAB，不依赖任何工具箱**
（BPSK、AWGN、LLR、BP 译码器、QC 提升、GF(2) 求逆全部自实现，与 Python 版只用
numpy/scipy 对应）。

---

## 一、文件对应（Python → MATLAB）

| Python | MATLAB | 作用 |
|--------|--------|------|
| `nr_ldpc.py` | `NRLDPCCode.m`、`loadBaseMatrix.m`、`gf2inv.m`、`shiftVec.m`、`edgesFromEntries.m` | 加载基图；QC 提升；5G NR 编码器（核求逆 + 累加）；Tanner 边表 |
| `decoder.py` | `Tanner.m` | `Tanner` 类 + 向量化 BP（归一化最小和 / 和积），综合征提前停止 |
| `channel.py` | `+ch/`（`ebn0_to_sigma`、`bpsk`、`awgn`、`llr_awgn`） | BPSK 调制、AWGN、LLR、`Eb/N0 → σ` |
| `sc_ldpc.py` | `SCLDPCCode.m` | 边扩展；耦合带状矩阵；逐位置 SC 编码；打孔/终止/LLR 记账；滑窗译码器 |
| `simulate.py` | `simulate.m` | 蒙特卡洛 BER/FER 扫描（分量码 / SC 全图 / SC 滑窗） |
| `experiments.py` | `experiments.m` | 参数实验：码率 / 码长 / 耦合链长 L / 耦合深度 w / 大链长 |
| `demo.py` | `demo.m` | 端到端演示（带状结构、编码校验、收发译码、译码波） |
| `analyze_bg.py` | `analyzeBg.m` | 在真实 BG1/BG2 上验证 5G 校验结构假设 |
| `plotting.py` | `plotCurves.m` | BER 曲线绘图（用 MATLAB 原生 `semilogy` + 导出 PNG，替代纯 SVG）|
| `tests.py` | `runTests.m` | 快速自检（`H_SC·c=0`、码率、收发译码恢复）|
| — | `crossValidate.m` | **逐比特** 对拍 Python 参考向量（见下） |

数据文件 `../data/NR_*.txt` 与 Python 版**共用**（`loadBaseMatrix` 按相对路径 `../data` 读取）。

---

## 二、快速开始

在 MATLAB 命令行中：

```matlab
cd matlab            % 进入本目录（自动加入搜索路径；各脚本也会 addpath 自身目录）

demo                 % 端到端演示：带状结构、编码校验、全图/滑窗译码、译码波
analyzeBg            % 验证 5G NR BG1/BG2 的可编码校验结构
runTests             % 快速自检（对应 tests.py 的全部不变量）
crossValidate        % 逐比特对拍 Python 分量码编码器 + BP 译码器
```

蒙特卡洛 BER 曲线（与 Python 子命令一一对应，可分别运行）：

```matlab
simulate component   % 5G NR BG2 分量码      -> results_component.mat/.json
simulate sc          % SC-LDPC 全图 BP       -> results_sc.mat/.json
simulate windowed    % SC-LDPC 滑窗 BP       -> results_windowed.mat/.json
simulate plot        % 合成 ber_curves.png + results.csv
```

参数实验（四个设计旋钮 + 大链长）：

```matlab
experiments L              % 耦合链长 L ∈ {8,16,40}      (BG2, Z=16, w=2)
experiments w              % 耦合深度 w ∈ {1,2,4}        (BG2, Z=16, L=30)
experiments rate           % 码率 R ∈ {0.20,0.33,0.50}   (BG2 截断, Z=24)
experiments rate_high      % 码率 R ∈ {0.6,0.7,0.8,0.9}  (BG1 截断, Z=24)
experiments len            % 码长 Z ∈ {16,32,64}         (BG2, w=2, L=24)
experiments chain 100      % 大耦合链长（滑窗译码），每个 L 一次：30/100/200/300
experiments chain_plot     % 合成 exp_chain.png
```

每个实验输出 `results_exp_<name>.mat/.json` 与 `exp_<name>.png`。

> 输出文件写入当前目录（`matlab/`），与 Python 版的输出分开，互不覆盖。

---

## 三、验证

本移植经过三层验证，确保与 Python 行为一致：

1. **`runTests`（MATLAB 内自检，对应 `tests.py`）**：分量码 `H·c=0` 且系统化（BG1/BG2、多个 Z）；
   SC 码 `H_SC·c=0`（终止 + 系统位边扩展）；码率 `R_L < R`；2 dB 下全图与滑窗译码均正确恢复。

2. **`crossValidate`（逐比特对拍）**：分量码无随机性，故 MATLAB 必须与 Python **逐比特相同**。
   本目录随附 `ref_component.json`（由 Python 原始模块生成的确定性参考：固定消息→码字、
   固定 LLR→最小和/和积硬判、`core_inv` 指纹）。`crossValidate` 对 6 种配置
   （BG1/BG2 × 多个 Z × 多个码率）核对**码字、最小和/和积译码输出、综合征、码率、维度**全部一致。

3. **静态分析**：全部 `.m` 文件通过 MATLAB 代码分析器（mlint）检查，无语法/代码错误。

> 此外，开发时用 NumPy 按 MATLAB 语义（列主序 `reshape`、1 索引、`circshift`）逐函数模拟了
> 本移植，与 Python 原实现逐比特一致（分量码编码/译码 6 配置 + SC 全部不变量），
> 进一步确认了索引/reshape 翻译的正确性。

---

## 四、实现要点与与 Python 的差异

- **循环移位约定**：`shiftVec(x,v) = circshift(x,-v)`，即 `(P^v x)[r] = x[(r+v) mod Z]`，
  与 Python 的 `np.roll(x,-v)` 完全一致；编码器（综合征累加）与提升后的 Tanner 图共用同一约定，
  故 `H·c=0` 严格成立。基矩阵元素 `v` 提升为 `circshift(eye(Z), v, 2)`（行 `r` 在列 `(r+v) mod Z` 处为 1）。

- **1 索引 vs 0 索引**：内部的块索引（`ent_i/ent_ri/ent_t/ent_cj`、窗口边界、掩码位置）
  保持与 Python 一致的 **0 索引**语义，仅在访问 MATLAB 数组、构造边表时 `+1`，
  以便与 Python 逐行对照、便于复核。

- **随机数**：`SCLDPCCode` 的边扩展、各仿真的消息/噪声用 MATLAB `RandStream`（`mt19937ar`）
  生成，**无法逐比特复刻 numpy 的 PCG64 流**。因此蒙特卡洛 BER 的具体数值会有统计涨落，
  但**趋势、瀑布位置、耦合增益与 Python 一致**；所有**结构性不变量**（`H_SC·c=0`、系统化、
  码率、终止）与码本身的正确性**完全复现**。

- **和积（sum-product）译码的退化角落**：当某条边的 LLR 恰为 0（打孔位初始 LLR=0）时，
  Python 版 `prod/tanh` 会出现 `0/0=NaN` 并向下传播；MATLAB 版对此做了**鲁棒处理**
  （`NaN → 0`）。二者仅在“打孔位 + 和积”这一从不出现于任何实验的退化情形下不同——
  所有 `simulate` / `experiments` / `demo` / `runTests` 均使用**归一化最小和**（`α=0.8`，5G 常用），
  该路径与 Python **逐比特一致**（已验证）；和积在无零 LLR 边时也与 Python 逐比特一致。

- **绘图**：用 MATLAB 原生 `semilogy` 导出 PNG（`plotCurves.m`），替代 Python 的纯 SVG 生成器；
  同时输出 `.mat` 与 `.json`（`jsonencode`）便于复用与与 Python 结果对照，`results.csv` 格式一致。

---

## 五、关于 `matlab -batch` 无头启动（macOS 注意）

在某些**无图形会话**的无头/自动化环境中，`matlab -batch "..."` 可能在桌面（MVM/CEF）
初始化阶段长时间不返回——这是 macOS 上 MATLAB 桌面服务的启动行为，**与本目录代码无关**。
**在你正常使用的 MATLAB 图形界面 / 命令行窗口中直接运行上述命令即可**（交互式启动不受影响）。
若确需无头批处理，建议在已登录图形会话的终端里运行。

代码本身已通过 mlint 静态分析与数值等价性验证；如遇启动问题，请在交互式 MATLAB 中执行。
