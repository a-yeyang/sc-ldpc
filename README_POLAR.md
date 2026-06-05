# 基于 5G NR 标准 Polar 码的空间耦合 Polar 码（SC-Polar / PIC）

> 一个自包含的 Python 实现：以 3GPP TS 38.212 的 5G NR **Polar 码**为分量码，
> 用 **部分信息耦合（Partially Information Coupled, PIC）** 构造空间耦合 Polar 码，
> 并在 BPSK + AWGN 信道下完成 **编码** 与 **译码**（SC / CRC 辅助 SCL / 置信传播 BP，
> 以及在耦合链上的 **窗口化软译码**）。
>
> 这是仓库内 [SC-LDPC 工作](README.md) 的 Polar 对应版本：分量码 NR-LDPC → **NR-Polar**，
> 耦合记忆 `w` → **耦合深度 `J`**，边扩展 → **共享信息位**，末端归零 → **两端 dummy 终止**，
> 滑窗 BP → **窗口化 SC/SCL/BP**。

---

## 一、空间耦合 Polar 是怎么回事（调研）

### 1.1 三大构造流派

把多个 Polar **分组码（code blocks, CB）** 沿"空间"耦合起来，让相邻分组共享信息，
从而像 SC-LDPC 那样把好的"边界"传播成一条译码波。文献主要分三条线：

| 流派 | 代表文献 | 耦合机制 | 是否用 3GPP Polar 分量码 |
|------|----------|----------|--------------------------|
| **① PIC — 部分信息耦合**（本实现） | Wu, Yang, Xie, Yuan, *Partially Information Coupled Polar Codes*, **IEEE Access 2018**；广义深度 J：Yu et al., **IEEE T-COMM 2021** | 相邻 CB **共享若干系统信息位**；链两端插 dummy 位终止 | ✅ 是（分量码就是标准 Polar） |
| ② 冻结位 / XOR 耦合 | Wang et al., *Improving Polar Codes by Spatial Coupling*, **ISITA 2018** | 一个 CB 的消息位当作相邻 CB 的冻结位（前馈）；或相邻位消息块模 2 相加 | ✅ 是 |
| ③ 卷积 Polar (CvPC) | Ferris–Poulin 2013（branching-MERA）；Ferris–Hirche–Poulin 2017 | 把 2×2 Arıkan 核换成带状卷积核 | ❌ 否（改了 Polar 变换本身） |

"**空间耦合 Polar**"字面最贴切的是 **① PIC**——它在文献里正式称为 *spatially-coupled*，
分量码是**标准（可为 3GPP）Polar 码**，耦合方式（共享信息位、深度 `J`、两端 dummy 终止）
与 SC-LDPC 几乎一一对应。CvPC 虽叫"卷积/耦合"，但它**替换了 Polar 变换**，
不再是 3GPP 分量码，故不选。本仓库实现 PIC。

### 1.2 PIC 耦合结构与终止

设链长 `L`，每块是一个 NR Polar 码。每块的 `A` 个信息位（坐落在最可靠的信息坐标上）
按角色划分为：

```
[ in^(1)..in^(J) |   fresh   | out^(1)..out^(J) ]   (+ 每块 CRC)
   \___ J·c ___/    Kfresh      \___ J·c ___/
```

* **耦合记忆 / 深度 `J`**：每块与前后各 `J` 个块共享，每个共享组 `c` 比特。
* **共享**：`in^(k)_t = out^(k)_{t-k}`（块 t 的"耦合入"就是块 t−k 的"耦合出"，同一批比特）。
* **终止**：链左端 `in^(k)_t (t<k)` 与右端 `out^(k)_t (t+k>L−1)` 取 **dummy 0**（已知），
  这就是触发译码波的边界（对应 SC-LDPC 的末端归零）。
* **码率损失**：共享位只算一次新信息，`R ≈ (A − J·c)/E < A/E`（耦合开销），与 SC-LDPC 的
  终止开销同理。

### 1.3 为什么耦合（以及一个关键的有限长事实）

* **耦合增益（有限长）**：PIC 在多种码率下优于非耦合 Polar，深度 `J=1` 约 +0.3 dB，`J≥2` 更多。
* **理论根**：借自 SC-LDPC 的**阈值饱和**（Kudekar–Richardson–Urbanke 2011）。⚠️ 但 Polar 侧
  目前是**有限长增益（PIC）/ SC 误差指数提升（CvPC，log₂3/2≈0.79）**，并没有被证明的
  Polar 阈值饱和定理。
* **本实现验证到的关键事实**（见 §6）：**硬判决 SC/SCL 耦合在等码率下没有增益**——每个共享位
  只被一个块"拥有"（从信道译出），所以每块要译的未知位数不变；增益必须靠**软译码（BP）的分集**
  ——一个共享位被**两个块**保护，BP 把两块的信道证据合并，才把分集兑现成增益。

---

## 二、基于 3GPP TS 38.212 的分量 Polar 码构造（`nr_polar.py`）

完全遵循 TS 38.212 §5.3.1 + §5.4.1（实现并校验通过）：

1. **极化变换**：`x = u·G_N`，`G_N = F^{⊗n}`，`F=[[1,0],[1,1]]`，`N=2^n`（非比特反序）。
2. **可靠度序列**：Table 5.3.1.2-1 的长度-1024 序列 `Q`（`data/polar_Q_Nmax.txt`，已校验为 0..1023 的合法置换，且满足嵌套性）。`Q[0]` 最不可靠、`Q[1023]` 最可靠。
3. **码长 `N`**（§5.3.1）：由 `(K, E, n_max)` 推出（`n_max=9` 用于 PBCH/PDCCH，`10` 用于 PUCCH）。
4. **速率匹配**（§5.4.1）：32 子块交织 `P` + 三种模式——**重复**（`E≥N`）、**打孔**（`K/E≤7/16`，取尾部）、**缩短**（否则取头部）。
5. **冻结/信息集**（§5.3.1.2）：未传坐标预冻结（打孔再加 `3N/4`/`9N/16` 规则），其余按 `Q` 取最可靠的 `K=A+L_crc` 个坐标承载 信息+CRC。
6. **CRC**：CRC6 / CRC11 / CRC24C（§5.1 生成多项式），用于 CA-SCL 及逐块校验。

```python
from nr_polar import NRPolarCode
code = NRPolarCode(A=40, E=128, n_max=9, crc_len=11)   # 40 信息位 -> 128 发送位
e   = code.encode(a)                 # CRC 附加 + 极化 + 速率匹配
ah  = code.decode_sc(llr)            # 连续消除
ah, ok = code.decode_scl(llr, L=8)   # CRC 辅助 SCL
```

---

## 三、PIC 空间耦合（`sc_polar.py`）

```python
from sc_polar import SCPolarCode
sc = SCPolarCode(NRPolarCode(A=40, E=128), L=40, J=1, c=8, seed=0)
cw, info = sc.encode(rng)            # 逐块编码，自动放置共享位 + 两端 dummy
llr      = sc.make_llr(cw, sigma, rng)
info_hat, crc_pass = sc.decode_bp(llr, bp_iter=30)   # 窗口化软译码（有耦合增益）
```

---

## 四、译码算法（`polar_decoder.py` / `polar_bp.py`）

| 译码器 | 文件 | 说明 |
|--------|------|------|
| **SC** | `polar_decoder.py` | 连续消除，自然序蝶形递归，O(N log N) |
| **CA-SCL** | `polar_decoder.py` | LLR 域路径度量的列表译码 + CRC 选径（5G 主力，支持**动态冻结**：把某坐标钳到邻块已译值）|
| **BP** | `polar_bp.py` | 因子图软入软出 BP（归一化最小和盒加），输出每个 u 坐标的外信息 |
| **窗口化 SC/SCL（硬）** | `sc_polar.decode` | 双向逐块扫描，邻块 CRC 通过即把共享位钳为已知 |
| **窗口化 BP（软）** | `sc_polar.decode_bp` | 逐块 BP，**在共享位上与邻块交换外信息 LLR** 并迭代——真正兑现耦合分集 |

窗口化 BP 是 SC-LDPC 滑窗 BP 的 Polar 翻版：共享位 `g=out^(k)_t=in^(k)_{t+k}`，
块 t 对 `g` 的外信息送入块 t+k 作先验，反之亦然，多轮前/后向扫描。

---

## 五、代码结构

| 文件 | 作用 |
|------|------|
| `nr_polar.py` | 3GPP NR Polar 分量码：可靠度序列、`N` 推导、速率匹配、冻结集、CRC、极化变换、编码、SC/SCL 接口 |
| `polar_decoder.py` | **SC** 与 **CA-SCL** 译码器（LLR 域路径度量，支持动态冻结） |
| `polar_bp.py` | 因子图 **BP** 软入软出译码器（耦合所需的外信息来源）|
| `sc_polar.py` | **PIC 耦合层**：共享位映射、深度 `J`、两端 dummy 终止、逐块编码、`make_llr`、窗口化 SC/SCL/BP 译码 |
| `experiments_polar.py` | AWGN 实验：`main`（耦合增益）/ `algos`（译码算法对比）/ `J`（耦合深度）/ `wave`（译码波）|
| `tests_polar.py` | 自检：极化变换=u·G_N、序列置换、速率匹配、CRC、高 SNR 恢复、SCL(L=1)=SC、PIC 一致性 |
| `data/polar_Q_Nmax.txt` | 3GPP Table 5.3.1.2-1 长度-1024 可靠度序列（已校验）|

---

## 六、实验结果（BPSK / AWGN）

> 重跑：`python experiments_polar.py main|algos|J|wave`，再 `python experiments_polar.py plot`。

### 6.1 耦合增益：PIC 窗口化 BP vs 非耦合 BP（等码率 R≈0.25）—— `exp_polar_main.svg`

同一译码器（BP）、同一码率下，PIC 软窗口化 BP 明显优于非耦合 BP，且增益随 SNR 增大
（典型的空间耦合分集特征）：

| Eb/N0 | 非耦合 BP | **PIC 窗口化 BP** | 增益 |
|-------|-----------|-------------------|------|
| 3.0 dB | 3.1e-2 | **1.9e-2** | ~1.7× |
| 3.5 dB | 1.2e-2 | **6.3e-3** | ~1.8× |
| 4.0 dB | 5.3e-3 | **2.4e-3** | ~2.2× |
| 4.5 dB | 1.7e-3 | **6.7e-4** | ~2.5× |

作为强参照，**非耦合 CA-SCL**（5G 主力译码器）在此短码长下绝对性能最好（约 2 dB 即跌到 0 误码）；
耦合增益是**同译码器（BP）内**的有限长效应，与文献（PIC ~0.3 dB）一致。

### 6.2 译码算法对比（同一耦合码）—— `exp_polar_algos.svg`

* **PIC-SC（list=1，硬）**：误差传播严重，FER≈1 直到高 SNR——硬连续消除不适合耦合。
* **PIC-CA-SCL（硬）**：列表显著缓解，但**仍不及非耦合 CA-SCL**（见 §6.1），印证"硬耦合等码率无增益"。
* **PIC-窗口化 BP（软）**：在 BP 族内兑现耦合分集。

### 6.3 耦合深度 `J`（等码率，`J·c=8` 固定）—— `exp_polar_J.svg`

`J=2`（4.7e-3 @3.5dB）优于 `J=1`（8.3e-3 @3.5dB），即 **`J≥2` 增益更大**，与 Yu et al. 2021 一致。

### 6.4 译码波 —— `exp_polar_wave.svg`

逐块误码率 vs 块位置：两端（终止 dummy 锚定）最可靠，向中间升高（波从两端向内传播）。
例如 3.0 dB：端块 ≈0.07 vs 中间块 ≈0.14（~2×），是空间耦合的标志性特征。

---

## 七、快速开始

```bash
python tests_polar.py                  # 全部自检
python experiments_polar.py main       # 耦合增益主实验 -> results_polar_main.json + exp_polar_main.svg
python experiments_polar.py algos      # 译码算法对比
python experiments_polar.py J          # 耦合深度
python experiments_polar.py wave       # 译码波
python experiments_polar.py plot       # 由 JSON 重新出图
```

---

## 八、设计选择与局限

- **为何选 PIC？** 它就是文献里的"空间耦合 Polar"，分量码忠于 3GPP，耦合结构与本仓库 SC-LDPC 一一对应。
- **硬 vs 软**（核心结论）：等码率下硬 SC/SCL 耦合无增益（共享位归属唯一，未知位数不变）；
  软窗口化 BP 通过共享位**分集**获得增益。这是本实现验证到的、也是 SC-Polar 译码的关键。
- **绝对性能**：短码长下非耦合 CA-SCL 仍最强；耦合增益是同译码器（BP）内的有限长效应。
  更大 `N`、更长链 `L`、更优窗口调度可进一步放大增益。
- **复杂度**：窗口化 BP 逐块顺序进行，可按块向量化加速（未做）；`J` 增大时外迭代轮数随之增加。
- **未实现**：PC 冻结位（上行小块）、分布式 CRC 交织、信道交织、联合窗口 SCL；可作为扩展。

---

## 九、参考文献

1. X. Wu, L. Yang, Y. Xie, J. Yuan, *"Partially Information Coupled Polar Codes,"* **IEEE Access**, vol. 6, 2018.
2. Y. Yu, Z. Wang, et al., *"Generalized Partially Information Coupled Polar Codes With Arbitrary Coupling Depth…,"* **IEEE Trans. Communications**, 2021.
3. L. Wang et al., *"Improving Polar Codes by Spatial Coupling,"* **ISITA**, 2018.
4. A. J. Ferris, C. Hirche, D. Poulin, *"Convolutional Polar Codes,"* arXiv:1704.00715 / IEEE Trans. IT.
5. S. Kudekar, T. Richardson, R. Urbanke, *"Threshold Saturation via Spatial Coupling…,"* **IEEE Trans. IT**, 2011/2013.
6. A. R. Iyengar, P. Siegel, R. Urbanke, J. Wolf, *"Windowed Decoding of Spatially Coupled Codes,"* **IEEE Trans. IT**, 2013.
7. E. Arıkan, *"Channel Polarization…,"* **IEEE Trans. IT**, 2009；及其 Polar BP 译码。
8. I. Tal, A. Vardy, *"List Decoding of Polar Codes,"* **IEEE Trans. IT**, 2015；A. Balatsoukas-Stimming et al., *"LLR-based SCL decoding,"* 2015.
9. 3GPP TS 38.212, *"NR; Multiplexing and channel coding,"* §5.3.1（Polar）, §5.4.1（速率匹配）, §5.1（CRC）.
