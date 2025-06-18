
# 语音识别大作业实验报告

**作者：翟仲浩 / 2023010706**  
**日期：2025年6月**

## 摘要

本实验报告阐述了一个基于动态时间规整（DTW）和多种语音特征（LPC倒谱、FFT倒谱）的数字语音识别系统。系统通过短时能量和过门限率（TCR）进行语音活动检测（VAD），并提取16点LPC倒谱特征和20点FFT倒谱特征，利用DTW Barycenter Averaging (DBA) 算法构建稳健的数字模板，并使用DTW算法评估目标语音到各数字模版的欧式距离最小路径，实现对数字0-9的单数字和多数字识别任务。

---

## 系统设计与实现

本语音识别系统主要包含三个核心模块：`utils.py`、`train.py` 和 `main.py`。

### utils.py：工具函数库

#### 语音活动检测（VAD）

`vad_energy_tcr` 函数实现了基于短时能量和过门限率的语音活动检测。  
该函数首先对信号进行去直流处理，然后分帧（20ms帧长、10ms帧移）计算每帧的能量和TCR。  
阈值设置为归一化后大于等于0.01，结合中值滤波平滑处理，区分语音活动区域。对于多数字识别（`mode=multi`），函数返回多个语音段。

```python
def vad_energy_tcr(signal, fs, mode, plot=False):
    energy /= np.max(energy)
    tcr /= np.max(tcr) if np.max(tcr) > 0 else 1
    vad = (energy > energy_threshold) | (tcr > tcr_thres)
    vad = medfilt(vad.astype(float), kernel_size=5)
    # ... (extract start/end based on mode)
    return start_list, end_list
```

#### 特征提取

`extract_features` 函数从语音片段中提取LPC倒谱、FFT倒谱及其一阶差分特征。

```python
def extract_features(signal, fs):
    feat_lpc = lpc_cepstrum(frame, order=16)
    feat_fft = fft_cepstrum(frame, num_coeffs=20)
    frames_lpc = np.array(frames_lpc)
    frames_fft = np.array(frames_fft)

    delta_features_lpc = calculate_delta(frames_lpc, N=2)
    delta_features_fft = calculate_delta(frames_fft, N=2)

    frames_lpc = np.concatenate((frames_lpc, delta_features_lpc), axis=1)
    frames_fft = np.concatenate((frames_fft, delta_features_fft), axis=1)
    return frames_lpc, frames_fft
```

#### DTW距离计算

`dtw_distance` 实现了DTW算法，计算两个特征序列之间的相似度。

```python
def dtw_distance(s1, s2):
    scaler_s1 = StandardScaler()
    s1 = scaler_s1.fit_transform(s1)

    scaler_s2 = StandardScaler()
    s2 = scaler_s2.fit_transform(s2)

    N, M = len(s1), len(s2)
    D = np.full((N + 1, M + 1), np.inf)
    D[0, 0] = 0

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost = np.linalg.norm(s1[i - 1] - s2[j - 1])
            if j > 1:
                D[i, j] = cost + min(D[i-1, j], D[i-1, j-1], D[i-1, j-2]) 
            else:
                D[i, j] = cost + min(D[i-1, j], D[i-1, j-1])
    return D[N, M]
```

---

### train.py：模板构建

#### 特征序列重采样

```python
def resample_features(features, target_length):
    x_old = np.linspace(0, current_length - 1, current_length)
    x_new = np.linspace(0, current_length - 1, target_length)
    for i in range(num_coeffs):
        f_interp = interp1d(x_old, features[:, i], kind='linear', fill_value="extrapolate")
        resampled_features[:, i] = f_interp(x_new)
    return resampled_features
```

#### 稳健模板构建

`build_templates` 函数遍历训练数据并构建数字模板，使用 DBA 进行多样本平均，减少个体差异影响。

```python
def build_templates(data_dir):
    robust_lpc_template = DBA(np.array(resampled_lpc_sequences,dtype=object), max_iter=10)
    robust_fft_template = DBA(np.array(resampled_fft_sequences,dtype=object), max_iter=10)
    templates[digit] = (robust_lpc_template, robust_fft_template)
```

---

### main.py：数字识别

#### 数字识别流程

```python
def recognize_digit(signal, fs, templates, mode):
    start, end = vad_energy_tcr(signal, fs, mode, plot=True)
    best_digit_list = []
    for i in range(len(start)):
        seg = signal[start[i]:end[i]]
        feat_lpc, feat_fft = extract_features(seg, fs)
        min_dist = float('inf')
        best_digit = None
        for digit in range(10):
            dist_lpc = dtw_distance(feat_lpc, templates[digit][0])/feat_lpc.shape[0]
            dist_fft = dtw_distance(feat_fft, templates[digit][1])/feat_fft.shape[0]
            dist = dist_lpc + dist_fft
            if dist <= min_dist:
                min_dist = dist
                best_digit = digit
        best_digit_list.append(best_digit)
    return best_digit_list
```

#### 评估模式

- **`single` 模式（单数字识别）**：每个文件只含一个数字。
- **`multi` 模式（多数字识别）**：文件中含多个数字，VAD拆分后逐个识别。

---

## 实验过程与结果

### 数据集

使用 [0-9数字语音库](https://gitcode.com/open-source-toolkit/b2012/?utm_source=tools_gitcode&index=top&type=card) 中的数据作为训练集，测试集为自录语音，统一采样率为16kHz。

### 训练过程

- VAD 提取语音段  
- 提取 LPC 和 FFT 倒谱及 Delta 特征  
- 重采样所有特征到统一长度  
- 使用 DBA 构建稳健模板  
- 模板保存在 `templates.pkl` 中

### 识别与评估

#### 单数字识别 (`--mode single`)

- 准确率：**44.00%**

#### 多数字识别 (`--mode multi`)

- 整体序列准确率：**50.00%**（10个中有5个序列完全正确）
- 单个数字准确率：**58.33%**

---

## 结论

本实验成功实现了一个基于DTW的数字语音识别系统，展示了在单数字和多数字识别上的基本功能。

---
