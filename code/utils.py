import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import soundfile as sf
import os
import librosa
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

def vad_energy_tcr(signal, fs, mode, plot=False):
    signal = signal - np.mean(signal)  # 消除直流分量
    frame_len = int(0.02 * fs)        # 帧长 20ms
    frame_shift = int(0.01 * fs)      # 帧移 10ms
    energy_threshold = 0.01            # 能量阈值（归一化后）
    tcr_thres = 0.01                # 过门限率阈值（归一化后）
    
    # 计算背景噪声的RMS（假设前100ms是噪声）
    noise_segment = signal  # 取前100ms作为噪声段
    if len(noise_segment) == 0:
        noise_rms = 0.001  # 避免除零
    else:
        noise_rms = np.sqrt(np.mean(noise_segment ** 2))  # 噪声的RMS
    tcr_threshold = 2 * noise_rms     # 过门限率阈值（噪声RMS的2倍）

    n_frames = (len(signal) - frame_len) // frame_shift + 1
    energy = np.zeros(n_frames)
    tcr = np.zeros(n_frames)          # 过门限率（Threshold Crossing Rate）

    for i in range(n_frames):
        frame = signal[i * frame_shift : i * frame_shift + frame_len]
        rate_frame = frame  # 初始化过门限率数组
        energy[i] = np.sum(frame ** 2)  # 短时能量
        # 过门限率：统计幅度超过tcr_threshold的点数，并归一化
        rate_frame[np.abs(frame)<=tcr_threshold]=0
        rate_frame[rate_frame>tcr_threshold]=rate_frame[rate_frame>tcr_threshold]-tcr_threshold
        rate_frame[rate_frame<-tcr_threshold]=rate_frame[rate_frame<-tcr_threshold]+tcr_threshold
        tcr[i] = np.sum(abs(np.diff(np.sign(rate_frame)))) / 2
        # tcr[i] = np.sum(np.abs(frame) > tcr_threshold) / frame_len

    # 归一化
    energy /= np.max(energy)
    tcr /= np.max(tcr) if np.max(tcr) > 0 else 1  # 避免除零

    # 语音活动检测
    vad = (energy > energy_threshold) | (tcr > tcr_thres)  # TCR阈值设为0.1（经验值）
    vad = medfilt(vad.astype(float), kernel_size=5)  # 中值滤波平滑

    # 绘制能量和过门限率曲线
    if plot:
        time_axis = np.arange(n_frames) * frame_shift / fs  # 时间轴（秒）
        full_time = np.arange(len(signal)) / fs

        plt.figure(figsize=(12, 6))

        # 绘制原始信号
        plt.plot(full_time, signal / np.max(np.abs(signal)), label="Normalized Signal", color='gray', alpha=0.7)

        # 绘制能量
        plt.plot(time_axis, energy, label="Energy", color='blue', linewidth=2)

        # 绘制过门限率 TCR
        plt.plot(time_axis, tcr, label="TCR", color='orange', linewidth=2)

        # 阈值线
        plt.axhline(y=energy_threshold, color='blue', linestyle='--', label="Energy Threshold")
        plt.axhline(y=tcr_thres, color='orange', linestyle='--', label="TCR Threshold")

        plt.title("Speech Activity Detection (Signal, Energy, TCR)")
        plt.xlabel("Time (s)")
        plt.ylabel("Normalized Values")
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # 计算语音段的起始和结束位置
    idx = np.where(vad > 0)[0]
    if len(idx) == 0:
        return 0, len(signal)
    else:
        start_list = []
        end_list = []
        if mode == "single":
            start = idx[0] * frame_shift
            end = min(len(signal), idx[-1] * frame_shift + frame_len)
            start_list.append(start)
            end_list.append(end)
            return start_list, end_list
        elif mode == "multi":
            # import pdb; pdb.set_trace()
            prev = -2
            for i in range(len(idx)):
                # import pdb; pdb.set_trace()
                if i == 0 or idx[i] != prev + 1:
                # 一个新段开始
                    start = idx[i] * frame_shift
                if i == len(idx) - 1 or idx[i+1] != idx[i] + 1:
                # 一个段结束
                    end = min(len(signal), (idx[i] + 1) * frame_shift + frame_len)
                    if end - start > 10 * frame_len:
                        start_list.append(start)
                        end_list.append(end)
                prev = idx[i]
            return start_list, end_list
            

def calculate_delta(features, N=2):
    """
    计算特征的Delta（一阶差分）特征。
    参考公式：d_t = (sum_{k=1}^{N} k * (c_{t+k} - c_{t-k})) / (2 * sum_{k=1}^{N} k^2)
    其中 c_t 是当前帧的特征向量。
    边界处理：在序列两端，使用填充或重复边界值来计算差分。
    """
    num_frames, num_coeffs = features.shape
    delta_features = np.zeros_like(features)

    denominator = 2 * np.sum(np.arange(1, N + 1)**2)

    for t in range(num_frames):
        for k in range(1, N + 1):
            # 处理边界：如果 t+k 或 t-k 超出范围，使用最近的有效帧
            c_plus_k = features[min(t + k, num_frames - 1)]
            c_minus_k = features[max(t - k, 0)]
            delta_features[t] += k * (c_plus_k - c_minus_k)
        delta_features[t] /= denominator
    return delta_features

def durbin(r, order):
    a = np.zeros(order + 1)
    e = r[0]
    k = np.zeros(order)

    for i in range(1, order + 1):
        acc = r[i] + np.dot(a[1:i], r[i-1:0:-1])
        # acc = r[i] + np.sum(a[1:i] * r[i - 1::-1][:i-1])
        k[i - 1] = -acc / e
        a_new = a.copy()
        a_new[1:i] += k[i - 1] * a[i - 1:0:-1]
        a_new[i] = k[i - 1]
        a = a_new
        e *= 1 - k[i - 1] ** 2
    # print(a[1:],e)
    return a[1:], e

def lpc_cepstrum(signal, order=16):
    signal.astype(float)  # 确保信号是浮点型
    r = np.correlate(signal, signal, mode='full')
    r = r[len(signal)-1 : len(signal)-1+order+1]
    a, _ = durbin(r, order)

    cep = np.zeros(order)
    cep[0] = -np.log(r[0]+1e-10)  # Adding a small value to avoid log(0)
    for n in range(1, order):
        s = 0.0
        for k in range(1, n):
            s += (k / n) * cep[k] * a[n - k - 1]
        cep[n] = a[n - 1] + s
    return cep

def fft_cepstrum(signal,num_coeffs=20):
    fft_signal = np.fft.fft(signal)
    cepstrum = np.fft.ifft(np.log(np.abs(fft_signal))+1e-10)  # Adding a small value to avoid log(0)
    return cepstrum.real[:num_coeffs]

def dtw_distance(s1, s2):
    scaler_s1 = StandardScaler()
    s1 = scaler_s1.fit_transform(s1)
    
    scaler_s2 = StandardScaler()
    s2 = scaler_s2.fit_transform(s2)
    # import pdb; pdb.set_trace()
    N, M = len(s1), len(s2)
    D = np.full((N + 1, M + 1), np.inf)
    D[0, 0] = 0

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost = np.linalg.norm(s1[i - 1] - s2[j - 1]) # 计算当前帧的距离
            
            # 标准 DTW 路径选择
            # D[i, j] = cost + min(D[i-1, j],      # 从上方（s1 扩展）
            #                      D[i, j-1],      # 从左方（s2 扩展）
            #                      D[i-1, j-1])
            # 从左上方（s1 和 s2 同时进展）
            if j > 1:
                D[i, j] = cost + min(D[i-1, j],      # 从上方（s1 扩展）    # 从左方（s2 扩展）
                                 D[i-1, j-1],
                                 D[i-1,j-2])    # 从左上方（s1 和 s2 同时进展）
            else:
                D[i, j] = cost + min(D[i-1, j],      # 从上方（s1 扩展）    # 从左方（s2 扩展）
                                 D[i-1, j-1])

    return D[N, M]

def extract_features(signal, fs):
    frame_len = int(0.02 * fs)
    frame_shift = int(0.01 * fs)
    frames_lpc = []
    frames_fft = []
    # features=[]
    for i in range(0, len(signal) - frame_len, frame_shift):
        frame = signal[i:i + frame_len]
        frame = frame * np.hanning(frame_len)  # Apply Hanning window
        feat_lpc = lpc_cepstrum(frame, order=16)
        feat_fft = fft_cepstrum(frame, num_coeffs=20)
        frames_lpc.append(feat_lpc)
        frames_fft.append(feat_fft)
        
        # frame_features = np.concatenate((feat_lpc, feat_fft))
        # features.append(frame_features)
    frames_lpc = np.array(frames_lpc)
    frames_fft = np.array(frames_fft)
    
    delta_features_lpc = calculate_delta(frames_lpc, N=2)
    delta_features_fft = calculate_delta(frames_fft, N=2)
    
    frames_lpc = np.concatenate((frames_lpc, delta_features_lpc), axis=1)
    frames_fft = np.concatenate((frames_fft, delta_features_fft), axis=1)
    return frames_lpc, frames_fft
    # return features  # Return as a single feature vector

def resample_wav(input_filepath, target_fs):
    # import pdb; pdb.set_trace()
    audio_data, original_sample_rate = librosa.load(input_filepath, sr=None)
    resampled_audio = librosa.resample(y=audio_data, orig_sr=original_sample_rate, target_sr=target_fs)
    sf.write(input_filepath.replace(".wav", f"_{target_fs}hz.wav"), resampled_audio, target_fs)
    
if __name__ == "__main__":

    signal, fs = librosa.load('YOUR_DATA_DIR', sr=None)
    start,end = vad_energy_tcr(signal, fs, plot=True)
