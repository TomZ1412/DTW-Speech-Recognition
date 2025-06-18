import os
import librosa
import pickle
import numpy as np
from scipy.interpolate import interp1d
# 假设 vad_energy_tcr 和 extract_features 在 utils.py 中
# Ensure vad_energy_tcr and extract_features are in utils.py
from utils import vad_energy_tcr, extract_features 
import soundfile as sf # 用于创建虚拟文件
from utils import resample_wav
from tslearn.barycenters import dtw_barycenter_averaging as DBA

def resample_features(features, target_length):
    """
    将特征矩阵重采样到目标帧数，使用线性插值。
    
    参数:
        features (np.ndarray): 输入特征矩阵 (当前帧数, 系数数量)。
        target_length (int): 输出所需的帧数。
        
    返回:
        np.ndarray: 重采样后的特征矩阵 (目标帧数, 系数数量)。
    """
    # 如果特征帧数与目标帧数相同，直接返回
    if features.shape[0] == target_length:
        return features
    
    current_length = features.shape[0]
    num_coeffs = features.shape[1]
    
    resampled_features = np.zeros((target_length, num_coeffs))
    
    # 为原始序列创建插值点（x轴，表示帧索引）
    x_old = np.linspace(0, current_length - 1, current_length)
    # 为新序列创建插值点（x轴，表示新的帧索引）
    x_new = np.linspace(0, current_length - 1, target_length)
    
    # 遍历每个特征维度（即每个系数），进行独立插值
    for i in range(num_coeffs):
        # 创建一个插值函数
        # kind='linear' 表示线性插值
        # fill_value="extrapolate" 允许在原始数据范围外进行推断，以防万一
        f_interp = interp1d(x_old, features[:, i], kind='linear', fill_value="extrapolate")
        resampled_features[:, i] = f_interp(x_new)
        
    return resampled_features


def build_templates(data_dir):
    """
    通过对同一数字的多个录音的特征进行平均，构建稳健的 DTW 模板。
    
    参数:
        data_dir (str): 包含用于模板构建的 WAV 文件的目录。
                        假设文件命名格式为 '{digitLabel}_{speakerName}_{index}.wav'。
    
    返回:
        None: 将模板保存到 'templates.pkl'。
    """
    # 用于存储每个数字收集到的所有 LPC 和 FFT 特征序列
    collected_lpc_features = {digit: [] for digit in range(10)}
    collected_fft_features = {digit: [] for digit in range(10)}

    print(f"开始扫描目录: {data_dir} 中的音频文件...")
    for label in range(10):
        dir = os.path.join(data_dir, str(label))
        # import pdb; pdb.set_trace()
        for filename in os.listdir(dir):
            # 过滤 WAV 文件，假设 FSDD 命名约定（例如 '1_jackson_0.wav'）
            if filename.endswith(".wav"): 
                try:
                    # # 从文件名中提取数字标签（例如 "1_jackson_0.wav" -> 1）
                    # label_str = filename.split('_')[0]
                    # label = int(label_str)

                    # # 确保标签在 0-9 范围内
                    # if label not in range(10): 
                    #     print(f"跳过文件 {filename}: 标签 {label} 不是 0-9 之间的数字。")
                    #     continue

                    filepath = os.path.join(dir, filename)
                    y, sr = librosa.load(filepath, sr=None)
                    # # y = y.astype(np.float32)  # 确保音频数据为 float32 类型
                    # # 执行 VAD 获取语音片段
                    # # 批量处理时，不显示 VAD 曲线图，避免弹出过多窗口
                    start, end = vad_energy_tcr(y, sr, plot=False) 
                    speech_segment = y[start:end]
                    # speech_segment = y

                    # 如果 VAD 没有检测到有效语音，则跳过此文件
                    if len(speech_segment) == 0:
                        print(f"警告: 文件 {filename} 未检测到有效语音活动，跳过特征提取。")
                        continue

                    # 从语音片段中提取特征
                    feat_lpc, feat_fft = extract_features(speech_segment, sr)
                    # if None in (feat_lpc, feat_fft):
                    #     print(f"警告: 文件 {filename} 特征提取失败，跳过。")
                    #     continue

                    # 确保提取的特征非空
                    if feat_lpc.shape[0] > 0 and feat_fft.shape[0] > 0:
                        collected_lpc_features[label].append(feat_lpc)
                        collected_fft_features[label].append(feat_fft)
                        print(f"已处理文件: {filename}, 标签: {label}, LPC特征形状: {feat_lpc.shape}, FFT特征形状: {feat_fft.shape}")
                    else:
                        print(f"警告: 文件 {filename} 提取到零长度特征，跳过。")

                except ValueError:
                    print(f"错误: 无法从文件名 {filename} 提取数字标签，跳过。")
                    continue
                except Exception as e:
                    print(f"处理文件 {filename} 时发生错误: {e}。 跳过。")
                    continue

    # 现在，处理收集到的特征以创建稳健模板
    templates = {}
    print("\n开始生成稳健模板...")
    for digit in range(10):
        lpc_sequences = collected_lpc_features[digit]
        fft_sequences = collected_fft_features[digit]

        if not lpc_sequences or not fft_sequences:
            print(f"警告: 未找到数字 {digit} 的足够录音。将不创建模板。")
            continue

        # 确定重采样的目标帧长（例如，使用中位数帧长以减少异常值影响）
        lpc_lengths = [seq.shape[0] for seq in lpc_sequences]
        fft_lengths = [seq.shape[0] for seq in fft_sequences]
        
        target_lpc_length = int(np.median(lpc_lengths)) if lpc_lengths else 0
        target_fft_length = int(np.median(fft_lengths)) if fft_lengths else 0

        if target_lpc_length == 0 or target_fft_length == 0:
            print(f"警告: 数字 {digit} 的目标帧长为零，跳过模板创建。")
            continue
            
        resampled_lpc_sequences = []
        for seq in lpc_sequences:
            if seq.shape[0] > 0: # 确保序列非空，避免对空序列进行重采样
                resampled_lpc_sequences.append(resample_features(seq, target_lpc_length))
            else:
                # 如果原始序列为空（虽然上面已经过滤），则添加一个零填充的序列
                resampled_lpc_sequences.append(np.zeros((target_lpc_length, feat_lpc.shape[1])))
                
        resampled_fft_sequences = []
        for seq in fft_sequences:
            if seq.shape[0] > 0:
                resampled_fft_sequences.append(resample_features(seq, target_fft_length))
            else:
                resampled_fft_sequences.append(np.zeros((target_fft_length, feat_fft.shape[1])))

        # 对重采样后的序列进行逐元素平均，生成稳健模板
        # robust_lpc_template = np.mean(resampled_lpc_sequences, axis=0)
        # robust_fft_template = np.mean(resampled_fft_sequences, axis=0)
        robust_lpc_template = DBA(np.array(resampled_lpc_sequences,dtype=object), max_iter=10)
        robust_fft_template = DBA(np.array(resampled_fft_sequences,dtype=object), max_iter=10)
        
        templates[digit] = (robust_lpc_template, robust_fft_template)
        print(f"数字 {digit} 模板已创建。LPC 模板形状: {robust_lpc_template.shape}, FFT 模板形状: {robust_fft_template.shape}")

    # 保存稳健模板到文件
    template_filename = "templates.pkl"
    with open(template_filename, "wb") as f:
        pickle.dump(templates, f)
    print(f"\n稳健模板已保存到 {template_filename}")


if __name__ == "__main__":
    train_data_dir = ""

    print("\n--- 开始构建稳健模板 ---")
    build_templates(train_data_dir)

    # 验证创建的模板（可选）
    try:
        with open("templates.pkl", "rb") as f:
            loaded_templates = pickle.load(f)
        
        print("\n--- 已加载模板验证 ---")
        for digit, (lpc_temp, fft_temp) in loaded_templates.items():
            print(f"数字 {digit}: LPC 模板形状: {lpc_temp.shape}, FFT 模板形状: {fft_temp.shape}")
    except FileNotFoundError:
        print("templates.pkl 文件未找到，请检查模板是否成功生成。")
    except Exception as e:
        print(f"加载或验证模板时发生错误: {e}")