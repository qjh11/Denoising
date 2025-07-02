import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy import signal  # 注意导入scipy的signal模块
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体（解决中文乱码问题）
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统常用字体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号显示异常
plt.rcParams.update({'font.size': 14})  # 全局字体大小
# 定义去均值、去趋势并归一化的函数
def demean_detrended_normalize_time_series(series):
    """去均值、去趋势并归一化"""
    mean_val = np.mean(series)
    demeaned_series = series - mean_val
    detrended_series = signal.detrend(demeaned_series)
    # norm to -1 and 1
    min_val = np.min(detrended_series)
    max_val = np.max(detrended_series)
    normalized_series = (detrended_series - min_val) / (max_val - min_val)  # Normalize to [0, 1]
    normalized_series = normalized_series * 2 - 1  # Rescale to [-1, 1]
    return normalized_series

# 自适应阈值计算函数
def adaptive_threshold_calculation(coeffs):
    thresholds = []
    for j in range(len(coeffs)):
        wavelet_coeffs = coeffs[j]  # 获取每一层的小波系数，包括第零层的近似系数
        N_j = len(wavelet_coeffs)
        median_val = np.median(np.abs(wavelet_coeffs))  # 计算系数的中值
        delta_j = median_val / 0.6745  # 标准差估计
        # lambda_j = (delta_j * np.sqrt(2 * np.log(N_j))) / np.sqrt(j + 1)  # 阈值计算
        lambda_j = (delta_j * np.sqrt(2 * np.log(N_j))) / np.power(j, 1 / (j + 1))  # 阈值计算
        thresholds.append(lambda_j)  # 记录每一层的阈值
    return thresholds


# 自定义阈值函数的应用
def apply_threshold_function(coeffs, thresholds, alpha=0.1, beta=2.0):
    denoised_coeffs = []
    for j, detail_coeffs in enumerate(coeffs):
        threshold = thresholds[j]
        denoised_detail = np.zeros_like(detail_coeffs)
        for i in range(len(detail_coeffs)):
            if detail_coeffs[i] > threshold:
                denoised_detail[i] = detail_coeffs[i] - threshold / (alpha * (detail_coeffs[i] - threshold) ** beta + 1)
            elif detail_coeffs[i] < -threshold:
                denoised_detail[i] = detail_coeffs[i] + threshold / (alpha * (-detail_coeffs[i] - threshold) ** beta + 1)
            else:
                denoised_detail[i] = 0
        denoised_coeffs.append(denoised_detail)
    return denoised_coeffs

def calculate_snr(noisy, denoised):
    noise = noisy - denoised
    snr = 10 * np.log10(np.sum(denoised ** 2) / np.sum(noise ** 2))
    return snr


# # 计算信噪比有干净的信号（SNR）
# def calculate_snr(original, denoised):
#     noise = original - denoised
#     snr = 10 * np.log10(np.sum(original ** 2) / np.sum(noise ** 2))
#     return snr

def train_threshold(file_path):
    data = np.load(file_path)
    # data = data.T  # 转置操作
    noisy_signal = data[0, :]  # 提取第一行数据
    # noisy_signal = data[:]
    # noisy_signal = np.load(file_path)
#     # 加载原始信号和含噪信号
    original_signal = np.load(r'C:/Users/DELL/PycharmProjects/pythonProject1/DL_denoise/17055452/Example_Data.npy')
#     # noisy_signal = np.load(r'C:\Users\DELL\Desktop\有代码论文\小波变换\Denoising-BTwavelet-master\ex_synth+Noise.npy')
# # # 对齐原始信号和噪声信号长度
#     original_signal = original_signal[:1000]
#     noisy_signal = noisy_signal[:1000]
# # # 确保信号是 1D 数组
#     original_signal = original_signal.squeeze()
#     noisy_signal = noisy_signal.squeeze()

# 信号归一化处理
    normalized_noisy_signal = demean_detrended_normalize_time_series(noisy_signal)

    # 小波分解设置
    wavelet = 'db4'
    levels = 2
    coeffs = pywt.wavedec(normalized_noisy_signal, wavelet, level=levels)

    # 提取近似系数和细节系数
    approximation_coeffs = coeffs[0]
    detail_coeffs = coeffs[1:]


# 计算自适应阈值
    thresholds = adaptive_threshold_calculation(detail_coeffs)

    # 应用阈值函数
    denoised_detail_coeffs = apply_threshold_function(detail_coeffs, thresholds)

    # 替换细节系数为去噪后的系数
    denoised_coeffs_full = [approximation_coeffs] + denoised_detail_coeffs

    # 使用逆小波变换重构信号
    denoised_normalized_signal = pywt.waverec(denoised_coeffs_full, wavelet)

    # 反归一化
    min_val = np.min(noisy_signal)
    max_val = np.max(noisy_signal)
    denoised_signal = 0.5 * (denoised_normalized_signal + 1) * (max_val - min_val) + min_val

# # 绘制原始信号
#     plt.switch_backend('TkAgg')
#     plt.figure(figsize=(12, 4))
#     plt.plot(original_signal, label="Original Signal", color="blue", linestyle="-")
#     plt.legend()
#     plt.title("Original Signal")
#     plt.xlabel("Sample Index")
#     plt.ylabel("Amplitude")
#     plt.show()

# # 绘制含噪信号
#     plt.figure(figsize=(12, 4))
#     plt.plot(normalized_noisy_signal, label="Noisy Signal", color="orange", alpha=0.7)
#     plt.legend()
#     plt.title("Noisy Signal")
#     plt.xlabel("Sample Index")
#     plt.ylabel("Amplitude")
#     plt.ylim(-1, 1)
#     plt.show()
#
#     # 绘制去噪后的信号
#     plt.figure(figsize=(12, 4))
#     plt.plot(denoised_normalized_signal, label="Denoised Signal", color="green", alpha=0.8)
#     plt.legend()
#     plt.title("Denoised Signal")
#     plt.xlabel("Sample Index")
#     plt.ylabel("Amplitude")
#     plt.ylim(-1, 1)
#     plt.show()

    # 绘制含噪信号
    plt.figure(figsize=(12, 4))
    plt.plot(normalized_noisy_signal, label="Noisy Signal", color="tab:blue", alpha=0.8)
    plt.legend()
    plt.title("Noisy Signal", fontsize=14)
    plt.xlabel("Sample Index", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.ylim(-1, 1)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)  # 添加网格线
    plt.minorticks_on()  # 开启次网格
    plt.tick_params(axis='both', labelsize=8, direction='in',length=4)  # 调整刻度样式
    plt.tight_layout()
    plt.show()
    # 绘制去噪后的信号
    plt.figure(figsize=(12, 4))
    plt.plot(denoised_normalized_signal, label="Denoised Signal", color="tab:blue", alpha=0.8)
    plt.legend()
    plt.title("Denoised Signal", fontsize=14)
    plt.xlabel("Sample Index", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.ylim(-1, 1)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)  # 添加网格线
    plt.minorticks_on()  # 开启次网格
    plt.tick_params(axis='both', labelsize=10, direction='in',length=4)  # 调整刻度样式
    plt.tight_layout()
    plt.show()

    # # 可视化原始信号与加噪信号
    # plt.figure(figsize=(15, 5), dpi=300)
    # plt.plot(denoised_signal, label="深度学习去噪\nSNR=8.64dB", color='blue')
    # plt.legend()
    # plt.xlabel("时间")
    # plt.ylabel("幅值")
    # plt.show()

    # # 输出去噪后的信噪比
    noisy_signal = noisy_signal[:6000]
    denoised_signal = denoised_signal[:6000]
    # 输出去噪后的信噪比
    snr_value = calculate_snr(noisy_signal, denoised_signal)
    #     snr_value = calculate_snr(original_signal, denoised_signal)
    print("去噪后的信噪比（SNR）:", snr_value)

# 使用方法示例：
if __name__ == "__main__":
    file_path = r'C:\Users\DELL\Desktop\single/noisy_singles.npy'
    train_threshold(file_path)
