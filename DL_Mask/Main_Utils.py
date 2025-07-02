import matplotlib.pyplot as plt
from Utils import *
from Utils import yc_patch_inv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from scipy import signal
from Utils import butter_bandpass_filter_zi, yc_patch

# 定义 CompactLayer 类
class CompactLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CompactLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

# def calculate_snr(original, rec):
#     noise = original - rec
#     snr = 10 * np.log10(np.sum(original ** 2) / np.sum(noise ** 2))
#     return snr

def calculate_snr(noisy_signal, rec):
    noise = noisy_signal - rec
    snr = 10 * np.log10(np.sum(rec ** 2) / np.sum(noise ** 2))
    return snr


# 自编码器模型
class Autoencoder(nn.Module):
    def __init__(self, input_size, D1, D2, D3, D4, D5, D6):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            CompactLayer(2, D1),  # 输入的维度是2（实部和虚部）
            CompactLayer(D1, D2),
            CompactLayer(D2, D3),
            CompactLayer(D3, D4),
            CompactLayer(D4, D5),
            CompactLayer(D5, D6)
        )
        self.decoder = nn.Sequential(
            CompactLayer(D6, D5),
            CompactLayer(D5, D4),
            CompactLayer(D4, D3),
            CompactLayer(D3, D2),
            CompactLayer(D2, D1),
            CompactLayer(D1, D1),
            nn.Flatten(),
            nn.Linear(D1 * input_size, input_size),  # 将卷积输出展平成线性层
            nn.Softplus()  # 使用Softplus激活函数
        )

    def forward(self, x1, x2):
        # 输入是实部和虚部
        x1 = x1.unsqueeze(1)  # 增加通道维度
        x2 = x2.unsqueeze(1)  # 增加通道维度
        input_img3 = torch.cat((x1, x2), dim=1)  # 在通道维度上拼接

        # 编码器
        encoded = self.encoder(input_img3)

        # 解码器
        decoded = self.decoder(encoded)

        return decoded


# 自定义损失函数
def mask_loss(y_true, y_pred):
    nois = y_true - y_pred
    maskbiS = 1 / (1 + torch.abs(nois) / (torch.abs(y_pred) + 1e-8))  # 防止除零错误
    return torch.abs(1 - maskbiS).mean()  # 平均损失

# 6. 绘制图形
def plot_results(dnPatchA, outA,outAM, dn, rec):
    plt.figure()
    plt.imshow(dnPatchA, aspect='auto')
    plt.show()

    plt.figure()
    plt.imshow(outA, aspect='auto')
    plt.show()

    plt.figure()
    plt.imshow(outAM, aspect='auto')
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(dn, color='tab:blue', linewidth=1.0)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.minorticks_on()  # 开启次网格
    plt.tick_params(axis='both', direction='in', length=4, labelsize=8)  # 调整刻度样式
    plt.ylim()  # 根据信号范围设置Y轴
    plt.xlim(0, len(dn))  # 根据数据长度设置X轴
    plt.legend(loc='upper right', fontsize=8)
    # 调整布局并显示
    plt.tight_layout()
    plt.show()

    # 绘制图像
    plt.figure(figsize=(12, 4))  # 设置图像的宽扁比例
    plt.plot(rec, label='Reconstructed Signal after Denoising', color='tab:blue', linewidth=1.0)
    # 添加标题和标签
    plt.title('Reconstructed Signal after Denoising', fontsize=12, pad=10)
    plt.xlabel('', fontsize=10)  # 如果不需要，可以留空
    plt.ylabel('', fontsize=10)
    # 设置网格线与次网格
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.minorticks_on()  # 开启次网格
    plt.tick_params(axis='both', direction='in', length=4, labelsize=8)  # 调整刻度样式
    # 设置坐标轴范围
    plt.ylim()  # 根据信号范围设置Y轴
    plt.xlim(0, len(rec))  # 根据数据长度设置X轴
    # 显示图例
    plt.legend(loc='upper right', fontsize=8)
    # 调整布局并显示
    plt.tight_layout()
    plt.show()

def train_Main_Utils(file_path):
    data = np.load(file_path)
    # data = data.T  # 提取第一行数据
    # dn = data[0, :]  # 提取第一行数据
    dn = data[:]
    #dn = np.load('C:/Users/DELL/PycharmProjects/pythonProject1/DL_denoise/17055452/Example_Data.npy')
    dn =  butter_bandpass_filter_zi(dn, 1, 45, 100, order=10)
    fs = 100
    le = 20
    leov = int(le/2)
    f, t, Zxx = signal.stft(dn, fs=100, window='hann', nperseg=le, noverlap=leov, nfft=2*le, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=- 1)
    dnPatchR = np.real(Zxx) #实部
    dnPatchI = np.imag(Zxx) #虚部
    dnPatchA = np.abs(Zxx) #振幅
    if np.max(dnPatchA)>1: #归一化
      maA = np.max(np.abs(dnPatchA))
      maR = np.max(np.abs(dnPatchR))
      maI = np.max(np.abs(dnPatchI))
      dnPatchR = dnPatchR/np.max(np.abs(dnPatchR))
      dnPatchI = dnPatchI/np.max(np.abs(dnPatchI))
      dnPatchA = dnPatchA/np.max(np.abs(dnPatchA))

    # Patching
    w1 =8
    w2 =8 #窗口大小为 8 行、8 列
    s1z =1
    s2z =1 #滑动窗口的步长为 1，表示每次移动 1 行和 1 列
    dn_patchA = yc_patch(dnPatchA,w1,w2,s1z,s2z)
    dn_patchR = yc_patch(dnPatchR,w1,w2,s1z,s2z)
    dn_patchI = yc_patch(dnPatchI,w1,w2,s1z,s2z) # 分块操作

    # 加载数据
    dataNoiseR = torch.tensor(dn_patchR, dtype=torch.float32)
    dataNoiseI = torch.tensor(dn_patchI, dtype=torch.float32)
    dataNoiseA = torch.tensor(dn_patchA, dtype=torch.float32)

    # 输入大小和模型参数
    INPUT_SIZE2 = dataNoiseR.shape[1]
    D1 = 32
    D2 = int(D1 / 2)
    D3 = int(D2 / 2)
    D4 = int(D3 / 2)
    D5 = int(D4 / 2)
    D6 = int(D5 / 2)

    # 初始化模型
    autoencoder = Autoencoder(INPUT_SIZE2, D1, D2, D3, D4, D5, D6)
    # 打印模型结构

    # 使用 Adam 优化器
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.0001)

    # 使用 TensorDataset 和 DataLoader 来加载数据
    dataset = TensorDataset(dataNoiseR, dataNoiseI, dataNoiseA)
    batch_size = int(np.ceil(len(dataNoiseR) * 0.005))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 训练模型
    epochs =4
    early_stop_patience = 5
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        autoencoder.train()

        for batch_r, batch_i, batch_a in dataloader:
            optimizer.zero_grad()

            # 前向传播
            outputs = autoencoder(batch_r, batch_i)

            # 计算损失
            loss = mask_loss(batch_a, outputs)
            epoch_loss += loss.item()

            # 反向传播和优化
            loss.backward()
            optimizer.step()

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}')
        # Early Stopping
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(autoencoder.state_dict(), 'best_model_STFT_STEAD_Real_10.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping...")
                break

    # Predict 预测信号幅值 xA
    # 加载 PyTorch 模型
    autoencoder = Autoencoder(INPUT_SIZE2, D1, D2, D3, D4, D5, D6)
    autoencoder.load_state_dict(torch.load('best_model_STFT_STEAD_Real_10.pth'))
    autoencoder.eval()  # 设置模型为评估模式

    with torch.no_grad():
        xA = autoencoder(dataNoiseR, dataNoiseI)
    print(xA)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    X1 = torch.transpose(xA, 0, 1).detach().numpy()  # 转置并转换为 numpy 数组（如果后续处理需要）
    n1, n2 = dnPatchA.shape  # 假设 dnPatchA 是一个 numpy 数组
    outA = yc_patch_inv(X1, n1, n2, w1, w2, s1z, s2z)
    outA = np.array(outA)


    # Thresholding 阈值处理与滤波
    f, t, Zxxdn = signal.stft(dn, fs=100, window='hann', nperseg=le, noverlap=leov, nfft=2*le, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=- 1)
    ZxxdnR = np.real(Zxxdn)
    ZxxdnI = np.imag(Zxxdn)
    yy = 1
    sizmedian = 15
    if 'maA' not in locals():
        maA = np.max(np.abs(dnPatchA))  # 计算 dnPatchA 的最大值作为 maA
    outAM = (outA * maA)
    outAM = outAM/np.max(np.abs(outAM))

    for iu in range(outAM.shape[0]):
        tmp = np.copy(outAM[iu,:])
        #tmp[tmp>=1*np.mean(tmp)]=1
        tmp[tmp<1*np.mean(tmp)]=1e-10
        outAM[iu,:] = tmp
    mea = np.mean(outAM)
    outAM[outAM>=yy*mea]=1
    outAM[outAM<yy*mea]=1e-10
    outAM = ndimage.median_filter(outAM, size=sizmedian)

    # 获取第10列的掩膜值
    print("Values in the 10th column of the Mask:")
    print(outAM[:, 9])

    # 信号重构
    dn_mask = np.real(Zxxdn)*outAM + 1j*np.imag(Zxxdn)*outAM
    # rec = signal.istft(dn_mask,  fs=100, window='hann', nperseg=le, noverlap=leov, nfft=2*le, boundary='zeros')
    _, rec = signal.istft(dn_mask, fs=fs, window='hann', nperseg=le, noverlap=leov, nfft=2 * le, boundary='zeros')
    rec = np.array(rec)
    # 绘图
    plot_results(dnPatchA, outA,outAM, dn, rec)
    # plt.plot(rec[1,:])
    # plt.figure()
#有原始信号
    # snr_value = calculate_snr(original_signal, rec)
    snr_value = calculate_snr(dn, rec)
    print("去噪后的信噪比（SNR）:", snr_value)



if __name__ == "__main__":
     file_path = r'C:/Users/DELL/PycharmProjects/pythonProject1/DL_denoise/17055452/Example_Data.npy'
     # original_signal = np.load(r'C:\Users\DELL\Desktop\有代码论文\小波变换\Denoising-BTwavelet-master\ex_synth.npy')
     # original_signal = original_signal.T[0, :1000]
     train_Main_Utils(file_path)

    #
    # P = np.load('C:/Users/DELL/PycharmProjects/pythonProject1/DL_denoise/17055452/Example_Data_P.npy')
    # S = np.load('C:/Users/DELL/PycharmProjects/pythonProject1/DL_denoise/17055452/Example_Data_S.npy')
    #
    # font = {'family' : 'normal',
    #         'weight' : 'bold',
    #         'size'   : 16}
    #
    # plt.rc('font', **font)
    #
    # x = np.arange(len(dn))
    # fig = plt.figure(figsize=(20,5))
    # ax1 = plt.subplot(2,3,1)
    # plt.plot(x,dn)
    # plt.title('Seismic Data')
    # plt.xlim([0,3000])
    # plt.setp(ax1.get_xticklabels(), visible=False)
    #
    # ax2 = plt.subplot(2,3,2)
    # plt.plot(x[P-200:P+200],dn[P-200:P+200])
    # ymin, ymax = plt.ylim()
    # plt.vlines(P, ymin, ymax, color='r', linewidth=2)
    # plt.title('P-wave zoom')
    # plt.xlim([P-200,P+200])
    # plt.setp(ax2.get_xticklabels(), visible=False)
    #
    # ax3 = plt.subplot(2,3,3)
    # plt.plot(x[S-200:S+200], dn[S-200:S+200])
    # ymin, ymax = plt.ylim()
    # plt.vlines(S, ymin, ymax, color='r', linewidth=2)
    # plt.title('S-wave zoom')
    # plt.xlim([S-200,S+200])
    # plt.setp(ax3.get_xticklabels(), visible=False)
    #
    # ax4 = plt.subplot(2,3,4, sharex=ax1)
    # plt.plot(rec[1,:])
    # plt.xlabel('Samples',fontsize='large', fontweight='bold')
    # plt.xlim([0,3000])
    #
    # ax5 = plt.subplot(2,3,5, sharex=ax2)
    # plt.plot(x[P-200:P+200],rec[1,P-200:P+200])
    # ymin, ymax = plt.ylim()
    # plt.vlines(P, ymin, ymax, color='r', linewidth=2)
    # plt.xlabel('Samples',fontsize='large', fontweight='bold')
    # plt.xlim([P-200,P+200])
    #
    # ax6 = plt.subplot(2,3,6, sharex=ax3)
    # plt.plot(x[S-200:S+200],rec[1,S-200:S+200])
    # ymin, ymax = plt.ylim()
    # plt.vlines(S, ymin, ymax, color='r', linewidth=2)
    # plt.xlabel('Samples',fontsize='large', fontweight='bold')
    # plt.xlim([S-200,S+200])
    # plt.show()