import cv2
import numpy as np
import matplotlib.pyplot as plt
from latent_lrr import latent_lrr  # 假设latent_lrr函数所在的模块已经正确导入或者在同一个文件中定义了

# 模拟Matlab中的循环，这里只处理index为2的情况，若需要循环处理多个，可以添加循环逻辑
index = 2
path1 = f'./source_images/IV_images/IR{index}.jpg'
path2 = f'./source_images/IV_images/VIS{index}.jpg'
fuse_path = f'./fused_images/fused{index}_latlrr.jpg'

# 读取图像
image1 = cv2.imread(path1)
image2 = cv2.imread(path2)

# 检查图像是否成功读取
if image1 is None:
    print("无法读取图像，请检查图像路径是否正确！")
    exit()

# 创建窗口并显示图像，窗口名称为'Image Window'，可自行修改
cv2.namedWindow('Image Window', cv2.WINDOW_NORMAL)
cv2.imshow('Image Window', image1)

# 判断图像通道数，如果大于1则转为灰度图
if len(image1.shape) > 2:
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
if len(image2.shape) > 2:
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# 图像数据类型转换，（归一化到0-1范围）
image1 = image1.astype(np.float64) / 255.0
image2 = image2.astype(np.float64) / 255.0

lambda_ = 0.8
print('latlrr')
import time
start_time = time.time()

# 调用latent_lrr函数进行处理
X1 = image1
Z1, L1, E1 = latent_lrr(X1, lambda_)
X2 = image2
Z2, L2, E2 = latent_lrr(X2, lambda_)

end_time = time.time()
print(f"latlrr处理耗时: {end_time - start_time} 秒")
print('latlrr')

# 计算相关图像结果
I_lrr1 = X1 @ Z1
I_saliency1 = L1 @ X1
I_lrr1 = np.clip(I_lrr1, 0, 1)
I_saliency1 = np.clip(I_saliency1, 0, 1)
I_e1 = E1

I_lrr2 = X2 @ Z2
I_saliency2 = L2 @ X2
I_lrr2 = np.clip(I_lrr2, 0, 1)
I_saliency2 = np.clip(I_saliency2, 0, 1)
I_e2 = E2

# lrr部分
F_lrr = (I_lrr1 + I_lrr2) / 2
# saliency部分
F_saliency = I_saliency1 + I_saliency2

F = F_lrr + F_saliency

# 展示图像（使用matplotlib展示，这里会弹出多个图像窗口显示）
plt.figure()
plt.imshow(I_saliency1, cmap='gray')
plt.title('I_saliency1')
plt.axis('off')

plt.figure()
plt.imshow(I_saliency2, cmap='gray')
plt.title('I_saliency2')
plt.axis('off')

plt.figure()
plt.imshow(F, cmap='gray')
plt.title('F')
plt.axis('off')

# 保存融合后的图像（使用cv2保存为png格式）
cv2.imwrite(fuse_path, (F * 255).astype(np.uint8))