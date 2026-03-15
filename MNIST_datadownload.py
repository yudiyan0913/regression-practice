import torch
from torchvision import datasets
import matplotlib.pyplot as plt
import os

# 加载 MNIST 数据集（不会重新下载，会直接用你电脑里的）
dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=False  # 👈 设为 False，就不会再下载
)

# 创建一个文件夹来保存图片
save_dir = './mnist_images'
os.makedirs(save_dir, exist_ok=True)

# 把前 1000 张图保存为 png 文件
for i in range(1000):
    image, label = dataset[i]
    if label!=3:
        continue
    # 转换成 numpy 数组并显示
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {label}')
    plt.axis('off')
    
    # 保存为图片文件
    plt.savefig(f'{save_dir}/digit_{i}_label_{label}.png')
    plt.close()

print(f"图片已保存到 {save_dir} 文件夹")