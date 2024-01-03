'''
参考《ManualKDE: 交互式选点 x 核密度估计 x 热力图可视化 x 参数可调》
url: 
'''

import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import cv2
import tkinter as tk
from tkinter import filedialog
import os

import platform

density_matrix = None

def is_linux():
    if 'linux' in platform.platform().lower():
        return True
    return False

def ManualControlKDE(bw=40, default_directory='./', outshape=1.0):

    def toggle_neighbors(matrix, y, x):
        # 切换当前点的状态
        matrix[y, x] = 1 - matrix[y, x]
        # 切换四个相邻点的状态
        neighbors = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
        for neighbor_y, neighbor_x in neighbors:
            if 0 <= neighbor_y < matrix.shape[0] and 0 <= neighbor_x < matrix.shape[1]:
                matrix[neighbor_y, neighbor_x] = 1 - matrix[neighbor_y, neighbor_x]

    def close_event(event):
        if event.key == 'escape':
            # 关闭坐标轴
            plt.axis('off')
            # 设置空标题
            plt.title('')

            if not os.path.exists('./results/'):
                os.mkdir('./results/')

            base = os.path.basename(file_path).split('.')[0]
            plt.savefig('./results/{}_heat.png'.format(base), bbox_inches='tight')
            plt.close(event.canvas.figure)

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            x = int(round(event.xdata))
            y = int(round(event.ydata))

            # 切换当前点及其相邻点的状态
            toggle_neighbors(matrix, y, x)

            # Calculate kernel density
            density_matrix = calculate_kernel_density(matrix, bw)

            # Update the image
            update_image(img, density_matrix)

    def calculate_kernel_density(matrix, bw):
        data_points = np.argwhere(matrix == 1)
        kde = KernelDensity(bandwidth=bw, kernel='gaussian')
        kde.fit(data_points)
        x, y = np.meshgrid(np.linspace(0, matrix.shape[1] - 1, matrix.shape[1]), np.linspace(0, matrix.shape[0] - 1, matrix.shape[0]))
        grid_points = np.c_[y.ravel(), x.ravel()]
        log_density = kde.score_samples(grid_points)
        density_matrix = np.exp(log_density).reshape(x.shape)
        return density_matrix

    def update_image(img, density_matrix):
        plt.clf()
        plt.imshow(img)
        plt.imshow(density_matrix, cmap='jet', alpha=0.3, extent=(0, img.shape[1], img.shape[0], 0))
        #cmap可选 jet, viridis, coolwarm
        plt.title("Click on points")
        plt.draw()
    
    resume = True
    while resume:
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口

        # 打开文件选择对话框
        if is_linux():
            file_path = filedialog.askopenfilename(title="Select Image.", initialdir=default_directory, filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("JPEG files", "*.jpeg"), ("TIFF files", "*.tif")])
        else:
            file_path = filedialog.askopenfilename(title="选择文件", initialdir=default_directory, filetypes=[("图像文件", "*.png;*.jpg;*.jpeg")])
        
        if file_path:
            print("选择的文件路径:", file_path)
            resume = False
        else:
            print("未选择任何文件")
            resume = False

        root.destroy()

        image_path = file_path
        img = cv2.imread(image_path)
        img = cv2.cvtColor(cv2.resize(img, (int(outshape*img.shape[1]), int(outshape*img.shape[0]))), cv2.COLOR_BGR2RGB)
        print("处理后图像大小为:",img.shape)
        # 初始化矩阵
        matrix = np.zeros((img.shape[0], img.shape[1]), dtype=int)

        # 通过交互式绘图选择点
        # 完成选择后使用ESC键退出并保存
        plt.imshow(img)
        plt.title("Click on points (1 to select, 0 to deselect)")
        cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
        cid = plt.gcf().canvas.mpl_connect('key_press_event', close_event)
        plt.show()


default_directory = 'xxxxxx'  # 指定包含需要处理的图像的目录()
bandwidth = 40  # 影响热图效果
outshape = 1  # 输出大小比例，是否对图像进行resize
ManualControlKDE(bandwidth, default_directory, outshape)


