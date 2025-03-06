"""
定义一个类似卷积核的剪刀核,大小为5*5,用于计算核上的对称差
将所有中心对称的两个元素的差的绝对值求平均,注入核心表示梯度信息
这个核也可以被称为“对称差算子”
输入:(H, W, C)形状的三维数组
输出:(H, W, C)形状的三维数组
"""

import cupy as cp

# 输入arr的形状为(H, W, C),数据类型float16
# weights是5*5*3的权重矩阵,如果不输入则使用默认权重
def scissors_core(arr, weights=None):

    H, W, C = arr.shape

    # 定义默认权重
    if weights == None:
        weights_2d = cp.array(
            [[0.025,0.035,0.035,0.035,0.025],
             [0.035,0.050,0.070,0.050,0.035],
             [0.035,0.070,0.000,0.070,0.035],
             [0.035,0.050,0.070,0.050,0.035],
             [0.025,0.035,0.035,0.035,0.025]],
            dtype=cp.float16)
        # 对二维矩阵升维,形状5*5*C,正常来说是5*5*3
        weights = cp.repeat(weights_2d[:, :, cp.newaxis], C, axis=2)

    # 对输入的三维数组的H,W两个维度进行扩展,边缘采用复制
    padded = cp.pad(arr, pad_width=((2, 2), (2, 2), (0, 0)), mode='edge')
    
    # 为每个要计算的元素创建窗口视图(view),形状为(5,5,3),windows形状为(H,W,5,5,3)
    windows = cp.lib.stride_tricks.sliding_window_view(padded, (5,5,C)).squeeze(axis=2)

    # 生成对称窗口(双轴翻转,中心对称)
    flipped_windows = windows[..., ::-1, ::-1, :]

    # 计算绝对差值
    abs_diff = cp.abs(windows - flipped_windows)
    
    # (H,W,5,5,3)*(5,5,3)的矩阵,CuPy会进行广播,变为(H,W,5,5,3)*(1,1,5,5,3)执行逐元素乘法
    weighted_diff = cp.sum(abs_diff * weights, axis=(2,3)) 

    print(H, W, C)
    print("padded.shape:", padded.shape)
    print("windows.shape:", windows.shape)
    print("abs_diff.shape:", abs_diff.shape)  # 应为 (H,W,5,5,3)
    print("weights.shape:", weights.shape)    # 应为 (5,5,3)
    print("weighted_diff.shape:", weighted_diff.shape)

    return weighted_diff  # 形状 (H,W,3)


