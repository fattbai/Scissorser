# 用于输出剪刀算子处理后的梯度灰度图
import os
import cv2
import cupy as cp
import numpy as np
from scissors_core import scissors_core

# 读取图片（返回 numpy 数组）
img_cv2 = cv2.imread("D:\\projects\\python\\Watershed\\image.jpg")
if img_cv2 is None:
    raise FileNotFoundError("图片读取失败")

# 将 numpy 数组转为 CuPy 数组
img_cupy = cp.asarray(img_cv2, dtype=cp.float16)

# print(type(img_cupy))  # 输出: <class 'cupy.ndarray'>
# print(img_cupy.shape)  # 例如: (高度, 宽度, 通道数)

H, W, C = img_cupy.shape

# 实现滤波
img_diff = cp.asnumpy(scissors_core(img_cupy))

print(type(img_diff))  
print(img_diff.shape)
print(img_diff.dtype) 

# 压缩维度为(H,W)
img_diff = cp.mean(img_diff, axis=-1, keepdims=False)

# 转换格式，用于显示灰度图
img_diff_uint8 = np.clip(img_diff, 0, 255).astype(np.uint8) 

# 显示图像
cv2.imshow("Gray Image", img_diff_uint8)
cv2.waitKey(0)
cv2.destroyAllWindows()

# print(img_diff_uint8)

# 保存图像
save_dir = "D:\\projects\\python\\Watershed\\output"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "gray_image.jpg")

success = cv2.imwrite(save_path, img_diff_uint8)
if success:
    print(f"图像已保存至：{save_path}")
else:
    print("保存失败！请检查路径或权限")