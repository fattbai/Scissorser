import numpy as np
from concurrent.futures import ThreadPoolExecutor

class Scissorser:
    def __init__(self, thick=8):
        self.default_thick = thick
        self.use_cuda = self._check_cuda()
        self.xp = self._init_array_module()

    def _check_cuda(self):
        try:
            import cupy
            return cupy.cuda.runtime.getDeviceCount() > 0
        except:
            return False

    def _init_array_module(self):
        if self.use_cuda:
            import cupy
            return cupy
        return np

    def _vectorized_diff(self, channel_data, thick):
        """向量化计算窗口均值差异"""
        xp = self.xp
        is_1d = channel_data.ndim == 1
        
        # ========== 独立的一维处理逻辑 ==========
        if is_1d:
            frames = len(channel_data)
            cumsum = xp.concatenate([xp.zeros(1), xp.cumsum(channel_data)])
            
            # 一维专用索引计算（避免多维广播）
            indices = xp.arange(frames)
            right_starts = xp.maximum(0, indices - thick)
            left_ends = xp.minimum(frames, indices + thick + 1)
            
            # 一维优化计算
            right_sum = cumsum[indices] - cumsum[right_starts]
            left_sum = cumsum[left_ends] - cumsum[indices]
            right_counts = indices - right_starts
            left_counts = left_ends - indices
            
            return xp.abs(
                xp.divide(right_sum, right_counts, where=right_counts!=0, out=xp.zeros_like(right_sum)) - 
                xp.divide(left_sum, left_counts, where=left_counts!=0, out=xp.zeros_like(left_sum))
            )

        else:
            frames = channel_data.shape[-1]
            # 计算累积和
            cumsum = xp.concatenate([xp.zeros(1), xp.cumsum(channel_data)])
            
            # 生成索引
            indices = xp.arange(frames)
            right_starts = xp.maximum(0, indices - thick)
            left_ends = xp.minimum(frames, indices + thick + 1)

            # 计算窗口统计量
            right_sum = cumsum[indices] - cumsum[right_starts]
            left_sum = cumsum[left_ends] - cumsum[indices]
            right_counts = indices - right_starts
            left_counts = left_ends - indices

            # 避免除零错误
            right_mean = xp.divide(right_sum, right_counts, where=right_counts!=0, out=xp.zeros_like(right_sum))
            left_mean = xp.divide(left_sum, left_counts, where=left_counts!=0, out=xp.zeros_like(left_sum))
            
            return xp.abs(right_mean - left_mean)

    def process(self, data, thick=None):
        thick = self.default_thick if thick is None else thick
        data = self.xp.asarray(data)
            # 维度分支处理
        if data.ndim == 1:
            # 直接调用一维处理，不改变维度
            processed = self._vectorized_diff(data, thick)
            return processed.get() if self.use_cuda else processed

        else:
            original_shape = data.shape
            n_channels = original_shape[0]
            result = self.xp.zeros_like(data)

        # 并行处理通道
        def process_channel(c):
            channel_data = data[c]
            if channel_data.ndim > 1:  # 处理多维数据
                spatial_shape = channel_data.shape[:-1]
                channel_flat = channel_data.reshape(-1, original_shape[-1])
                diff_flat = self.xp.array([self._vectorized_dim(signal, thick) for signal in channel_flat])
                return c, diff_flat.reshape(spatial_shape + (-1,))
            return c, self._vectorized_diff(channel_data, thick)

        # GPU模式使用线程池，CPU模式使用进程池
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_channel, c) for c in range(n_channels)]
            for future in futures:
                c, channel_result = future.result()
                result[c] = channel_result

        return result.get() if self.use_cuda else result

    def compress(self, processed_data):
        return np.mean(processed_data, axis=-1).squeeze()