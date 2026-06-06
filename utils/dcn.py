import torch
import torch.nn as nn
import torchvision.ops as ops

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()
        
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        
        # 1. 偏移量生成器 (Offset Generator)
        # 输出通道为 2 * kernel_size * kernel_size (对应 x, y 偏移)
        # 如果是 Modulated DCN (v2)，还需要一个 mask，通道数为 3 * k * k
        # 这里我们实现标准的 Modulated DCN v2
        self.offset_conv = nn.Conv2d(in_channels, 
                                     3 * kernel_size * kernel_size, 
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=padding)
        
        # 初始化偏移量生成器权重为 0，保证初始状态等同于普通卷积
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        
        # 2. 实际的卷积权重
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=1)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        # 生成 offset 和 mask
        out = self.offset_conv(x)
        
        # 拆分 offset 和 mask
        # offset: [B, 2*k*k, H, W], mask: [B, k*k, H, W]
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask) # 调制标量限制在 [0, 1]
        
        # 调用 torchvision 官方 CUDA 算子
        return ops.deform_conv2d(input=x, 
                                 offset=offset, 
                                 weight=self.weight, 
                                 bias=self.bias, 
                                 stride=self.stride, 
                                 padding=self.padding, 
                                 mask=mask)