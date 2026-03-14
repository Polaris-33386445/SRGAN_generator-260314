import torch
from torch import nn
import torchvision
import math

# Generator, Discriminator, TruncatedVGG19
class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        # 生成器超参数
        scaling_factor = config.scaling_factor
        large_kernel_size = config.G.large_kernel_size
        small_kernel_size = config.G.small_kernel_size
        # 根据卷积核大小和步幅计算padding，使输出特征图的空间尺寸恰好等于(W/stride,H/stride)
        # math.floor((H + 2 * padding - kernel_size) / stride) + 1 = H / stride
        # 当输入H/W为偶数，stride=1或2，kernel_size为奇数时，padding = (kernel_size - 1) // 2
        large_padding = (large_kernel_size - 1) // 2
        small_padding = (small_kernel_size - 1) // 2
        n_channels = config.G.n_channels
        n_blocks = config.G.n_blocks
        # 生成器网络结构
        # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, n_channels, large_kernel_size, stride = 1, padding = large_padding), # k9n64s1
            nn.PReLU()
        )
        # 独立残差模块列表：每个block独立，避免共享参数
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
            nn.Conv2d(n_channels, n_channels, small_kernel_size, stride = 1, padding = small_padding), # k3n64s1
            nn.BatchNorm2d(n_channels),
            nn.PReLU(),
            nn.Conv2d(n_channels, n_channels, small_kernel_size, stride = 1, padding = small_padding), # k3n64s1
            nn.BatchNorm2d(n_channels)
            ) for _ in range(n_blocks)])
        self.conv2 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, small_kernel_size, stride = 1, padding = small_padding), # k3n64s1
            nn.BatchNorm2d(n_channels)
        )
        # 独立上采样模块列表：每个block独立，避免共享参数
        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
            nn.Conv2d(n_channels, n_channels * 4, small_kernel_size, stride = 1, padding = small_padding), # k3n256s1
            nn.PixelShuffle(2), # (n_channels * 4, W, H) -> (n_channels, W * 2, H * 2)
            nn.PReLU()
            ) for _ in range(int(math.log2(scaling_factor)))])
        self.conv3 = nn.Sequential(
            nn.Conv2d(n_channels, 3, large_kernel_size, stride = 1, padding = large_padding), # k9n3s1
            nn.Tanh() # SR图像的像素值映射到[-1, 1]范围内，和HR一致
        )

    def forward(self, lr_img):
        x = self.conv1(lr_img)
        x_1 = x
        for blk in self.res_blocks:
            x = x + blk(x) # element-wise sum,resnet connection,inplace=False
        x = self.conv2(x)
        x = x + x_1 # element-wise sum,skip connection,inplace=False
        for blk in self.upsample_blocks:
            x = blk(x)
        sr_img = self.conv3(x)
        return sr_img
    
class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        # 判别器超参数
        crop_size = config.crop_size
        kernel_size = config.D.kernel_size
        padding = (kernel_size - 1) // 2
        n_channels = config.D.n_channels
        n_blocks = config.D.n_blocks
        fc_size = config.D.fc_size
        # 判别器网络结构
        conv1 = nn.Sequential( # 第一层卷积
            nn.Conv2d(3, n_channels, kernel_size, stride = 1, padding = padding), # k3n64s1
            nn.LeakyReLU(0.2)
        )
        self.conv_blocks = nn.ModuleList() # 卷积块列表
        self.conv_blocks.append(conv1)
        for i in range(1, n_blocks):
            in_channels = n_channels * 2 ** ((i - 1) // 2)
            out_channels = n_channels * 2 ** (i // 2)
            stride = 1 if i % 2 == 0 else 2
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding = padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            )
            self.conv_blocks.append(conv_block)
        # 输出端网络：dense(1024)->leakyrelu(0.2)->dense(1)->sigmoid
        # flatten_size = (n_channels * 2 ** ((n_blocks - 1) // 2)) * (crop_size // (2 ** (n_blocks // 2))) ** 2
        deep_n_channels = n_channels * 2 ** ((n_blocks - 1) // 2) # 最后一层卷积块的输出通道数
        deep_crop_size = crop_size // (2 ** (n_blocks // 2)) # 最后一层卷积块输出特征图的空间尺寸
        flatten_size = deep_n_channels * deep_crop_size * deep_crop_size # 最后一层卷积块输出特征图展平后的维度
        self.fc_block = nn.Sequential(
            nn.Linear(in_features = flatten_size, out_features = fc_size),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features = fc_size, out_features = 1),
            nn.Sigmoid() # 输出二分类预测概率，表示图片被判定为HR的置信度
        )

    def forward(self, img):
        x = img
        for blk in self.conv_blocks:
            x = blk(x)
        x = x.view(x.size(0), -1) # 将张量x展平为(batch_size, C * W * H)，适应全连接模块的输入格式
        x = self.fc_block(x)
        return x
    
class TruncatedVGG19(nn.Module):
    def __init__(self, i, j):
        super(TruncatedVGG19, self).__init__()
        # 加载预训练的VGG19模型
        vgg19 = torchvision.models.vgg19(pretrained = True)
        # 采纳VGG19的特征提取器features，放弃分类器classifier
        # 对比第i层池化前，第j层卷积后的特征图
        conv_layers = 0 # 上一个池化层后的卷积层数
        pool_layers = 0 # 池化层数
        truncated_layers = [] # 已采纳的网络层
        target_conv_reached = False # 是否已达到目标卷积层
        for layer in vgg19.features.children():
            truncated_layers.append(layer)
            if isinstance(layer, nn.Conv2d):
                conv_layers += 1
            elif isinstance(layer, nn.MaxPool2d):
                pool_layers += 1
                conv_layers = 0
            if pool_layers == i - 1 and conv_layers == j:
                target_conv_reached = True
            if target_conv_reached and isinstance(layer, nn.ReLU):
                break # 截取到目标卷积层后的激活函数
        # 检查截取是否正确
        if pool_layers != i - 1 or conv_layers != j:
            raise ValueError(f"Invalid i={i} or j={j} for VGG19. Reached pool_layers={pool_layers}, conv_layers={conv_layers}.")
        self.truncated_vgg19 = nn.Sequential(*truncated_layers)
        # 冻结网络参数，VGG19仅作为特征提取器，不参与训练
        for param in self.truncated_vgg19.parameters():
            param.requires_grad = False

    def forward(self, img):
        return self.truncated_vgg19(img) # 输出fai(i,j)(img)
