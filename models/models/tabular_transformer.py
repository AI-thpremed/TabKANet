from typing import Dict, List


import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from models.models.base_blocks import Activation, ColumnEmbedding

class CatEncoder(nn.Module):
    def __init__(self, 
                 vocabulary: Dict[str, Dict[str, int]],
                 embedding_dim: int, nhead: int, num_layers: int, dim_feedforward: int, 
                 dropout_rate: float):
        """
        Categorical feature encoder.

        Parameters:
        - vocabulary (Dict[str, Dict[str, int]]): Vocabulary of categorical features
        - embedding_dim (int): Embedding dimension.
        - nhead (int): Number of attention heads.
        - num_layers (int): Number of transformer layers.
        - dim_feedforward (int): Dimension of the feedforward network model.
        - dropout_rate (float): Dropout rate.
        """
        super(CatEncoder, self).__init__()
        self.vocabulary = vocabulary
        self.model = nn.ModuleDict({
            'column_embedding_layer': ColumnEmbedding(vocabulary, embedding_dim),
            'transformer_encoder': nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=embedding_dim, 
                    nhead=nhead,
                    dim_feedforward=dim_feedforward, 
                    dropout=dropout_rate,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True),
                    num_layers=num_layers,
                    norm=nn.LayerNorm([embedding_dim]))})

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        x = [self.model['column_embedding_layer'](x[:, i], col) for i, col in enumerate(self.vocabulary)]
        x = torch.stack(x, dim=1)
        x = self.model['transformer_encoder'](x).view(batch_size, -1)
        # x = self.model['transformer_encoder'](x)
        return x

class NumEncoder(nn.Module):
    def __init__(self, num_features: int):
        """
        Continuous feature encoder.

        Parameters:
        - num_features (int): Number of continuous features.
        """
        super(NumEncoder, self).__init__()
        self.norm = nn.LayerNorm([num_features])
        
    def forward(self, x: torch.Tensor):
        return self.norm(x)


# class NumEncoderTransformer(nn.Module):
#     def __init__(self, num_continuous_features: int, embedding_dim: int, nhead: int, num_layers: int, dim_feedforward: int, dropout_rate: float):
#         """
#         Continuous feature encoder using Transformer architecture.

#         Parameters:
#         num_continuous_features, embedding_dim, nhead, num_layers, dim_feedforward, attn_dropout_rate
#         - num_features (int): Number of continuous features.
#         - embedding_dim (int): Dimension of the embedding.
#         - nhead (int): Number of attention heads.
#         - dim_feedforward (int): Dimension of the feedforward network.
#         - dropout_rate (float): Dropout rate.
#         - num_layers (int): Number of transformer layers.
#         """
#         super(NumEncoderTransformer, self).__init__()
#         self.norm = nn.LayerNorm([num_continuous_features])
#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(
#                 d_model=embedding_dim, 
#                 nhead=nhead,
#                 dim_feedforward=dim_feedforward, 
#                 dropout=dropout_rate,
#                 activation='gelu',
#                 batch_first=True,
#                 norm_first=True),
#             num_layers=num_layers,
#             norm=nn.LayerNorm([embedding_dim]))

#     def forward(self, x: torch.Tensor):
#         batch_size = x.size(0)

#         x=self.norm(x)
#         x = self.transformer(x).view(batch_size, -1)
#         return x




class NumEncoderTransformer(nn.Module):
    def __init__(self, num_continuous_features: int, embedding_dim: int, nhead: int, num_layers: int, dim_feedforward: int, dropout_rate: float):

        super(NumEncoderTransformer, self).__init__()
        self.norm = nn.LayerNorm([num_continuous_features])
        self.kanclassifier=KAN([num_continuous_features,64,embedding_dim * num_continuous_features])
    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)

        x=self.norm(x)
        x = self.kanclassifier(x)
        return x


class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, 
                 activation: str, dropout_rate: float):
        """
        MLP block.

        Parameters:
        - input_dim (int): Input dimension.
        - output_dim (int): Output dimension.
        - activation (str): Activation function.
        - dropout_rate (float): Dropout rate.
        """
        super(MLPBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm([output_dim]),
            Activation(activation),
            nn.Dropout(dropout_rate))

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, 
                 hidden_dims: List[int], activation: str, 
                 dropout_rate: float):
        """
        MLP model.

        Parameters:
        - input_dim (int): Input dimension.
        - output_dim (int): Output dimension.
        - hidden_dims (List[int]): List of hidden layer dimensions.
        - activation (str): Activation function.
        - dropout_rate (float): Dropout rate.
        """
        super(MLP, self).__init__()
        dims = [input_dim] + hidden_dims
        self.model = nn.Sequential(*(
            [
                MLPBlock(
                    dims[i], dims[i + 1], 
                    activation, dropout_rate) 
                for i in range(len(dims) - 1)] + \
                [nn.Linear(dims[-1], output_dim)]))
        
    def forward(self, x):
        return self.model(x)

class TabularTransformer(nn.Module):
    def __init__(self, 
                 output_dim: int, vocabulary: Dict[str, Dict[str, int]], num_continuous_features: int,
                 embedding_dim: int, nhead: int, num_layers: int, dim_feedforward: int, attn_dropout_rate: float,
                 mlp_hidden_dims: List[int], activation: str, ffn_dropout_rate: float):
        """
        Tabular transformer model.

        Parameters:
        - output_dim (int): Output dimension.
        - vocabulary (Dict[str, Dict[str, int]]): Vocabulary of categorical features.
        - num_continuous_features (int): Number of continuous features.
        - embedding_dim (int): Embedding dimension.
        - nhead (int): Number of attention heads.
        - num_layers (int): Number of transformer layers.
        - dim_feedforward (int): Dimension of the feedforward network model.
        - attn_dropout_rate (float): Dropout rate.
        - mlp_hidden_dims (List[int]): List of hidden layer dimensions.
        - activation (str): Activation function.
        - ffn_dropout_rate (float): Dropout rate.
        """
        super(TabularTransformer, self).__init__()


        #method 1 对于类别特征用transformer，对于数值，直接归一化拼接
        self.encoders = nn.ModuleDict({
            'categorical_feature_encoder': CatEncoder(vocabulary, embedding_dim, nhead, num_layers, dim_feedforward, attn_dropout_rate),
            'continuous_feature_encoder': NumEncoder(num_continuous_features),
        })
        self.classifier = MLP(embedding_dim * len(vocabulary) + num_continuous_features, output_dim, mlp_hidden_dims, activation, ffn_dropout_rate)


#income
# Epoch: 0, Train Loss: 0.1542, Val Loss: 0.2357, Val Metric: 0.9394, Test Metric: 0.9357    
# Epoch: 0, Train Loss: 0.1639, Val Loss: 0.2635, Val Metric: 0.9397, Test Metric: 0.9557                                                                                                                              
#MLP Epoch: 0, Train Loss: 0.1481, Val Loss: 0.2057, Val Metric: 0.9375, Test Metric: 0.9324                                               
#改进kan                Val Metric: 0.9352, Test Metric: 0.9257                               
# Val Metric: 0.9397, Test Metric: 0.9563 , Test AUC: 0.9965 


#blastchar
# Val Metric: 0.7207, Test Metric: 0.7480
# Val Metric: 0.7385, Test Metric: 0.7505   
#mlp Val Metric: 0.7256, Test Metric: 0.7491
#改进kan Val Metric: 0.7326, Test Metric: 0.7540



# heart
# Epoch: 0, Train Loss: 0.3978, Val Loss: 0.4492, Val Metric: 0.8821, Test Metric: 0.8354
# Epoch: 13, Train Loss: 0.3782, Val Loss: 0.4000, Val Metric: 0.8892, Test Metric: 0.8776                                              
#   mlp  Val Loss: 0.4802, Val Metric: 0.8749, Test Metric: 0.8301 
# 改进 kan。统一表示  Epoch: 14, Train Loss: 0.3403, Val Loss: 0.3269, Val Metric: 0.8968, Test Metric: 0.8784                                              
  
        #method 2 数值，用kan去扩充
        # self.encoders = nn.ModuleDict({
        #     'categorical_feature_encoder': CatEncoder(vocabulary, embedding_dim, nhead, num_layers, dim_feedforward, attn_dropout_rate),
        #     'continuous_feature_encoder': NumEncoderTransformer(num_continuous_features, embedding_dim, nhead, num_layers, dim_feedforward, attn_dropout_rate),
        # })
        # self.classifier = MLP(embedding_dim * (len(vocabulary) + num_continuous_features), output_dim, mlp_hidden_dims, activation, ffn_dropout_rate)
        
        # self.kanclassifier=KAN([embedding_dim * len(vocabulary) + embedding_dim * num_continuous_features,64,output_dim])



    def forward(self, categorical_x: torch.Tensor, continuous_x: torch.Tensor):
        categorical_x = self.encoders['categorical_feature_encoder'](categorical_x)
        continuous_x = self.encoders['continuous_feature_encoder'](continuous_x)
        x = torch.cat([categorical_x, continuous_x], dim=-1)

        # x = self.kanclassifier(x)
        
        x = self.classifier(x)
        return x
    














class KAN(torch.nn.Module): # 封装了一个KAN神经网络模型，可以用于对数据进行拟合和预测。
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        """
        初始化 KAN 模型。

        参数:
            layers_hidden (list): 包含每个隐藏层输入特征数量的列表。
            grid_size (int): 网格大小，默认为 5。
            spline_order (int): 分段多项式的阶数，默认为 3。
            scale_noise (float): 缩放噪声，默认为 0.1。
            scale_base (float): 基础缩放，默认为 1.0。
            scale_spline (float): 分段多项式的缩放，默认为 1.0。
            base_activation (torch.nn.Module): 基础激活函数，默认为 SiLU。
            grid_eps (float): 网格调整参数，默认为 0.02。
            grid_range (list): 网格范围，默认为 [-1, 1]。
        """
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False): # 调用每个KANLinear层的forward方法，对输入数据进行前向传播计算输出。
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
            update_grid (bool): 是否更新网格。默认为 False。

        返回:
            torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):#计算正则化损失的方法，用于约束模型的参数，防止过拟合。
        """
        计算正则化损失。

        参数:
            regularize_activation (float): 正则化激活项的权重，默认为 1.0。
            regularize_entropy (float): 正则化熵项的权重，默认为 1.0。

        返回:
            torch.Tensor: 正则化损失。
        """
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )





class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,  # 网格大小，默认为 5
        spline_order=3, # 分段多项式的阶数，默认为 3
        scale_noise=0.1,  # 缩放噪声，默认为 0.1
        scale_base=1.0,   # 基础缩放，默认为 1.0
        scale_spline=1.0,    # 分段多项式的缩放，默认为 1.0
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,  # 基础激活函数，默认为 SiLU（Sigmoid Linear Unit）
        grid_eps=0.02,
        grid_range=[-1, 1],  # 网格范围，默认为 [-1, 1]
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size # 设置网格大小和分段多项式的阶数
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size   # 计算网格步长
        grid = ( # 生成网格
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0] 
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)  # 将网格作为缓冲区注册

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features)) # 初始化基础权重和分段多项式权重
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:  # 如果启用独立的分段多项式缩放，则初始化分段多项式缩放参数
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise # 保存缩放噪声、基础缩放、分段多项式的缩放、是否启用独立的分段多项式缩放、基础激活函数和网格范围的容差
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()  # 重置参数

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)# 使用 Kaiming 均匀初始化基础权重
        with torch.no_grad():
            noise = (# 生成缩放噪声
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_( # 计算分段多项式权重
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:  # 如果启用独立的分段多项式缩放，则使用 Kaiming 均匀初始化分段多项式缩放参数
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        """
        计算给定输入张量的 B-样条基函数。

        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
        torch.Tensor: B-样条基函数张量，形状为 (batch_size, in_features, grid_size + spline_order)。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = ( # 形状为 (in_features, grid_size + 2 * spline_order + 1)
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        """
        计算插值给定点的曲线的系数。

        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
        y (torch.Tensor): 输出张量，形状为 (batch_size, in_features, out_features)。
        返回:
        torch.Tensor: 系数张量，形状为 (out_features, in_features, grid_size + spline_order)。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        # 计算 B-样条基函数
        A = self.b_splines(x).transpose(
            0, 1 # 形状为 (in_features, batch_size, grid_size + spline_order)
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features) # 形状为 (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(   # 使用最小二乘法求解线性方程组
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)  # 形状为 (in_features, grid_size + spline_order, out_features)
        result = solution.permute( # 调整结果的维度顺序
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        """
        获取缩放后的分段多项式权重。

        返回:
        torch.Tensor: 缩放后的分段多项式权重张量，形状与 self.spline_weight 相同。
        """
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor): # 将输入数据通过模型的各个层，经过线性变换和激活函数处理，最终得到模型的输出结果
        """
        前向传播函数。

        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。

        返回:
        torch.Tensor: 输出张量，形状为 (batch_size, out_features)。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight) # 计算基础线性层的输出
        spline_output = F.linear( # 计算分段多项式线性层的输出
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output  # 返回基础线性层输出和分段多项式线性层输出的和

    @torch.no_grad()
    # 更新网格。
    # 参数:
    # x (torch.Tensor): 输入张量，形状为 (batch_size, in_features)。
    # margin (float): 网格边缘空白的大小。默认为 0.01。
    # 根据输入数据 x 的分布情况来动态更新模型的网格,使得模型能够更好地适应输入数据的分布特点，从而提高模型的表达能力和泛化能力。
    def update_grid(self, x: torch.Tensor, margin=0.01): 
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)  # 计算 B-样条基函数
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)  # 调整维度顺序为 (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)  # 调整维度顺序为 (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0] # 对每个通道单独排序以收集数据分布
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)   # 更新网格和分段多项式权重
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        # 计算正则化损失，用于约束模型的参数，防止过拟合
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        """
        计算正则化损失。

        这是对原始 L1 正则化的简单模拟，因为原始方法需要从扩展的（batch, in_features, out_features）中间张量计算绝对值和熵，
        而这个中间张量被 F.linear 函数隐藏起来，如果我们想要一个内存高效的实现。

        现在的 L1 正则化是计算分段多项式权重的平均绝对值。作者的实现也包括这一项，除了基于样本的正则化。

        参数:
        regularize_activation (float): 正则化激活项的权重，默认为 1.0。
        regularize_entropy (float): 正则化熵项的权重，默认为 1.0。

        返回:
        torch.Tensor: 正则化损失。
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

