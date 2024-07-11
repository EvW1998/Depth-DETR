import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TransformerEncoder, TransformerEncoderLayer


class DepthPredictor(nn.Module):

    def __init__(self, model_cfg):
        """
        Initialize depth predictor and depth encoder
        Args:
            model_cfg [EasyDict]: Depth classification network config
        """
        super().__init__()
        depth_num_bins = int(model_cfg["num_depth_bins"])  # 80
        depth_min = float(model_cfg["depth_min"])  # 0.001
        depth_max = float(model_cfg["depth_max"])  # 60
        self.depth_max = depth_max  # 60

        bin_size = 2 * (depth_max - depth_min) / (depth_num_bins * (1 + depth_num_bins))  # 0.0185
        bin_indice = torch.linspace(0, depth_num_bins - 1, depth_num_bins)  # [0, 1, .... , 79]
        bin_value = (bin_indice + 0.5).pow(2) * bin_size / 2 - bin_size / 8 + depth_min
        bin_value = torch.cat([bin_value, torch.tensor([depth_max])], dim=0)
        self.depth_bin_values = nn.Parameter(bin_value, requires_grad=False)

        # Create modules
        d_model = model_cfg["hidden_dim"]  # 256
        self.downsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GroupNorm(32, d_model))
        self.proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
            nn.GroupNorm(32, d_model))
        self.upsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
            nn.GroupNorm(32, d_model))

        self.depth_head = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU())

        self.depth_classifier = nn.Conv2d(d_model, depth_num_bins + 1, kernel_size=(1, 1))

        depth_encoder_layer = TransformerEncoderLayer(d_model, nhead=8, dim_feedforward=256, dropout=0.1)

        self.depth_encoder = TransformerEncoder(depth_encoder_layer, 1)

        self.depth_pos_embed = nn.Embedding(int(self.depth_max) + 1, 256)  # [61, 256]

    def forward(self, feature, mask, pos):
        """

        :param feature: list len of 4, [[B, 256, 48, 160], [B, 256, 24, 80], [B, 256, 12, 40], [B, 256, 6, 20]]
        :param mask: mask with all False for level 2, [B, 24, 80]
        :param pos: PE of level 2, [B, 256, 24, 80]
        :return:
        """
       
        assert len(feature) == 4

        # foreground depth map
        # ipdb.set_trace()
        src_16 = self.proj(feature[1])  # [B, 256, 24, 80]
        src_32 = self.upsample(F.interpolate(feature[2], size=src_16.shape[-2:], mode='bilinear'))
        # [B, 256, 12, 40] -> [B, 256, 24, 80]
        src_8 = self.downsample(feature[0])  # [B, 256, 48, 160] -> [B, 256, 24, 80]

        # new_add
        # src_8 = self.proj(feature[0])
        # src_16 = self.upsample(F.interpolate(feature[1], size=src_8.shape[-2:], mode='bilinear'))
        # src_32 = self.upsample(F.interpolate(feature[2], size=src_8.shape[-2:], mode='bilinear'))
        ####
        src = (src_8 + src_16 + src_32) / 3

        src = self.depth_head(src)  # [B, 256, 24, 80] -> [B, 256, 24, 80] depth features
        # ipdb.set_trace()
        depth_logits = self.depth_classifier(src)  # [B, 256, 24, 80] -> [B, 81, 24, 80]

        depth_probs = F.softmax(depth_logits, dim=1)  # [B, 81, 24, 80] 每个像素点在每个深度区间内的概率

        # self.depth_bin_values.reshape(1, -1, 1, 1) -> [1, 81, 1, 1]
        weighted_depth = (depth_probs * self.depth_bin_values.reshape(1, -1, 1, 1)).sum(dim=1)  # [B, 24, 80] 预测深度
        print('weighted_depth: ', weighted_depth.shape)
        print(weighted_depth[0])

        # ipdb.set_trace()
        # depth embeddings with depth positional encodings
        B, C, H, W = src.shape  # B, 256, 24, 80
        src = src.flatten(2).permute(2, 0, 1)  # [B, 256, 24, 80] -> [B, 256, 1920] -> [1920, B, 256]  H与W展开
        mask = mask.flatten(1)  # [B, 24, 80] -> [B, 1920]
        pos = pos.flatten(2).permute(2, 0, 1)  # [B, 256, 24, 80] -> [B, 256, 1920] -> [1920, B, 256]

        depth_embed = self.depth_encoder(src, mask, pos)  # [1920, B, 256]
        depth_embed = depth_embed.permute(1, 2, 0).reshape(B, C, H, W)
        
        # [1920, B, 256] -> [B, 256, 1920] -> [B, 256, 24, 80]

        # ipdb.set_trace()
        depth_pos_embed_ip = self.interpolate_depth_embed(weighted_depth)  # [B, 24, 80] -> [B, 256, 24, 80] pe
        depth_embed = depth_embed + depth_pos_embed_ip

        """
        depth_logits: [B, 81, 24, 80] 深度分类结果
        depth_embed: [B, 256, 24, 80] 加入深度编码的经过encoder后的特征
        weighted_depth: [B, 24, 80] 预测深度
        depth_pos_embed_ip: [B, 256, 24, 80] 深度embedding
        """
        return depth_logits, depth_embed, weighted_depth, depth_pos_embed_ip

    def interpolate_depth_embed(self, depth):
        depth = depth.clamp(min=0, max=self.depth_max)  # [B, 24, 80] -> [B, 24, 80]
        pos = self.interpolate_1d(depth, self.depth_pos_embed)  # [B, 24, 80, 256]
        pos = pos.permute(0, 3, 1, 2)  # [B, 256, 24, 80]
        return pos

    def interpolate_1d(self, coord, embed):
        floor_coord = coord.floor()
        delta = (coord - floor_coord).unsqueeze(-1)
        floor_coord = floor_coord.long()
        ceil_coord = (floor_coord + 1).clamp(max=embed.num_embeddings - 1)
        return embed(floor_coord) * (1 - delta) + embed(ceil_coord) * delta
