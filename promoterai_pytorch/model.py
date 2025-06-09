import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import PromoterAIConfig
from transformers import PreTrainedModel

class MetaFormerBlock(nn.Module):
    def __init__(self, model_dim, kernel_size, dilation_rate):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(model_dim)
        self.depthwise = nn.Conv1d(
            in_channels=model_dim,
            out_channels=model_dim,
            kernel_size=kernel_size,
            dilation=dilation_rate,
            padding=(kernel_size // 2) * dilation_rate,
            groups=model_dim  # depthwise
        )

        self.norm2 = nn.BatchNorm1d(model_dim)
        self.fc1 = nn.Linear(model_dim, model_dim * 4)
        self.fc2 = nn.Linear(model_dim * 4, model_dim)

    def forward(self, x):
        # x: [B, L, C] → permute to [B, C, L] for Conv1D
        x = x.permute(0, 2, 1)
        residual = x
        out = self.norm1(x)
        out = self.depthwise(out)
        out = residual + out

        out = self.norm2(out)
        out = out.permute(0, 2, 1)  # back to [B, L, C]
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        out = out + out.permute(0, 2, 1).permute(0, 2, 1)  # residual again
        return out
    

class PromoterAI(PreTrainedModel):
    config_class = PromoterAIConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.input_proj = nn.Conv1d(4, config.model_dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            MetaFormerBlock(config.model_dim, config.kernel_size, config.dilation_rate(i))
            for i in range(config.num_blocks)
        ])

        self.output_heads = nn.ModuleList()
        for out_dim in config.output_dims:
            self.output_heads.append(
                nn.ModuleList([
                    nn.Linear(config.model_dim, out_dim)
                    for _ in range(config.num_blocks, 0, -config.shortcut_layer_freq)
                ])
            )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        layer_outputs = [self.input_proj(x).permute(0, 2, 1)]
        for block in self.blocks:
            layer_outputs.append(block(layer_outputs[-1]))

        outputs = []
        for j, head_layers in enumerate(self.output_heads):
            selected_layers = layer_outputs[-1::-self.config.shortcut_layer_freq][1:]
            pooled = torch.stack([
                F.relu(head(l))
                for head, l in zip(head_layers, selected_layers)
            ])
            avg = torch.mean(pooled, dim=0)

            if self.config.output_crop > 0:
                crop = self.config.output_crop // 2
                avg = avg[:, crop:-crop, :]
            outputs.append(avg)
        return outputs if len(outputs) > 1 else outputs[0]


class TwinWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

        # Freeze everything except layers with "output0" (if desired)
        for name, param in self.base.named_parameters():
            param.requires_grad = 'output0' in name

    def forward(self, ref, alt):
        out_ref = self.base(ref)
        out_alt = self.base(alt)
        if isinstance(out_ref, list):
            out_ref = out_ref[0]
            out_alt = out_alt[0]

        # [B, L, D] → difference → mean over L and D → [B]
        diff = out_alt - out_ref
        scalar = diff.mean(dim=(1, 2))
        return scalar
