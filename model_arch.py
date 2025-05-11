import torch
import torch.nn as nn
import timm

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super(ASPP, self).__init__()
        self.branches = nn.ModuleList()
        # 1x1 conv
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        ))
        # atrous convs
        for r in rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            ))
        # image pooling
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels*(len(self.branches)+1), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        res = []
        for b in self.branches:
            res.append(b(x))
        pooled = self.image_pool(x)
        pooled = nn.functional.interpolate(pooled, size=x.shape[2:], mode='bilinear', align_corners=False)
        res.append(pooled)
        x = torch.cat(res, dim=1)
        return self.project(x)
    
    
class DeepLabV3Plus(nn.Module):
    def __init__(self, backbone_name='convnextv2_base', num_classes=1):
        super(DeepLabV3Plus, self).__init__()
        # Load pretrained backbone
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True,
                                          out_indices=(1,2,3,4), dilations=(1,1,2,4))
        in_chs = self.backbone.feature_info.channels()[-1]
        lows_ch = self.backbone.feature_info.channels()[1]
        # ASPP
        self.aspp = ASPP(in_chs, 256, rates=[6,12,18])
        # Decoder
        self.reduce_low = nn.Sequential(
            nn.Conv2d(lows_ch, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256+48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        feats = self.backbone(x)
        low, mid, high, highest = feats
        x = self.aspp(highest)
        x = nn.functional.interpolate(x, size=low.shape[2:], mode='bilinear', align_corners=False)
        low = self.reduce_low(low)
        x = torch.cat([x, low], dim=1)
        x = self.decoder(x)
        h, w = feats[0].shape[2:]
        x = nn.functional.interpolate(x, size=(h * 4, w * 4), mode='bilinear', align_corners=False)
        return self.classifier(x)


