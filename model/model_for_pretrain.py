import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from transformers import SegformerForImageClassification

def resize(input, target_size=(352, 352)):
    return F.interpolate(input, (target_size[0], target_size[1]), mode='bilinear', align_corners=True)

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
            if m.weight is None:
                pass
            elif m.bias is not None:
                nn.init.zeros_(m.bias)
            else:
                nn.init.ones_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.ReLU6, nn.Upsample, Parameter, nn.AdaptiveAvgPool2d, nn.Sigmoid)):
            pass
        else:
            try:
                m.initialize()
            except:
                pass


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def initialize(self):
        weight_init(self)


class Fusion(nn.Module):
    def __init__(self, channel):
        super(Fusion, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv_upsample = BasicConv2d(channel, channel, 3, padding=1)
        self.cat_conv = BasicConv2d(channel*2, channel, 3, padding=1)

    def forward(self, x_low, x_high):
        x_mul = x_low * x_high
        x_cat = torch.cat((x_low, x_mul), dim=1)
        x_cat = self.cat_conv(x_cat)
        x_cat = self.relu(x_cat)
        
        return x_cat

    def initialize(self):
        weight_init(self)


class Multi_Scale_Feature_Fusion_Module(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Multi_Scale_Feature_Fusion_Module, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = BasicConv2d(in_channel, out_channel, 1)
        self.branch1 = BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=6, dilation=6)
        self.branch2 = BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=12, dilation=12)
        self.branch3 = BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=18, dilation=18)

        self.in_conv = nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)

        sq_x = self.in_conv(x)
        x1 = self.branch1(sq_x)
        x2 = self.branch2(sq_x + x1)
        x3 = self.branch3(sq_x + x2)

        x = self.relu(x0 + x3)

        return x

    def initialize(self):
        weight_init(self)


class GSANet(torch.nn.Module):
    def __init__(self):
        super(GSANet, self).__init__()
        self.rgb_encoder = SegformerForImageClassification.from_pretrained("nvidia/mit-b2")

        self.rgb_MFFM1 = Multi_Scale_Feature_Fusion_Module(512, 128)
        self.rgb_MFFM2 = Multi_Scale_Feature_Fusion_Module(320, 128)
        self.rgb_MFFM3 = Multi_Scale_Feature_Fusion_Module(128, 128)
        self.rgb_MFFM4 = Multi_Scale_Feature_Fusion_Module(64, 128)
        
        self.fusion1 = Fusion(128)
        self.fusion2 = Fusion(128)
        self.fusion3 = Fusion(128)

        self.linearr5 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.linearr6 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.linearr7 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.linearr8 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

        self.initialize()

    def forward(self, rgb):
        shape = rgb.size()[2:]
        rgb_features = self.rgb_encoder(rgb, output_hidden_states=True).hidden_states
        x4_r, x3_r, x2_r, x1_r = rgb_features[3], rgb_features[2], rgb_features[1], rgb_features[0]
        x1_r, x2_r, x3_r, x4_r = self.rgb_MFFM4(x1_r), self.rgb_MFFM3(x2_r), self.rgb_MFFM2(x3_r), self.rgb_MFFM1(x4_r)

        x1 = x1_r
        x2 = F.interpolate(x2_r, scale_factor=2, mode='bilinear')
        x3 = F.interpolate(x3_r, scale_factor=4, mode='bilinear')
        x4 = F.interpolate(x4_r, scale_factor=8, mode='bilinear')

        SA_4 = x4
        SA_3 = self.fusion3(x3, SA_4)
        SA_2 = self.fusion2(x2, SA_3)
        SA_1 = self.fusion1(x1, SA_2)

        map_24 = self.linearr8(SA_4)
        map_23 = self.linearr7(SA_3) + map_24
        map_22 = self.linearr6(SA_2) + map_23
        map_21 = self.linearr5(SA_1) + map_22

        out_1 = torch.sigmoid(resize(map_21, shape))
        out_2 = torch.sigmoid(resize(map_22, shape))
        out_3 = torch.sigmoid(resize(map_23, shape))
        out_4 = torch.sigmoid(resize(map_24, shape))

        preds_list = [out_1, out_2, out_3, out_4]

        return preds_list, None

    def initialize(self):
        weight_init(self)
