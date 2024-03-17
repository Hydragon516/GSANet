import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from fast_pytorch_kmeans import KMeans
from transformers import SegformerForImageClassification
from .feature_aggregation_transformer import Feature_Aggregation_Transformer

from einops import rearrange

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

def get_feature_clustering_masks(x, k):
    B, _, H, W = x.size()
    masks = []
    for b in range(B):
        batch_slice = x[b]
        batch_slice = rearrange(batch_slice, 'c h w -> (h w) c')

        kmeans = KMeans(n_clusters=k, mode='euclidean', max_iter=10, verbose=0)
        labels = kmeans.fit_predict(batch_slice)

        batch_masks = torch.zeros(k, H * W).to(x.device)
        for i in range(k):
            batch_masks[i, labels == i] = 1
        batch_masks = rearrange(batch_masks, 'c (h w) -> c h w', h=H, w=W).unsqueeze(0)

        masks.append(batch_masks)
    masks = torch.cat(masks, dim=0)

    return masks

def get_global_prototypes(target, mask, mode="softmax", dim=2):
    H, W = target.size(2), target.size(3)
    probs = rearrange(mask, 'b c h w -> b c (h w)')
    
    if mode == "softmax":
        ss_map = F.softmax(probs, dim=dim)
    elif mode == "sigmoid":
        ss_map = torch.sigmoid(probs)
    
    x = rearrange(target, 'b c h w -> b c (h w)')
    pb = torch.bmm(ss_map, x.transpose(1, 2))
    vis_mask = rearrange(ss_map, 'b c (h w) -> b c h w', h=H, w=W)
    return pb, vis_mask

def get_kmenas_prototypes(x, k):
    mask = get_feature_clustering_masks(x, k)
    mask = mask.clone().detach()
    probs = rearrange(mask, 'b c h w -> b c (h w)')
    x = rearrange(x, 'b c h w -> b c (h w)')
    pb = torch.bmm(probs, x.transpose(1, 2))
    return pb
    
def knn(target, ref, k):
    n_target = target / target.norm(dim=2, keepdim=True)
    n_ref = ref / ref.norm(dim=2, keepdim=True)
    corr = torch.matmul(n_target, n_ref.transpose(1, 2))
    topk = torch.topk(corr, k, dim=2)[1]
    return topk

def prototype_filter(target, ref, k=16):
    B = target.size(0)
    
    for i in range(len(ref)):
        topk = knn(target, ref[i], k)
        # frq = torch.mode(topk, dim=1)[0]

        filtered_pt = []
        for b in range(B):
            filtered_pt.append(ref[i][b, topk[b, :]])
        filtered_pt = torch.stack(filtered_pt, dim=0)
        new_ref_pt = filtered_pt.permute(0, 2, 3, 1)
    return new_ref_pt
    
def get_correlation_map(x, prototype_block):
    B, C, H, W = x.size()

    # prototype_block: (B,N,C)
    # x: (B,C,H,W)
    # corr: (B,N,H,W), -1~1
    n_p = prototype_block / prototype_block.norm(dim=2, keepdim=True)
    n_x = x.view(B, C, -1) / x.view(B, C, -1).norm(dim=1, keepdim=True)
    corr = torch.bmm(n_p, n_x).view(B, -1, H, W)
    return corr

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


class Res(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Res, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 1),
                                   nn.BatchNorm2d(in_channel), nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channel, in_channel, 3, 1, 1)
                                   )

        self.conv2 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 1),
                                   nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True)
                                   )

    def forward(self, feats):
        feats = feats + self.conv1(feats)
        feats = F.relu(feats, inplace=True)
        feats = self.conv2(feats)
        return feats

    def initialize(self):
        weight_init(self)


class Multi_Scale_Feature_Fusion_Module(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Multi_Scale_Feature_Fusion_Module, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=6, dilation=6)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=12, dilation=12)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=18, dilation=18)
        )

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


class Slot_Generator(nn.Module):
    def __init__(self, in_channels, out_channels, num_slots):
        super(Slot_Generator, self).__init__()
        self.sqz = nn.Conv2d(in_channels, out_channels, 1)
        self.slot_enc = nn.Conv2d(in_channels, num_slots, 1)
        self.FAT = Feature_Aggregation_Transformer(out_channels)
    
    def forward(self, target, ref):
        sqz_target = self.sqz(target) # B, C, H, W
        slot_target = self.slot_enc(target) # B, 2, H, W
        
        target_pt = get_kmenas_prototypes(sqz_target, 64)
        slot_pt, slot_mask = get_global_prototypes(sqz_target, slot_target, mode="softmax", dim=1) # B, 2, C

        ref_pt_list = []
        for i in range(len(ref)):
            sqz_ref = self.sqz(ref[i])
            ref_pt, _ = get_global_prototypes(sqz_ref, sqz_ref, mode="softmax", dim=2) # B, N, C
            ref_pt_list.append(ref_pt) # B, N, C
        
        ref_pt_block = torch.stack(ref_pt_list, dim=1) # b, cn, cl, c
        ref_pt_block = rearrange(ref_pt_block, 'b cn cl c -> b c cn cl')
        
        target_pt = rearrange(target_pt, 'b n c -> b c n') # B, C, N
        agg_pt = self.FAT(target_pt, ref_pt_block) # B, C, N
        agg_pt = rearrange(agg_pt, 'b c n -> b n c') # B, N, C
        ref_pt_list.append(agg_pt)
        
        return sqz_target, agg_pt, ref_pt_list, slot_pt, slot_mask
    
    def initialize(self):
        weight_init(self)

class Slot_Attention(nn.Module):
    def __init__(self, in_channels, iters=3):
        super(Slot_Attention, self).__init__()
        self.iters = iters
        self.FAT = Feature_Aggregation_Transformer(in_channels)

    def forward(self, ref_pt_list, slots):
        for _ in range(self.iters):
            filtered_ref_pt = prototype_filter(slots, ref_pt_list)
            filtered_ref_pt = rearrange(filtered_ref_pt, 'b cl c cn -> b c cn cl')
            slots = rearrange(slots, 'b n c -> b c n')
            slots = self.FAT(slots, filtered_ref_pt) # B, C, 2
            slots = rearrange(slots, 'b c n -> b n c') # B, 2, C

        return slots

    def initialize(self):
        weight_init(self)

class GSANet(torch.nn.Module):
    def __init__(self, num_slots=2):
        super(GSANet, self).__init__()
        self.cfg = None
        self.rgb_encoder = SegformerForImageClassification.from_pretrained("nvidia/mit-b1")
        self.flow_encoder = SegformerForImageClassification.from_pretrained("nvidia/mit-b1")


        self.rgb_MFFM1 = Multi_Scale_Feature_Fusion_Module(512, 128)
        self.rgb_MFFM2 = Multi_Scale_Feature_Fusion_Module(320, 128)
        self.rgb_MFFM3 = Multi_Scale_Feature_Fusion_Module(128, 128)
        self.rgb_MFFM4 = Multi_Scale_Feature_Fusion_Module(64, 128)

        self.flow_MFFM1 = Multi_Scale_Feature_Fusion_Module(512, 128)
        self.flow_MFFM2 = Multi_Scale_Feature_Fusion_Module(320, 128)
        self.flow_MFFM3 = Multi_Scale_Feature_Fusion_Module(128, 128)
        self.flow_MFFM4 = Multi_Scale_Feature_Fusion_Module(64, 128)

        self.rgb_SG = Slot_Generator(128, 128, num_slots)
        self.flow_SG = Slot_Generator(128, 128, num_slots)

        self.rgb_SA = Slot_Attention(128)
        self.flow_SA = Slot_Attention(128)
        
        self.fusion1 = Fusion(128)
        self.fusion2 = Fusion(128)
        self.fusion3 = Fusion(128)

        self.rgb_fusion = nn.Conv2d(128 + 64 + num_slots, 128, 1)
        self.flow_fusion = nn.Conv2d(128 + 64 + num_slots, 128, 1)

        self.total_fuse1 = Res(128 * 2, 128)
        self.total_fuse2 = Res(128 * 2, 128)
        self.total_fuse3 = Res(128 * 2, 128)
        self.total_fuse4 = Res(128 * 2, 128)

        self.linearr5 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.linearr6 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.linearr7 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.linearr8 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

        self.initialize()

    def get_ref_feature(self, rgb_ref, flow_ref):
        N = rgb_ref.size(1) // 3

        rgb_ref4 = []
        flow_ref4 = []

        with torch.no_grad():
            for n in range(N):
                ref_slice = rgb_ref[:, n * 3: n * 3 + 3, :, :]
                x4_r = (self.rgb_encoder(ref_slice, output_hidden_states=True).hidden_states)[3]
                x4_r = self.rgb_MFFM1(x4_r)
                rgb_ref4.append(x4_r)
            
            for n in range(N):
                ref_slice = flow_ref[:, n * 3: n * 3 + 3, :, :]
                x4_f = (self.flow_encoder(ref_slice, output_hidden_states=True).hidden_states)[3]
                x4_f = self.flow_MFFM1(x4_f)
                flow_ref4.append(x4_f)
    
        return rgb_ref4, flow_ref4


    def forward(self, rgb, flow, rgb_ref, flow_ref):
        shape = rgb.size()[2:]
        rgb_ref, flow_ref = self.get_ref_feature(rgb_ref, flow_ref)

        rgb_features = self.rgb_encoder(rgb, output_hidden_states=True).hidden_states
        x4_r, x3_r, x2_r, x1_r = rgb_features[3], rgb_features[2], rgb_features[1], rgb_features[0]

        flow_features = self.flow_encoder(flow, output_hidden_states=True).hidden_states
        x4_f, x3_f, x2_f, x1_f = flow_features[3], flow_features[2], flow_features[1], flow_features[0]

        x1_r, x2_r, x3_r, x4_r = self.rgb_MFFM4(x1_r), self.rgb_MFFM3(x2_r), self.rgb_MFFM2(x3_r), self.rgb_MFFM1(x4_r)
        x1_f, x2_f, x3_f, x4_f = self.flow_MFFM4(x1_f), self.flow_MFFM3(x2_f), self.flow_MFFM2(x3_f), self.flow_MFFM1(x4_f)

        sqz_x3_r, rgb_agg_pt, rgb_ref_pt_list, rgb_slots, rgb_slot_mask = self.rgb_SG(x3_r, rgb_ref)
        sqz_x3_f, flow_agg_pt, flow_ref_pt_list, flow_slots, flow_slot_mask = self.flow_SG(x3_f, flow_ref)

        rgb_refine_slots = self.rgb_SA(rgb_ref_pt_list, rgb_slots)
        flow_refine_slots = self.flow_SA(flow_ref_pt_list, flow_slots)

        new_x3_r = get_correlation_map(sqz_x3_r, rgb_agg_pt)
        new_x3_f = get_correlation_map(sqz_x3_f, flow_agg_pt)

        slot_x3_r = get_correlation_map(sqz_x3_r, rgb_refine_slots)
        slot_x3_f = get_correlation_map(sqz_x3_f, flow_refine_slots)

        x3_r = self.rgb_fusion(torch.cat([x3_r, new_x3_r, slot_x3_r], dim=1))
        x3_f = self.flow_fusion(torch.cat([x3_f, new_x3_f, slot_x3_f], dim=1))

        x1 = self.total_fuse1(torch.cat([x1_r, x1_f], dim=1))
        x2 = self.total_fuse2(torch.cat([x2_r, x2_f], dim=1))
        x3 = self.total_fuse3(torch.cat([x3_r, x3_f], dim=1))
        x4 = self.total_fuse4(torch.cat([x4_r, x4_f], dim=1))

        x1 = x1
        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear')
        x3 = F.interpolate(x3, scale_factor=4, mode='bilinear')
        x4 = F.interpolate(x4, scale_factor=8, mode='bilinear')

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

        slot_x3_r = (slot_x3_r + 1) / 2
        slot_x3_f = (slot_x3_f + 1) / 2
        
        fg_rgb_rm = slot_x3_r[:, 0, :, :].unsqueeze(1)
        fg_flow_rm = slot_x3_f[:, 0, :, :].unsqueeze(1)
        bg_rgb_rm = slot_x3_r[:, 1, :, :].unsqueeze(1)
        bg_flow_rm = slot_x3_f[:, 1, :, :].unsqueeze(1)

        preds_list = [out_1, out_2, out_3, out_4]

        coarse_slot_r = resize(rgb_slot_mask[:, 0, :, :].unsqueeze(1), shape)
        coarse_slot_f = resize(flow_slot_mask[:, 0, :, :].unsqueeze(1), shape)

        fine_slot1_r = resize(fg_rgb_rm, shape)
        fine_slot2_r = resize(1 - bg_rgb_rm, shape)

        fine_slot1_f = resize(fg_flow_rm, shape)
        fine_slot2_f = resize(1 - bg_flow_rm, shape)

        coarse_slot_list = [coarse_slot_r, coarse_slot_f]
        fine_slot_list = [fine_slot1_r, fine_slot2_r, fine_slot1_f, fine_slot2_f]

        total_list = preds_list + coarse_slot_list + fine_slot_list

        return total_list, fine_slot_list

    def initialize(self):
        weight_init(self)