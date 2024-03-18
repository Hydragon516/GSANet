import os
import numpy as np
import cv2
import random

import torch
import torch.nn.functional as F
from dataloader.data_for_video import get_loader, get_testloader
from loss import structure_loss
from model.model_for_video import GSANet

import config as config
from logger import *

os.environ["CUDA_VISIBLE_DEVICES"] = config.TRAIN['GPU']

folder = os.path.join(config.DATA['data_root'], config.DATA['DAVIS_val'])
valid_list = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
best_jaccard = 0

def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    ckpt_name = []
    cnt = 0
    for name, param in state_dict.items():
        name = name.replace("module.", "")
        try:
            if name not in list(own_state.keys()):
                ckpt_name.append(name)
                continue
            own_state[name].copy_(param)
            cnt += 1
            print(name)
        except:
            pass

        try:
            new_name = name.replace("rgb_", "flow_")
            if new_name not in list(own_state.keys()):
                ckpt_name.append(new_name)
                continue
            own_state[new_name].copy_(param)
            cnt += 1
            print(new_name)
        except:
            pass

    print('#reused param: {}'.format(cnt))

    return model

def pred_convert(x, size):
    x = F.upsample(x, size=size, mode='bilinear', align_corners=False)
    x = x.permute(0, 2, 3, 1).cpu().detach().squeeze(0).squeeze(-1).numpy()
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return x

def train(epoch, trainloader, optimizer, model, device, scheduler):
    model.train()
    avg_loss1 = 0

    PTTM = pttm()

    for idx, batch in enumerate(trainloader):
        PTTM.print_status(epoch, idx, trainloader)
        images, gts, flows, ref_images, ref_flows = batch
        
        if config.TRAIN['img_size'] == 384:
            s = 384
        else:
            s = random.choice([384, 416, 448, 480, 512])
        
        if images.size(0) == 1:
            continue

        images, gts, flows = images.to(device), gts.to(device), flows.to(device)
        ref_images, ref_flows = ref_images.to(device), ref_flows.to(device)

        images = F.interpolate(images, size=(s, s), mode='bicubic', align_corners=False)
        flows = F.interpolate(flows, size=(s, s), mode='bicubic', align_corners=False)
        gts = F.interpolate(gts, size=(s, s), mode='bicubic', align_corners=False)
        ref_images = F.interpolate(ref_images, size=(s, s), mode='bicubic', align_corners=False)
        ref_flows = F.interpolate(ref_flows, size=(s, s), mode='bicubic', align_corners=False)

        total_loss = 0
        optimizer.zero_grad()
        pred, _ = model(images, flows, ref_images, ref_flows)

        for l in range(len(pred)):
            total_loss += structure_loss(pred[l], gts)

        avg_loss1 += total_loss.item()

        total_loss.backward()
        optimizer.step()
        scheduler.step()
    
    print("")

    avg_loss1 = avg_loss1 / (idx + 1)

    print(
        "Epoch: #{0} Batch: {1}\t"
        "Lr: {lr:.7f}\t"
        "LOSS pred: {loss1:.4f}\n"
        .format(epoch, idx, lr=optimizer.param_groups[-1]['lr'], loss1=avg_loss1)
    )

    avg_loss1 = 0

def valid(epoch, model, device, work_dir):
    global best_jaccard
    
    print("Evaluating model...")

    test_loader = get_testloader(config.DATA['DAVIS_val'])
    J_buffer = {item:[] for item in valid_list}

    model.eval()
    with torch.no_grad():
        PTTM = pttm()
        
        for idx, batch in enumerate(test_loader):
            PTTM.print_status(epoch, idx, test_loader)
            image, gt, flow, info, _, ref_images, ref_flows = batch

            s = config.TRAIN['img_size']
            image = F.interpolate(image, size=(s, s), mode='bicubic', align_corners=False)
            flow = F.interpolate(flow, size=(s, s), mode='bicubic', align_corners=False)
            gt = F.interpolate(gt, size=(s, s), mode='bicubic', align_corners=False)
            ref_images = F.interpolate(ref_images, size=(s, s), mode='bicubic', align_corners=False)
            ref_flows = F.interpolate(ref_flows, size=(s, s), mode='bicubic', align_corners=False)

            B = image.shape[0]

            ori_H = info[0][0]
            ori_W = info[0][1]

            image = image.to(device)
            flow = flow.to(device)
            gt = gt.to(device)
            ref_images = ref_images.to(device)
            ref_flows = ref_flows.to(device)

            preds, _ = model(image, flow, ref_images, ref_flows)

            res = preds[0]

            for b in range(B):
                res_slice = res[b, :, :, :].unsqueeze(0).float()
                gt_slice = gt[b, :, :, :].unsqueeze(0).float()
                info_slice = info[1][b]

                res_slice = F.upsample(res_slice, size=(ori_H[b].item(), ori_W[b].item()), mode='bilinear', align_corners=False)
                res_slice = (res_slice - res_slice.min()) / (res_slice.max() - res_slice.min() + 1e-8)
                res_slice[res_slice > 0.5] = 1
                res_slice[res_slice <= 0.5] = 0

                gt_slice = F.upsample(gt_slice, size=(ori_H[b].item(), ori_W[b].item()), mode='bilinear', align_corners=False)
                gt_slice /= (torch.max(gt_slice) + 1e-8)

                J_buffer[info_slice].append((torch.sum(res_slice * gt_slice) / (torch.sum(res_slice) + torch.sum(gt_slice) - torch.sum(res_slice * gt_slice))).item())
        
        print("")

        total_jaccard_list = []

        for name in valid_list:
            total_jaccard_list.append(np.mean(np.array(J_buffer[name])))
        total_jaccard = np.mean(np.array(total_jaccard_list))
        
        print("total J: {:.4f}".format(total_jaccard), "best J: {:.4f}".format(best_jaccard))

        if total_jaccard > best_jaccard:
            best_jaccard = total_jaccard

            save_model(work_dir, epoch, model, 'best')
            print("Saved best model!")

            return True
    
    return False

def visual(device, work_dir):
    model = GSANet(num_slots=config.TRAIN['slot_num'])
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    model_dir = os.path.join(work_dir, "model")

    checkpoint = torch.load(model_dir + "/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loader = get_testloader(config.DATA['DAVIS_val'])

    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            image, gt, flow, info, img_for_post, ref_images, ref_flows = batch
            B = image.shape[0]

            ori_H = info[0][0]
            ori_W = info[0][1]

            image = image.to(device)
            flow = flow.to(device)
            ref_images = ref_images.to(device)
            ref_flows = ref_flows.to(device)

            s = config.TRAIN['img_size']
            image = F.interpolate(image, size=(s, s), mode='bicubic', align_corners=False)
            flow = F.interpolate(flow, size=(s, s), mode='bicubic', align_corners=False)
            gt = F.interpolate(gt, size=(s, s), mode='bicubic', align_corners=False)
            ref_images = F.interpolate(ref_images, size=(s, s), mode='bicubic', align_corners=False)
            ref_flows = F.interpolate(ref_flows, size=(s, s), mode='bicubic', align_corners=False)

            preds, slots = model(image, flow, ref_images, ref_flows)

            flow = flow.permute(0, 2, 3, 1).cpu().detach()
            res = preds[0]

            for b in range(B):
                res_slice = res[b, :, :, :].unsqueeze(0)
                gt_slice = gt[b, :, :, :].squeeze(0)
                flow_slice = flow[b, :, :, :].squeeze(0)
                ori_image_slice = img_for_post[b, :, :, :].squeeze(0)
                
                gt_slice = np.asarray(gt_slice, np.float32)
                gt_slice /= (gt_slice.max() + 1e-8)

                res_slice = pred_convert(res_slice, (ori_H[b].item(), ori_W[b].item()))

                cat_res = cv2.cvtColor(np.array(res_slice * 255), cv2.COLOR_GRAY2BGR)
                cat_res = cv2.resize(cat_res, dsize=(ori_H[b].item(), ori_W[b].item()), interpolation=cv2.INTER_AREA)
                cat_res = cat_res.astype(np.uint8)
                
                cat_gt = cv2.cvtColor(np.array(gt_slice * 255), cv2.COLOR_GRAY2BGR)
                cat_gt = cv2.resize(cat_gt, dsize=(ori_H[b].item(), ori_W[b].item()), interpolation=cv2.INTER_AREA)
                cat_gt = cat_gt.astype(np.uint8)

                cat_flow = cv2.cvtColor(np.array(flow_slice * 255), cv2.COLOR_RGB2BGR)
                cat_flow = cv2.resize(cat_flow, dsize=(ori_H[b].item(), ori_W[b].item()), interpolation=cv2.INTER_AREA)
                cat_flow = cat_flow.astype(np.uint8)

                cat_ori = cv2.cvtColor(np.array(ori_image_slice), cv2.COLOR_RGB2BGR)
                cat_ori = cv2.resize(cat_ori, dsize=(ori_H[b].item(), ori_W[b].item()), interpolation=cv2.INTER_AREA)
                cat_ori = cat_ori.astype(np.uint8)

                result_list = [cat_ori, cat_flow, cat_res, cat_gt]

                for slot in slots:
                    slot_slice = slot[b, 0, :, :].unsqueeze(0).unsqueeze(0)
                    slot_slice = pred_convert(slot_slice, (ori_H[b].item(), ori_W[b].item()))
                    slot_np = cv2.cvtColor(np.array(slot_slice * 255), cv2.COLOR_GRAY2BGR)
                    slot_np = cv2.resize(slot_np, dsize=(ori_H[b].item(), ori_W[b].item()), interpolation=cv2.INTER_AREA)
                    slot_np = slot_np.astype(np.uint8)
                    slot_np = cv2.applyColorMap(slot_np, cv2.COLORMAP_JET)
                    slot_np = cv2.addWeighted(cat_ori, 0.5, slot_np, 0.5, 0)
                    result_list.append(slot_np)

                result = cv2.hconcat(result_list)

                valid_name = info[1][b]
                name = info[2][b]

                total_dir = os.path.join(work_dir, "result", "total", valid_name)
                if not os.path.exists(total_dir):
                    os.makedirs(total_dir)
                
                pred_dir = os.path.join(work_dir, "result", "pred", valid_name)
                if not os.path.exists(pred_dir):
                    os.makedirs(pred_dir)
                
                gt_dir = os.path.join(work_dir, "result", "gt", valid_name)
                if not os.path.exists(gt_dir):
                    os.makedirs(gt_dir)
 
                cv2.imwrite(os.path.join(total_dir, name), result)
                cv2.imwrite(os.path.join(pred_dir, name), cat_res)
                cv2.imwrite(os.path.join(gt_dir, name), cat_gt)


def main():
    work_dir = make_new_work_space()
    save_config_file(work_dir)
    
    print("Check device...")
    device = torch.device("cuda")
    print(device)
    print("ok!")

    print("Load model...")
    model = GSANet(num_slots=config.TRAIN['slot_num'])

    checkpoint = torch.load(config.DATA['best_pretrained_model'])
    model = load_my_state_dict(model, checkpoint['model_state_dict'])

    model = model.to(device)
    model = torch.nn.DataParallel(model)
    
    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))
    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("Total number of learning parameters: {}".format(num_params_update))
    print("ok!")

    print("Load optimizer...")
    params = model.parameters()
    optimizer = torch.optim.Adam(params, config.TRAIN['learning_rate'])

    train_loader = get_loader()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*config.TRAIN['epoch'], eta_min=config.TRAIN['learning_rate']/10)
    print("ok!")

    print("Training start!")
    for epoch in range(config.TRAIN['epoch']):
        print("Load dataset...")
        train_loader = get_loader()
        print("ok!")
        train(epoch, train_loader, optimizer, model, device, scheduler)
        save = valid(epoch, model, device, work_dir)
        print("")

        if save:    
            visual(device, work_dir)

    print("Training finish!")


if __name__ == '__main__':
    main()