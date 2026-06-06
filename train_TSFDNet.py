import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
working_path = os.path.dirname(os.path.abspath(__file__))
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from utils.loss import CrossEntropyLoss2d, weighted_BCE_logits, SCA_Loss
from utils.utils import accuracy, SCDD_eval_all, AverageMeter
from datasets import RS_SECOND as RS
# from datasets import RS_JL1 as RS
from models.TSFDNet import TSFDNet as Net

# 固定随机种子
seed = 14528
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

NET_NAME = 'TSFDNet'
DATA_NAME = 'SECOND_' + str(seed)

args = {
    'train_batch_size': 4,
    'val_batch_size': 4,
    'lr': 0.0001,
    'epochs': 50,
    'gpu': True,
    'weight_decay': 1e-2,
    'print_freq': 50,
    'predict_step': 5,
    'pred_dir': os.path.join(os.getcwd(), 'results', DATA_NAME),
    'chkpt_dir': os.path.join(os.getcwd(), 'checkpoints', DATA_NAME),
    'log_dir': os.path.join(os.getcwd(), 'logs', DATA_NAME, NET_NAME),
    'load_path': "/home/zhj/TSFDNet/backbone_weights.pth"
}

os.makedirs(args['log_dir'], exist_ok=True)
os.makedirs(args['pred_dir'], exist_ok=True)
os.makedirs(args['chkpt_dir'], exist_ok=True)
writer = SummaryWriter(args['log_dir'])

def main():
    print(f"\nClasses: {RS.ST_CLASSES}\n")
    net = Net(3, output_nc=RS.num_classes, class_names=RS.ST_CLASSES, img_size=512).cuda()
    net = nn.DataParallel(net)

    if os.path.exists(args['load_path']):
        model_dict = net.state_dict()
        pretrained_dict = torch.load(args['load_path'])
        temp_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(temp_dict)
        net.load_state_dict(model_dict)
        print(f"Loaded {len(temp_dict)} keys from backbone weights.")

    train_set = RS.Data('train', random_flip=True)
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)

    val_set = RS.Data('val')
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], shuffle=False, num_workers=4)

    criterion = CrossEntropyLoss2d(ignore_index=0).cuda()

    gamma_params, coop_params, clip_params, base_params = [], [], [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        if "gamma" in name or "alpha" in name:
            gamma_params.append(param)
        elif "prompt_learner" in name or "ctx" in name:
            coop_params.append(param)
        elif "clip_model.transformer.resblocks.11" in name or "clip_model.ln_final" in name:
            clip_params.append(param)
        else:
            base_params.append(param)

    optimizer = optim.AdamW([
        {'params': base_params, 'lr': args['lr'], 'weight_decay': 1e-2},
        {'params': gamma_params, 'lr': args['lr']*10, 'weight_decay': 0},
        {'params': coop_params, 'lr': args['lr'], 'weight_decay': 5e-4},
        {'params': clip_params, 'lr': args['lr']*0.01, 'weight_decay': 1e-3}
    ], betas=(0.9, 0.999))

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'], eta_min=1e-6)

    train(train_loader, net, criterion, optimizer, scheduler, val_loader)
    writer.close()
    print('Training finished.')

def train(train_loader, net, criterion, optimizer, scheduler, val_loader):
    bestaccT = bestaccV = 0
    bestFscdV = 0.0
    bestloss = 1.0
    criterion_sc = SCA_Loss().cuda()
    curr_epoch = 0
    begin_time = time.time()

    while curr_epoch < args['epochs']:
        net.train()
        acc_meter = AverageMeter()
        train_seg_loss = AverageMeter()
        train_bn_loss = AverageMeter()
        train_sc_loss = AverageMeter()
        torch.cuda.empty_cache()
        start = time.time()

        for i, data in enumerate(train_loader):
            imgs_A, imgs_B, labels_A, labels_B = data
            if args['gpu']:
                imgs_A = imgs_A.cuda()
                imgs_B = imgs_B.cuda()
                labels_A = labels_A.cuda().long()
                labels_B = labels_B.cuda().long()
                labels_bn = (labels_A>0).unsqueeze(1).float().cuda()

            optimizer.zero_grad()
            out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)

            loss_bn = weighted_BCE_logits(out_change[0], labels_bn) if isinstance(out_change, list) else weighted_BCE_logits(out_change, labels_bn)
            if isinstance(out_change, list) and len(out_change)>1:
                for aux_out in out_change[1:]:
                    loss_bn += 0.4*weighted_BCE_logits(aux_out, labels_bn)

            loss_seg = 0.5*(criterion(outputs_A*labels_bn, labels_A) + criterion(outputs_B*labels_bn, labels_B))
            loss_sc = criterion_sc(outputs_A, outputs_B, labels_bn)
            loss = loss_seg + loss_bn + loss_sc

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            # 更新 Accuracy
            outputs_A = outputs_A.cpu().detach()
            outputs_B = outputs_B.cpu().detach()
            labels_A_np = labels_A.cpu().detach().numpy()
            labels_B_np = labels_B.cpu().detach().numpy()
            final_change_pred = out_change[0] if isinstance(out_change, list) else out_change
            change_mask = F.sigmoid(final_change_pred).cpu().detach()>0.5
            preds_A = (torch.argmax(outputs_A, dim=1) * change_mask.squeeze().long()).numpy()
            preds_B = (torch.argmax(outputs_B, dim=1) * change_mask.squeeze().long()).numpy()
            acc_curr_meter = AverageMeter()
            for pred_A, pred_B, label_A, label_B in zip(preds_A, preds_B, labels_A_np, labels_B_np):
                acc_A, _ = accuracy(pred_A, label_A)
                acc_B, _ = accuracy(pred_B, label_B)
                acc_curr_meter.update((acc_A + acc_B)/2)
            acc_meter.update(acc_curr_meter.avg)

            train_seg_loss.update(loss_seg.cpu().detach().numpy())
            train_bn_loss.update(loss_bn.cpu().detach().numpy())
            train_sc_loss.update(loss_sc.cpu().detach().numpy())

            if (i+1) % args['print_freq'] == 0:
                print('[epoch %d] [iter %d/%d %.1fs] lr %.6f seg %.4f bn %.4f sc %.4f acc %.2f' %
                    (curr_epoch, i+1, len(train_loader), time.time()-start, optimizer.param_groups[0]['lr'],
                    train_seg_loss.val, train_bn_loss.val, train_sc_loss.val, acc_meter.val*100))

        # 验证
        Fscd_v, mIoU_v, Sek_v, acc_v, loss_v = validate(val_loader, net, criterion, curr_epoch)
        if acc_meter.avg>0: bestaccT=acc_meter.avg
        if Fscd_v>bestFscdV:
            bestFscdV=Fscd_v
            bestaccV=acc_v
            bestloss=loss_v
            torch.save(net.state_dict(), os.path.join(args['chkpt_dir'],
                NET_NAME+'_%de_mIoU%.2f_Sek%.2f_Fscd%.2f_OA%.2f.pth' %
                (curr_epoch, mIoU_v*100, Sek_v*100, Fscd_v*100, acc_v*100)))

        print('Total time: %.1fs Best Train acc %.2f, Val Fscd %.2f acc %.2f loss %.4f' %
            (time.time()-begin_time, bestaccT*100, bestFscdV*100, bestaccV*100, bestloss))
        curr_epoch += 1
        scheduler.step()


def validate(val_loader, net, criterion, curr_epoch):
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()
    val_loss = AverageMeter()
    acc_meter = AverageMeter()
    preds_all, labels_all = [], []

    for data in val_loader:
        imgs_A, imgs_B, labels_A, labels_B = data
        imgs_A = imgs_A.cuda()
        imgs_B = imgs_B.cuda()
        labels_A = labels_A.cuda().long()
        labels_B = labels_B.cuda().long()
        labels_bn = (labels_A>0).unsqueeze(1).float().cuda()

        with torch.no_grad():
            res = net(imgs_A, imgs_B)
            out_change = res[0]
            outputs_A = res[1]
            outputs_B = res[2]
            if isinstance(out_change, list): out_change = out_change[0]

            # TTA: horizontal flip
            imgs_A_flip = torch.flip(imgs_A, [3])
            imgs_B_flip = torch.flip(imgs_B, [3])
            res_flip = net(imgs_A_flip, imgs_B_flip)
            out_change_flip = res_flip[0]
            outputs_A_flip = res_flip[1]
            outputs_B_flip = res_flip[2]
            if isinstance(out_change_flip, list): out_change_flip = out_change_flip[0]

            out_change = (out_change + torch.flip(out_change_flip, [3])) / 2
            outputs_A = (outputs_A + torch.flip(outputs_A_flip, [3])) / 2
            outputs_B = (outputs_B + torch.flip(outputs_B_flip, [3])) / 2

            loss_A = criterion(outputs_A*labels_bn, labels_A)
            loss_B = criterion(outputs_B*labels_bn, labels_B)
            loss = 0.5*(loss_A+loss_B) + weighted_BCE_logits(out_change, labels_bn)
            val_loss.update(loss.cpu().detach().numpy())

            labels_A = labels_A.cpu().detach().numpy()
            labels_B = labels_B.cpu().detach().numpy()
            outputs_A = outputs_A.cpu().detach()
            outputs_B = outputs_B.cpu().detach()
            change_mask = F.sigmoid(out_change).cpu().detach()>0.5
            preds_A = (torch.argmax(outputs_A, dim=1)*change_mask.squeeze().long()).numpy()
            preds_B = (torch.argmax(outputs_B, dim=1)*change_mask.squeeze().long()).numpy()

            for pred_A, pred_B, label_A, label_B in zip(preds_A, preds_B, labels_A, labels_B):
                acc_A, _ = accuracy(pred_A, label_A)
                acc_B, _ = accuracy(pred_B, label_B)
                preds_all.append(pred_A)
                preds_all.append(pred_B)
                labels_all.append(label_A)
                labels_all.append(label_B)
                acc_meter.update((acc_A+acc_B)/2)

    Fscd, IoU_mean, Sek = SCDD_eval_all(preds_all, labels_all, RS.num_classes)
    print('%.1fs Val loss %.2f Fscd %.2f IoU %.2f Sek %.2f Accuracy %.2f' %
          (time.time()-start, val_loss.average(), Fscd*100, IoU_mean*100, Sek*100, acc_meter.average()*100))

    return Fscd, IoU_mean, Sek, acc_meter.avg, val_loss.avg


if __name__ == '__main__':
    main()
