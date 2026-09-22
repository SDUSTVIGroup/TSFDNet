import time
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import accuracy, SCDD_eval_all, AverageMeter
from datasets import RS_SECOND as RS
from models.TSFDNet import TSFDNet as Net


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TSFDNet on SECOND with four-direction test-time augmentation.",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pred_batch_size', type=int, default=1, help='prediction batch size')
    parser.add_argument('--test_dir', default='.../SECOND/test', help='test split directory containing im1, im2, label1 and label2')
    parser.add_argument('--chkpt_path', default='/home/zhj/TSFDNet/checkpoints/SECOND_14528/TSFDNet_23e_mIoU74.27_Sek25.07_Fscd64.28_OA88.30.pth', help='converted TSFDNet state_dict checkpoint')
    return parser.parse_args()


def main():
    begin_time = time.time()
    opt = parse_args()
    net = Net(input_nc=3,output_nc=RS.num_classes,class_names=RS.ST_CLASSES,img_size=512).cuda()
    ckpt = torch.load(opt.chkpt_path, map_location='cpu')

    # Remove the DataParallel prefix before strict loading.
    new_ckpt = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            new_ckpt[k[7:]] = v
        else:
            new_ckpt[k] = v

    net.load_state_dict(new_ckpt, strict=True)
    print("Checkpoint loaded successfully.")

    net.eval()

    test_set = RS.Data(opt.test_dir, random_flip = False)
    test_loader = DataLoader(test_set, batch_size=opt.pred_batch_size)
    validate(test_loader, net)
    time_use = time.time() - begin_time
    print('Total time: %.2fs'%time_use)


def validate(val_loader, net, threshold=0.5):
    net.eval()
    torch.cuda.empty_cache()

    start = time.time()
    acc_meter = AverageMeter()

    preds_all = []
    labels_all = []

    for data in tqdm(val_loader):

        imgs_A, imgs_B, labels_A, labels_B = data

        imgs_A = imgs_A.cuda().float()
        imgs_B = imgs_B.cuda().float()
        labels_A = labels_A.cuda().long()
        labels_B = labels_B.cuda().long()

        with torch.no_grad():

            # Original, horizontal, vertical and combined flips.
            res0 = net(imgs_A, imgs_B)
            out0 = res0[0]
            pred_A0 = res0[1]
            pred_B0 = res0[2]

            if isinstance(out0, list):
                out0 = out0[0]

            res_h = net(
                torch.flip(imgs_A, [3]),
                torch.flip(imgs_B, [3])
            )

            out_h = res_h[0]
            pred_A_h = res_h[1]
            pred_B_h = res_h[2]

            if isinstance(out_h, list):
                out_h = out_h[0]

            out_h = torch.flip(out_h, [3])
            pred_A_h = torch.flip(pred_A_h, [3])
            pred_B_h = torch.flip(pred_B_h, [3])

            res_v = net(
                torch.flip(imgs_A, [2]),
                torch.flip(imgs_B, [2])
            )

            out_v = res_v[0]
            pred_A_v = res_v[1]
            pred_B_v = res_v[2]

            if isinstance(out_v, list):
                out_v = out_v[0]

            out_v = torch.flip(out_v, [2])
            pred_A_v = torch.flip(pred_A_v, [2])
            pred_B_v = torch.flip(pred_B_v, [2])

            res_hv = net(
                torch.flip(imgs_A, [2, 3]),
                torch.flip(imgs_B, [2, 3])
            )

            out_hv = res_hv[0]
            pred_A_hv = res_hv[1]
            pred_B_hv = res_hv[2]

            if isinstance(out_hv, list):
                out_hv = out_hv[0]

            out_hv = torch.flip(out_hv, [2, 3])
            pred_A_hv = torch.flip(pred_A_hv, [2, 3])
            pred_B_hv = torch.flip(pred_B_hv, [2, 3])

            # Average aligned logits before sigmoid and argmax.
            out_change = (
                out0 + out_h + out_v + out_hv
            ) / 4.0

            outputs_A = (
                pred_A0 + pred_A_h + pred_A_v + pred_A_hv
            ) / 4.0

            outputs_B = (
                pred_B0 + pred_B_h + pred_B_v + pred_B_hv
            ) / 4.0

        labels_A_np = labels_A.cpu().numpy()
        labels_B_np = labels_B.cpu().numpy()

        outputs_A_cpu = outputs_A.cpu()
        outputs_B_cpu = outputs_B.cpu()

        change_mask = (
            torch.sigmoid(out_change).cpu() > threshold
        )

        preds_A = torch.argmax(outputs_A_cpu, dim=1)
        preds_B = torch.argmax(outputs_B_cpu, dim=1)

        preds_A = (
            preds_A * change_mask.squeeze(1).long()
        ).numpy()

        preds_B = (
            preds_B * change_mask.squeeze(1).long()
        ).numpy()

        for pred_A, pred_B, label_A, label_B in zip(
                preds_A,
                preds_B,
                labels_A_np,
                labels_B_np):

            acc_A, _ = accuracy(pred_A, label_A)
            acc_B, _ = accuracy(pred_B, label_B)

            preds_all.append(pred_A)
            preds_all.append(pred_B)
            labels_all.append(label_A)
            labels_all.append(label_B)

            acc_meter.update((acc_A + acc_B) * 0.5)

    Fscd, IoU_mean, Sek = SCDD_eval_all(
        preds_all,
        labels_all,
        RS.num_classes
    )

    curr_time = time.time() - start

    print(
        '%.1fs Fscd: %.2f IoU: %.2f Sek: %.2f Accuracy: %.2f Threshold: %.2f'
        % (
            curr_time,
            Fscd * 100,
            IoU_mean * 100,
            Sek * 100,
            acc_meter.average() * 100,
            threshold
        )
    )

    return Fscd, IoU_mean, Sek, acc_meter.avg

if __name__ == '__main__':
    main()
