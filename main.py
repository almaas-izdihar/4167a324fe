from __future__ import print_function
from cProfile import label
from secrets import choice
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing as mp
import torch.utils.data.distributed
import torch.distributed as dist
from loader import custom_dataloader
from models.network import get_network
from loss.kd_loss import KD, RefineLoss, Mixup
from utils.dir_maker import DirectroyMaker
from utils.AverageMeter import AverageMeter
from utils.metric import metric_ece_aurc_eaurc
from utils.color import Colorer
from utils.etc import progress_bar, is_main_process, save_on_master, paser_config_save, set_logging_defaults
import os, logging
import argparse
import numpy as np
from collections import defaultdict
import random
def parse_args():
    parser = argparse.ArgumentParser(description='None')
    parser.add_argument('--experiment_type', default='exp',type=str)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--weight', default=4.0, type=float)
    parser.add_argument('--weight2', default=1.0, type=float)
    parser.add_argument('--lr_decay_rate', default=0.1, type=float)
    parser.add_argument('--lr_decay_schedule', default=[100, 150], nargs='*', type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--end_epoch', default=200, type=int)
    parser.add_argument('--EHSKD', action='store_true')
    parser.add_argument('--mult_sample', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--experiments_dir', type=str, default='models')
    parser.add_argument('--classifier_type', type=str, default='ResNet18')
    parser.add_argument('--data_path', type=str, default="../../Cifar100")
    parser.add_argument('--data_type', type=str, default="cifar100")
    parser.add_argument('--saveckp_freq', default=200, type=int)
    parser.add_argument('--rank', default=-1, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_backend', default='nccl', type=str)
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:8080', type=str)
    parser.add_argument('--workers', default=16, type=int)
    parser.add_argument('--multiprocessing_distributed', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--confidence_gate', action='store_true')
    parser.add_argument('--tau_max', default=0.7, type=float)
    parser.add_argument('--tau_min', default=0.1, type=float)
    parser.add_argument('--correct_gate', action='store_true')
    parser.add_argument('--soft_weight', action='store_true')
    parser.add_argument('--true_class_weight', action='store_true')
    parser.add_argument('--seed', default=2024, type=int)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return check_args(args)
def check_args(args):
    try:
        assert args.end_epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args
def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    for milestone in args.lr_decay_schedule:
        lr *= args.lr_decay_rate if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res    
C = Colorer.instance()
def main():
    args = parse_args()
    print(C.green("[!] Start the PS-KD."))
    print(C.green("[!] Created by LG CNS AI Research(LAIR)"))
    dir_maker = DirectroyMaker(root=args.experiments_dir, save_model=True, save_log=True, save_config=True)
    model_log_config_dir = dir_maker.experiments_dir_maker(args)
    model_dir = model_log_config_dir[0]
    log_dir = model_log_config_dir[1]
    config_dir = model_log_config_dir[2]
    paser_config_save(args, config_dir)
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,model_dir, log_dir, args))
        print(C.green("[!] Multi/Single Node, Multi-GPU All multiprocessing_distributed Training Done."))
        print(C.underline(C.red2('[Info] Save Model dir:')), C.red2(model_dir))
        print(C.underline(C.red2('[Info] Log dir:')), C.red2(log_dir))
        print(C.underline(C.red2('[Info] Config dir:')), C.red2(config_dir)) 
    else:
        print(C.green("[!] Multi/Single Node, Single-GPU per node, multiprocessing_distributed Training Done."))
        main_worker(0, ngpus_per_node, model_dir, log_dir, args)
        print(C.green("[!] All Single GPU Training Done"))
        print(C.underline(C.red2('[Info] Save Model dir:')), C.red2(model_dir))
        print(C.underline(C.red2('[Info] Log dir:')), C.red2(log_dir))
        print(C.underline(C.red2('[Info] Config dir:')), C.red2(config_dir))
def main_worker(gpu, ngpus_per_node, model_dir, log_dir, args):
    best_acc = 0
    net = get_network(args)
    args.ngpus_per_node = ngpus_per_node
    args.gpu = gpu
    if args.gpu is not None:
        print(C.underline(C.yellow("[Info] Use GPU : {} for training".format(args.gpu))))
    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    print(C.green("[!] [Rank {}] Distributed Init Setting Done.".format(args.rank)))
    if not torch.cuda.is_available():
        print(C.red2("[Warnning] Using CPU, this will be slow."))
    elif args.distributed:
        if args.gpu is not None:
            print(C.green("[!] [Rank {}] Distributed DataParallel Setting Start".format(args.rank)))
            torch.cuda.set_device(args.gpu)
            net.cuda(args.gpu)
            args.workers = int((args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
            args.batch_size = int(args.batch_size / args.ngpus_per_node)
            print(C.underline(C.yellow("[Info] [Rank {}] Workers: {}".format(args.rank, args.workers))))
            print(C.underline(C.yellow("[Info] [Rank {}] Batch_size: {}".format(args.rank, args.batch_size))))
            net = torch.nn.parallel.DistributedDataParallel(net,device_ids=[args.gpu],broadcast_buffers=False,find_unused_parameters=True)
            print(C.green("[!] [Rank {}] Distributed DataParallel Setting End".format(args.rank)))
        else:
            net.cuda()
            net = torch.nn.parallel.DistributedDataParallel(net)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        net = net.cuda(args.gpu)
    else:
        net = torch.nn.DataParallel(net).cuda()
    set_logging_defaults(log_dir, args)
    train_loader, valid_loader, train_sampler = custom_dataloader.dataloader(args)
    criterion_CE = nn.CrossEntropyLoss().cuda(args.gpu)
    if args.EHSKD:
        criterion_KD = KD().cuda(args.gpu)
    else:
        criterion_KD = None
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    scaler = GradScaler()
    all_predictions = torch.zeros(len(train_loader.dataset), len(train_loader.dataset.classes), dtype=torch.float32)
    now_predictions = torch.zeros(len(train_loader.dataset), len(train_loader.dataset.classes), dtype=torch.float32)
    print(C.underline(C.yellow("[Info] all_predictions matrix shape {}".format(all_predictions.shape))))
    if args.resume:
        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        else:
            if args.distributed:
                dist.barrier()
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
        args.start_epoch = checkpoint['epoch'] + 1 
        best_acc = checkpoint['best_acc']
        all_predictions = checkpoint['prev_predictions'].cpu()
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(C.green("[!] [Rank {}] Model loaded".format(args.rank)))
        del checkpoint
    for epoch in range(args.start_epoch, args.end_epoch):
        adjust_learning_rate(optimizer, epoch, args)
        if args.distributed:
            train_sampler.set_epoch(epoch)
        now_predictions = train(
                                all_predictions,
                                now_predictions,
                                criterion_CE,
                                criterion_KD,
                                optimizer,
                                net,
                                epoch,
                                train_loader,
                                args,
                                scaler)
        if args.distributed:
            dist.barrier()
        acc = val(
                  criterion_CE,
                  net,
                  epoch,
                  valid_loader,
                  args)
        save_dict = {
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_acc' : best_acc,
                    'accuracy' : acc,
                    'prev_predictions': all_predictions
                    }
        if acc > best_acc:
            best_acc = acc
            logger = logging.getLogger('best')
            logger.info('\n###########################\nbest:{}\n###########################'.format(best_acc))
            save_on_master(save_dict,os.path.join(model_dir, 'checkpoint_best.pth'))
            if is_main_process():
                print(C.green("[!] Save best checkpoint."))
        if args.saveckp_freq and (epoch + 1) % args.saveckp_freq == 0:
            save_on_master(save_dict,os.path.join(model_dir, f'checkpoint_{epoch:03}.pth'))
            if is_main_process():
                print(C.green("[!] Save checkpoint."))
    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()
        print(C.green("[!] [Rank {}] Distroy Distributed process".format(args.rank)))
def train(all_predictions,
          now_predictions,
          criterion_CE,
          criterion_KD,
          optimizer,
          net,
          epoch,
          train_loader,
          args,
          scaler):
    train_top1 = AverageMeter()
    train_top5 = AverageMeter()
    train_losses = AverageMeter()
    correct = 0
    total = 0
    net.train()
    current_LR = get_learning_rate(optimizer)[0]
    tau_t = args.tau_max - (args.tau_max - args.tau_min) * (epoch / max(args.end_epoch - 1, 1))
    gate_pct_accum = 0.0
    gate_batches = 0
    for batch_idx, (inputs, targets, input_indices) in enumerate(train_loader):
        optimizer.zero_grad()
        if args.gpu is not None:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
        if args.EHSKD:
            targets_numpy = targets.cpu().detach().numpy()
            identity_matrix = torch.eye(len(train_loader.dataset.classes))
            targets_one_hot = identity_matrix[targets_numpy]
            if epoch == 0:
                all_predictions[input_indices] = targets_one_hot
                now_predictions[input_indices] = targets_one_hot
            outputs_T = now_predictions[input_indices].cuda()
            with autocast():
                outputs_S = net(inputs)
                if isinstance(outputs_S, list):
                    outputs_S = outputs_S[0][0]
                loss = criterion_CE(outputs_S, targets)
                if epoch != 0:
                    _, mixup_loss = Mixup(net, inputs, targets, criterion_CE, alpha=0.4)
                    loss += mixup_loss
                    loss += criterion_KD(outputs_S, outputs_T, 3.0) * 9.0 * args.weight
                    loss += RefineLoss(targets, outputs_S, outputs_T) * args.weight2
            if args.distributed:
                gathered_prediction = [torch.ones_like(outputs_S) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_prediction, outputs_S)
                gathered_prediction = torch.cat(gathered_prediction, dim=0)
                gathered_indices = [torch.ones_like(input_indices.cuda()) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_indices, input_indices.cuda())
                gathered_indices = torch.cat(gathered_indices, dim=0)
        else:
            with autocast():
                outputs_S = net(inputs)
                if isinstance(outputs_S, list):
                    outputs_S = outputs_S[0][0]
                loss = criterion_CE(outputs_S, targets)
        train_losses.update(loss.item(), inputs.size(0))
        err1, err5 = accuracy(outputs_S.data, targets, topk=(1, 5))
        train_top1.update(err1.item(), inputs.size(0))
        train_top5.update(err5.item(), inputs.size(0))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        _, predicted = torch.max(outputs_S, 1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if args.EHSKD:
            if args.distributed:
                for jdx in range(len(gathered_prediction)):
                    idx = gathered_indices[jdx]
                    now_predictions[idx] = (gathered_prediction[jdx].cpu().detach().float() * args.beta
                                            + now_predictions[idx] * (1 - args.beta))
            else:
                # true-class soft weight: scale EMA update by prob at true class (normalized).
                # combines confidence + correctness: overconfident-wrong gets near-zero weight,
                # confident-correct gets weight > 1 (normalized). mean(beta_eff) = beta = 0.5.
                if args.true_class_weight:
                    probs = torch.softmax(outputs_S.detach().float().cpu(), dim=1)
                    w = probs[torch.arange(len(targets)), targets.cpu()]  # prob at true class [B]
                    w_norm = w / w.mean().clamp(min=1e-6)                 # normalize: mean=1.0
                    beta_eff = (args.beta * w_norm).clamp(max=1.0)        # cap at 1.0
                    now_predictions[input_indices] = (
                        probs * beta_eff.unsqueeze(1) +
                        now_predictions[input_indices] * (1 - beta_eff.unsqueeze(1))
                    )
                    gate_pct_accum += w_norm.mean().item() * 100          # always ~100% by design
                # soft weighting: scale per-sample EMA update speed by prediction confidence.
                # NOTE: shown to degrade ECE (~31 vs ~2) — mean(beta_eff)=0.28 causes lagging teacher.
                # kept for ablation reference only.
                elif args.soft_weight:
                    probs = torch.softmax(outputs_S.detach().float().cpu(), dim=1)
                    C = probs.size(1)
                    w = ((probs.max(dim=1).values - 1.0 / C) / (1.0 - 1.0 / C)).clamp(min=0)
                    beta_eff = args.beta * w                          # shape [B]
                    now_predictions[input_indices] = (
                        probs * beta_eff.unsqueeze(1) +
                        now_predictions[input_indices] * (1 - beta_eff.unsqueeze(1))
                    )
                    gate_pct_accum += w.mean().item() * 100           # avg weight as proxy
                # correct-prediction gate: only update memory bank when model predicts correctly.
                # avoids injecting overconfident wrong predictions into the teacher.
                elif args.correct_gate:
                    probs = torch.softmax(outputs_S.detach().float().cpu(), dim=1)
                    _, predicted = torch.max(outputs_S, 1)
                    mask = (predicted.cpu() == targets.cpu())
                    idx  = input_indices[mask]
                    now_predictions[idx] = probs[mask] * args.beta + now_predictions[idx] * (1 - args.beta)
                    gate_pct_accum += mask.float().mean().item() * 100
                # confidence gate: update only when max softmax > tau_t (linearly decayed).
                # NOTE: shown to degrade ECE (~25 vs ~2) and top1 (-2pp vs EMA-SKD) across
                # all tau_max values tested (0.7, 0.3). kept for ablation reference only.
                elif args.confidence_gate:
                    probs = torch.softmax(outputs_S.detach().float().cpu(), dim=1)
                    mask  = probs.max(dim=1).values > tau_t
                    idx   = input_indices[mask]
                    now_predictions[idx] = probs[mask] * args.beta + now_predictions[idx] * (1 - args.beta)
                    gate_pct_accum += mask.float().mean().item() * 100
                else:
                    now_predictions[input_indices] = outputs_S.detach().float().cpu() * args.beta + now_predictions[input_indices] * (1 - args.beta)
                    gate_pct_accum += 100.0
                gate_batches += 1
        progress_bar(epoch,batch_idx, len(train_loader), args, 'lr: {:.1e} |  loss: {:.3f} | top1_acc: {:.3f} | top5_acc: {:.3f} | correct/total({}/{})'.format(
            current_LR, train_losses.avg, train_top1.avg, train_top5.avg, correct, total))
    if args.distributed:
        dist.barrier()
        if args.EHSKD:
            now_preds_gpu = now_predictions.cuda(args.gpu)
            dist.all_reduce(now_preds_gpu, op=dist.ReduceOp.SUM)
            now_predictions = now_preds_gpu.cpu()
    if is_main_process():
        logger = logging.getLogger('train')
        gate_pct = gate_pct_accum / max(gate_batches, 1)
        logger.info('[Epoch {}] [EHSKD {}] [lr {:.1e}] [train_loss {:.3f}] [train_top1_acc {:.3f}] [train_top5_acc {:.3f}] [correct/total {}/{}] [gate {:.1f}%] [tau {:.3f}]'.format(
            epoch,
            args.EHSKD,
            current_LR,
            train_losses.avg,
            train_top1.avg,
            train_top5.avg,
            correct,
            total,
            gate_pct,
            tau_t))
    return now_predictions
def val(criterion_CE,
        net,
        epoch,
        val_loader,
        args):
    val_top1 = AverageMeter()
    val_top5 = AverageMeter()
    val_losses = AverageMeter()
    targets_list = []
    confidences = []
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(val_loader):              
            if args.gpu is not None:
                inputs = inputs.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)
            targets_numpy = targets.cpu().numpy()
            targets_list.extend(targets_numpy.tolist())
            outputs = net(inputs)
            if isinstance(outputs, list):
                outputs = outputs[0][0]
            softmax_predictions = F.softmax(outputs, dim=1)
            softmax_predictions = softmax_predictions.cpu().numpy()
            for values_ in softmax_predictions:
                confidences.append(values_.tolist())
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            loss = criterion_CE(outputs, targets)
            val_losses.update(loss.item(), inputs.size(0))
            err1, err5 = accuracy(outputs.data, targets, topk=(1, 5))
            val_top1.update(err1.item(), inputs.size(0))
            val_top5.update(err5.item(), inputs.size(0))
            progress_bar(epoch, batch_idx, len(val_loader), args,'val_loss: {:.3f} | val_top1_acc: {:.3f} | val_top5_acc: {:.3f} | correct/total({}/{})'.format(
                        val_losses.avg,
                        val_top1.avg,
                        val_top5.avg,
                        correct,
                        total))
    if args.distributed:
        dist.barrier()
    if is_main_process():
        ece,aurc,eaurc = metric_ece_aurc_eaurc(confidences,
                                               targets_list,
                                               bin_size=0.1)
        logger = logging.getLogger('val')
        logger.info('[Epoch {}] [val_loss {:.3f}] [val_top1_acc {:.3f}] [val_top5_acc {:.3f}] [ECE {:.3f}] [AURC {:.3f}] [EAURC {:.3f}] [correct/total {}/{}]'.format(
                    epoch,
                    val_losses.avg,
                    val_top1.avg,
                    val_top5.avg,
                    ece,
                    aurc,
                    eaurc,
                    correct,
                    total))
    return val_top1.avg
if __name__ == '__main__':
    main()
