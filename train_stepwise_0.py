import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.vit import interpolate_pos_embed
from transformers import BertTokenizerFast

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import logging
from types import MethodType
from tools.env import init_dist
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from models import box_ops
from tools.multilabel_metrics import AveragePrecisionMeter, get_multi_label
from models.HAMMER import HAMMER


def setlogger(log_file):
    filehandler = logging.FileHandler(log_file)
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    def epochInfo(self, set, idx, loss, acc):
        self.info('{set}-{idx:d} epoch | loss:{loss:.8f} | auc:{acc:.4f}%'.format(
            set=set,
            idx=idx,
            loss=loss,
            acc=acc
        ))

    logger.epochInfo = MethodType(epochInfo, logger)

    return logger


def text_input_adjust(text_input, fake_word_pos, device):
    input_ids_remove_SEP = [x[:-1] for x in text_input.input_ids]
    maxlen = max([len(x) for x in text_input.input_ids])-1
    input_ids_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in input_ids_remove_SEP]
    text_input.input_ids = torch.LongTensor(input_ids_remove_SEP_pad).to(device)

    attention_mask_remove_SEP = [x[:-1] for x in text_input.attention_mask]
    attention_mask_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in attention_mask_remove_SEP]
    text_input.attention_mask = torch.LongTensor(attention_mask_remove_SEP_pad).to(device)

    fake_token_pos_batch = []
    for i in range(len(fake_word_pos)):
        fake_token_pos = []
        fake_word_pos_decimal = np.where(fake_word_pos[i].numpy() == 1)[0].tolist()
        subword_idx = text_input.word_ids(i)
        subword_idx_rm_CLSSEP = subword_idx[1:-1]
        subword_idx_rm_CLSSEP_array = np.array(subword_idx_rm_CLSSEP)
        for i in fake_word_pos_decimal:
            fake_token_pos.extend(np.where(subword_idx_rm_CLSSEP_array == i)[0].tolist())
        fake_token_pos_batch.append(fake_token_pos)

    return text_input, fake_token_pos_batch


def train(args, model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, summary_writer):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_MAC', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_BIC', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_bbox', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_giou', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_TMG', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_MLC', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 100
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    global_step = epoch * len(data_loader)

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (image, label, text, fake_image_box, fake_word_pos, W, H) in enumerate(metric_logger.log_every(args, data_loader, print_freq, header)):

        if config['schedular']['sched'] == 'cosine_in_step':
            scheduler.adjust_learning_rate(optimizer, i / len(data_loader) + epoch, args, config)

        optimizer.zero_grad()

        image = image.to(device, non_blocking=True)
        text_input = tokenizer(text, max_length=128, truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False)
        text_input, fake_token_pos = text_input_adjust(text_input, fake_word_pos, device)

        alpha = config['alpha'] if epoch > 0 else config['alpha'] * min(1, i / len(data_loader))

        loss_MAC, loss_BIC, loss_bbox, loss_giou, loss_TMG, loss_MLC = model(image, label, text_input, fake_image_box, fake_token_pos, alpha=alpha)

        loss = config['loss_MAC_wgt']*loss_MAC \
             + config['loss_BIC_wgt']*loss_BIC \
             + config['loss_bbox_wgt']*loss_bbox \
             + config['loss_giou_wgt']*loss_giou \
             + config['loss_TMG_wgt']*loss_TMG \
             + config['loss_MLC_wgt']*loss_MLC

        loss.backward()
        optimizer.step()

        metric_logger.update(loss_MAC=loss_MAC.item())
        metric_logger.update(loss_BIC=loss_BIC.item())
        metric_logger.update(loss_bbox=loss_bbox.item())
        metric_logger.update(loss_giou=loss_giou.item())
        metric_logger.update(loss_TMG=loss_TMG.item())
        metric_logger.update(loss_MLC=loss_MLC.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch==0 and i%step_size==0 and i<=warmup_iterations and config['schedular']['sched'] != 'cosine_in_step':
            scheduler.step(i//step_size)

        global_step += 1

        if args.log:
            lossinfo = {
                'lr': optimizer.param_groups[0]["lr"],
                'loss_MAC': loss_MAC.item(),
                'loss_BIC': loss_BIC.item(),
                'loss_bbox': loss_bbox.item(),
                'loss_giou': loss_giou.item(),
                'loss_TMG': loss_TMG.item(),
                'loss_MLC': loss_MLC.item(),
                'loss': loss.item(),
            }
            for tag, value in lossinfo.items():
                summary_writer.add_scalar(tag, value, global_step)

    metric_logger.synchronize_between_processes()
    if args.log:
        print("Averaged stats:", metric_logger.global_avg(), flush=True)
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(args, model, data_loader, tokenizer, device, config):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    y_true, y_pred = [], []
    cls_nums_all, cls_acc_all = 0, 0
    TP_all, TN_all, FP_all, FN_all = 0, 0, 0, 0

    multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
    multi_label_meter.reset()

    for i, (image, label, text, fake_image_box, fake_word_pos, W, H) in enumerate(metric_logger.log_every(args, data_loader, 200, header)):

        image = image.to(device, non_blocking=True)
        text_input = tokenizer(text, max_length=128, truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False)
        text_input, fake_token_pos = text_input_adjust(text_input, fake_word_pos, device)

        logits_real_fake, logits_multicls, output_coord, logits_tok = model(image, label, text_input, fake_image_box, fake_token_pos, is_train=False)

        cls_label = torch.ones(len(label), dtype=torch.long).to(image.device)
        real_label_pos = np.where(np.array(label) == 'orig')[0].tolist()
        cls_label[real_label_pos] = 0

        y_pred.extend(F.softmax(logits_real_fake, dim=1)[:,1].cpu().flatten().tolist())
        y_true.extend(cls_label.cpu().flatten().tolist())

        pred_acc = logits_real_fake.argmax(1)
        cls_nums_all += cls_label.shape[0]
        cls_acc_all += torch.sum(pred_acc == cls_label).item()

        target, _ = get_multi_label(label, image)
        multi_label_meter.add(logits_multicls, target)

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    AUC_cls = roc_auc_score(y_true, y_pred)
    ACC_cls = cls_acc_all / cls_nums_all
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    EER_cls = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    return AUC_cls, ACC_cls, EER_cls


import matplotlib.pyplot as plt

def main_worker(gpu, args, config):
    if gpu is not None:
        args.gpu = gpu

    init_dist(args)

    log_dir = os.path.join(args.output_dir, 'log'+ args.log_num)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'shell.txt')
    logger = setlogger(log_file)
    yaml.dump(config, open(os.path.join(log_dir, 'config.yaml'), 'w'))

    summary_writer = SummaryWriter(log_dir) if args.log else None

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best, best_epoch = 0, 0

    train_dataset, val_dataset = create_dataset(config)
    tokenizer = BertTokenizerFast.from_pretrained(args.text_encoder)

    model = HAMMER(args=args, config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True)
    model = model.to(device)

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)
    if config['schedular']['sched'] == 'cosine_in_step':
        args.lr = config['optimizer']['lr']

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # --- Optional step-wise training ---
    if args.step_size is not None:
        step_size = args.step_size
        num_samples = len(train_dataset)
        step_indices = list(range(step_size, num_samples + step_size, step_size))
        if step_indices[-1] > num_samples:
            step_indices[-1] = num_samples
    else:
        step_indices = [len(train_dataset)]

    # Liste per salvare le metriche ad ogni step
    auc_steps = []
    acc_steps = []
    eer_steps = []

    # Pulizia file di log metriche per scrittura append
    metrics_log_path = os.path.join(log_dir, 'metrics_per_step.log')
    with open(metrics_log_path, 'w') as f:
        f.write("Metriche AUC, Accuracy ed EER per ogni step di training\n")

    for step_idx in range(len(step_indices)):
        start_idx = 0
        end_idx = step_indices[step_idx]
        if args.step_size is not None:
            print(f"Training on samples [{start_idx}:{end_idx}] ({end_idx-start_idx})")

        train_subset = torch.utils.data.Subset(train_dataset, range(start_idx, end_idx))
        if args.distributed:
            train_sampler = create_sampler([train_subset], [True], args.world_size, args.rank)[0]
        else:
            train_sampler = None

        train_loader, val_loader = create_loader([train_subset, val_dataset],
                                                 [train_sampler, None],
                                                 batch_size=[config['batch_size_train'], config['batch_size_val']],
                                                 num_workers=[4, 4],
                                                 is_trains=[True, False],
                                                 collate_fns=[None, None])

        for epoch in range(start_epoch, max_epoch):
            train_stats = train(args, model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config, summary_writer)
            AUC_cls, ACC_cls, EER_cls = evaluation(args, model_without_ddp, val_loader, tokenizer, device, config)

            print(f"Step {step_idx}, Epoch {epoch}: AUC={AUC_cls:.4f}, ACC={ACC_cls:.4f}, EER={EER_cls:.4f}")

            if args.log and args.step_size is not None:
                lossinfo = {
                    'AUC_cls': round(AUC_cls*100, 4),
                    'ACC_cls': round(ACC_cls*100, 4),
                    'EER_cls': round(EER_cls*100, 4),
                }
                for tag, value in lossinfo.items():
                    summary_writer.add_scalar(f'step_{step_idx}_'+tag, value, epoch)

        # Salviamo le metriche al termine dello step (dopo tutti gli epoch su quel sottoinsieme)
        auc_steps.append(AUC_cls)
        acc_steps.append(ACC_cls)
        eer_steps.append(EER_cls)

        # Salvataggio metriche su file di log
        with open(metrics_log_path, 'a') as f:
            f.write(f"Step {step_idx}: AUC={AUC_cls:.4f}, ACC={ACC_cls:.4f}, EER={EER_cls:.4f}\n")

        start_epoch = 0

    # Dopo tutti gli step, plottiamo l’andamento delle metriche
    plt.figure(figsize=(12, 6))
    steps = range(1, len(auc_steps)+1)
    plt.plot(steps, auc_steps, marker='o', label='AUC')
    plt.plot(steps, acc_steps, marker='o', label='Accuracy')
    plt.plot(steps, eer_steps, marker='o', label='EER')
    plt.xlabel('Step di training (numero di sottoinsiemi dati)')
    plt.ylabel('Valore metrica')
    plt.title('Andamento delle prestazioni ad ogni step di training')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'metrics_step_plot.png'))
    plt.show()

    if utils.is_main_process():
        torch.save(model_without_ddp.state_dict(), os.path.join(log_dir, 'final_model.pth'))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--rank', default=-1, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23459', type=str)
    parser.add_argument('--dist-backend', default='gloo', type=str)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--log_num', '-l', type=str)
    parser.add_argument('--model_save_epoch', type=int, default=20)
    parser.add_argument('--token_momentum', default=False, action='store_true')

    # --- Optional step size parameter ---
    parser.add_argument('--step_size', default=None, type=int,
                        help='Optional: number of samples per training step (e.g., 10000)')

    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    if args.launcher == 'none':
        args.launcher = 'pytorch'
        main_worker(0, args, config)
    else:
        ngpus_per_node = torch.cuda.device_count()
        args.ngpus_per_node = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args, config))
