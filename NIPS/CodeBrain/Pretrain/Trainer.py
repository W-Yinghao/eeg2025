import numpy as np
import torch
import math
import time
import json
import os
import datetime
from ptflops import get_model_complexity_info
from torch.nn import MSELoss
from torchinfo import summary
from tqdm import tqdm
from Utils.util import generate_mask
import Utils.util as utils
from Utils.util import NativeScalerWithGradNormCount as NativeScaler
import torch.nn as nn
from einops import rearrange
from Utils.util import generate_mask


class TFDual_Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.device = torch.device(f"cuda:{self.params.cuda}" if torch.cuda.is_available() else "cpu")
        self.data_loader = data_loader
        self.model = model.to(self.device)

        if self.params.parallel:
            print("Parallel")
            # self.device_ids = [0, 1, 2, 3, 4, 5, 6]
            self.device_ids = [3, 4, 5]
            self.model = torch.nn.DataParallel(self.model, device_ids = self.device_ids)
        else:
            self.device_ids = [0]

        self.data_length = len(self.data_loader)
        self.channel_list = list(range(20))
        if self.params.parallel:
            summary(self.model.module, input_size=(params.batch_size, 19, 30, 200),
                    input_chans = self.channel_list)
        else:
            summary(self.model, input_size=(1, 19, 30, 200), input_chans = self.channel_list)

        macs, params = get_model_complexity_info(self.model, (19, 30, 200), as_strings = True,
                                                 print_per_layer_stat = True, verbose = True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.params.lr,
                                           weight_decay = self.params.weight_decay)

        if self.params.lr_scheduler=='CosineAnnealingLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max = 40 * self.data_length, eta_min = 1e-5
            )

    def train_one_epoch(self, model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer,
                        device: torch.device,
                        epoch: int,
                        loss_scaler,
                        clip_grad: float = 0,
                        log_writer=None,
                        lr_scheduler=None,
                        start_steps=None,
                        lr_schedule_values=None,
                        args=None,
                        ):
        model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 10

        if hasattr(model.module, 'quantize'):
            try:
                model.module.quantize.reset_cluster_size(device)
                print("Reset the codebook statistic info in quantizer before each epoch")
            except:
                pass

        step_loader = 0
        for step, batch in enumerate(tqdm(metric_logger.log_every(self.data_loader, print_freq, header))):
            it = start_steps + step + step_loader
            if lr_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
            EEG = batch.float().to(device, non_blocking=True) / 100
            with torch.amp.autocast('cuda', enabled=True):
                loss, log_loss = model(EEG, input_chans = self.channel_list)
            loss_value = np.mean(loss.data.cpu().numpy())

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value), flush = True)

            optimizer.zero_grad()
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss.mean(), optimizer, clip_grad=clip_grad,
                                    parameters=model.parameters(), create_graph=is_second_order)
            loss_scale_value = loss_scaler.state_dict()["scale"]

            metric_logger.update(loss = loss_value)
            new_log_loss = {k.split('/')[-1]: v.mean() for k, v in log_loss.items()
                            if k.split('/')[-1] not in ['total_loss']}
            metric_logger.update(**new_log_loss)

            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)
            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]
            metric_logger.update(weight_decay=weight_decay_value)
            metric_logger.update(grad_norm=grad_norm)

            if log_writer is not None:
                log_writer.update(**new_log_loss, head = "train/loss")

                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(min_lr=min_lr, head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                log_writer.update(grad_norm=grad_norm, head="opt")
                log_writer.update(loss_scale=loss_scale_value, head="opt")

                log_writer.set_step()

            if lr_scheduler is not None:
                lr_scheduler.step_update(start_steps + step + step_loader)
        step_loader += step
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)

        if hasattr(model.module, 'quantize_t') or hasattr(model.module, 'quantize_f'):
            try:
                codebook_cluster_size_t = model.module.quantize_t._codebook.cluster_size
                codebook_cluster_size_f = model.module.quantize_f._codebook.cluster_size
            except:
                codebook_cluster_size_t = model.module.quantize_t.cluster_size
                codebook_cluster_size_f = model.module.quantize_f.cluster_size
            zero_cnt_t = (codebook_cluster_size_t == 0).sum().item()
            zero_cnt_f = (codebook_cluster_size_f == 0).sum().item()
            train_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            train_stat['Unused_code_t'] = zero_cnt_t
            train_stat['Unused_code_f'] = zero_cnt_f
            print(f"Unused code in temporal codebook: {zero_cnt_t}")
            print(f"Unused code in frqeuncy codebook: {zero_cnt_f}")
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(f"Unused code in temporal codebook: {zero_cnt_t} \n")
                f.write(f"Unused code in frqeuncy codebook: {zero_cnt_f} \n")
            return train_stat

        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


    def train_all(self):
        args = self.params
        if not args.eval:
            print("Model = %s" % str(self.model))
        for part in ['encoder', 'decoder_t', 'decoder_f']:
            model_part = eval(f"self.model.module.{part}")
            n_learnable_parameters = sum(p.numel() for p in model_part.parameters() if p.requires_grad)
            n_fix_parameters = sum(p.numel() for p in model_part.parameters() if not p.requires_grad)
            print(f'number of learnable params in model.{part}: {n_learnable_parameters / 1e6} M')
            print(f'number of fixed params in model.{part}: {n_fix_parameters / 1e6} M')

        n_learnable_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_fix_parameters = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        print(f'total number of learnable params: {n_learnable_parameters / 1e6} M')
        print(f'total number of fixed params in : {n_fix_parameters / 1e6} M')

        total_batch_size = args.batch_size * len(self.device_ids)
        args.lr = total_batch_size / 128 * args.lr
        print("LR = %.8f" % args.lr)
        print("Min LR = %.8f" % args.min_lr)
        print("Weigth Decay = %.8f" % args.weight_decay)
        print("Batch size = %d" % total_batch_size)
        num_training_steps_per_epoch = self.data_length // len(self.device_ids)
        print("Number of training steps = %d" % num_training_steps_per_epoch)
        print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

        loss_scaler = NativeScaler()

        log_writer = utils.TensorboardLogger(log_dir = args.log_dir)
        print("Use step level LR & WD scheduler!")
        lr_schedule_values = utils.cosine_scheduler(
            args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs = args.warmup_epochs, warmup_steps = args.warmup_steps,
        )

        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()

        for epoch in range(args.start_epoch, args.epochs):
            if log_writer is not None:
                log_writer.set_step(epoch * num_training_steps_per_epoch)
            train_stats = self.train_one_epoch(
                self.model,
                self.optimizer,
                self.device,
                epoch,
                loss_scaler,
                args.clip_grad,
                log_writer = log_writer,
                start_steps = epoch * num_training_steps_per_epoch,
                lr_schedule_values = lr_schedule_values,
                args = args
            )
            if args.output_dir:
                if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                    utils.save_model(
                        args=args, model=self.model, model_without_ddp=self.model.module, optimizer=self.optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, save_ckpt_freq=args.save_ckpt_freq)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch}

            if args.output_dir and utils.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))



class EEGSSM_Trainer(object):
    def __init__(self, params, data_loader, model, vqnsp: torch.nn.Module):
        self.params = params
        self.device = torch.device(f"cuda:{self.params.cuda}" if torch.cuda.is_available() else "cpu")
        self.data_loader = data_loader
        self.model = model.to(self.device)
        self.vqnsp = vqnsp.to(self.device)
        self.channel_list = list(range(20))
        self.criterion = MSELoss(reduction='mean').to(self.device)
        self.vqloss = nn.CrossEntropyLoss().to(self.device)

        if self.params.parallel:
            device_ids = [3, 4, 5]
            # device_ids = [6, 7]
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

        self.data_length = len(self.data_loader)
        summary(self.model, input_size=(1, 19, 30, 200))
        macs, params = get_model_complexity_info(self.model, (19, 30, 200), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                           weight_decay=self.params.weight_decay)
        if self.params.lr_scheduler=='CosineAnnealingLR':
            self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=40*self.data_length, eta_min=1e-5
            )


    def train(self):
        best_loss = 10000

        for epoch in range(self.params.epochs):
            losses = []
            for x in tqdm(self.data_loader, mininterval = 10):
                self.optimizer.zero_grad()
                x = x.to(self.device)/100
                if self.params.need_mask:
                    bz, ch_num, patch_num, patch_size = x.shape
                    mask = generate_mask(
                        bz, ch_num, patch_num, mask_ratio=self.params.mask_ratio, device=self.device,
                    )
                    y_t, y_f = self.model(x, mask = mask)
                    with torch.no_grad():
                        input_t_ids, input_f_ids = self.vqnsp.get_codebook_indices(x, self.channel_list)
                        mask = rearrange(mask, 'b s c -> b (s c)')
                        codes_t, codes_f = input_t_ids[mask == 1], input_f_ids[mask == 1]
                    loss = self.vqloss(y_t, codes_t) + self.vqloss(y_f, codes_f)
                else:
                    y_t, y_f = self.model(x)
                    with torch.no_grad():
                        input_t_ids, input_f_ids = self.vqnsp.get_codebook_indices(x, self.channel_list)
                    loss = self.vqloss(y_t, input_t_ids) + self.vqloss(y_f, input_f_ids)
                loss.backward()
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()
                losses.append(loss.data.cpu().numpy())
            mean_loss = np.mean(losses)
            learning_rate = self.optimizer.state_dict()['param_groups'][0]['lr']
            print(f'Epoch {epoch+1}: Training Loss: {mean_loss:.6f}, Learning Rate: {learning_rate:.6f}')
            if  mean_loss < best_loss:
                model_path = rf'{self.params.model_dir}/epoch{epoch+1}_loss{mean_loss}.pth'
                torch.save(self.model.state_dict(), model_path)
                print("model save in " + model_path)
                best_loss = mean_loss
