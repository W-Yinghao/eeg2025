from tqdm import tqdm
import torch
from Downstream.finetune_evaluator import Evaluator
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from timeit import default_timer as timer
import numpy as np
import copy
import os

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader
        self.device = torch.device(f"cuda:{self.params.cuda}" if torch.cuda.is_available() else "cpu")

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        self.model = model.cuda()
        if self.params.downstream_dataset in ['FACED', 'SEED-V', 'ISRUC_S1', 'ISRUC_S3', 'BCIC2020-T3', 'TUEV']:
            self.criterion = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).cuda()
        elif self.params.downstream_dataset in ['MentalArithmetic', 'SHU-MI', 'CHB-MIT', 'TUAB']:
            self.criterion = BCEWithLogitsLoss().cuda()

        self.best_model_states = None

        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:

                backbone_params.append(param)

                if params.frozen:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                other_params.append(param)

        if self.params.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                               weight_decay=self.params.weight_decay)
        else:
            if self.params.multi_lr:
                self.optimizer = torch.optim.SGD([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ],  momentum=0.9, weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, momentum=0.9,
                                                 weight_decay=self.params.weight_decay)

        self.data_length = len(self.data_loader['train'])
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.params.epochs * self.data_length, eta_min=1e-6
        )
        self._use_wandb = HAS_WANDB and hasattr(params, 'wandb_project') and params.wandb_project and wandb.run is not None
        print(f"[Trainer] WandB logging enabled: {self._use_wandb}")
        if self._use_wandb:
            print(f"[Trainer] WandB run: {wandb.run.name} ({wandb.run.url})")
        print(self.model)

    def _wlog(self, metrics, step=None):
        if self._use_wandb:
            wandb.log(metrics, step=step, commit=True)

    def train_for_multiclass(self):
        f1_best = 0
        kappa_best = 0
        acc_best = 0
        cm_best = None
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)
                if self.params.downstream_dataset == 'ISRUC_S1' or self.params.downstream_dataset == 'ISRUC_S3':
                    loss = self.criterion(pred.transpose(1, 2), y)
                else:
                    loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, kappa, f1, cm = self.val_eval.get_metrics_for_multiclass(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        kappa,
                        f1,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                print(cm)
                self._wlog({
                    'train/loss': float(np.mean(losses)),
                    'val/acc': float(acc), 'val/kappa': float(kappa), 'val/f1': float(f1),
                    'lr': float(optim_state['param_groups'][0]['lr']),
                    'epoch': epoch + 1,
                }, step=epoch + 1)
                with open(self.params.file_name, "a") as file:
                    file.write(
                        "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}, "
                        "LR: {:.5f}, Time elapsed {:.2f} mins\n".format(
                            epoch + 1,
                            np.mean(losses),
                            acc,
                            kappa,
                            f1,
                            optim_state['param_groups'][0]['lr'],
                            (timer() - start_time) / 60
                        )
                    )
                    file.write(str(cm) + "\n")
                if kappa > kappa_best:
                    print("kappa increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                        acc,
                        kappa,
                        f1,
                    ))
                    with open(self.params.file_name, "a") as file:
                        file.write("kappa increasing....saving weights !! \n")
                        file.write("Val Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}\n".format(
                        acc,
                        kappa,
                        f1,
                    ))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    kappa_best = kappa
                    f1_best = f1
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

                    print("***************************Test************************")
                    with open(self.params.file_name, "a") as file:
                        file.write("***************************Test************************\n")
                    acc, kappa, f1, cm = self.test_eval.get_metrics_for_multiclass(self.model)
                    print("***************************Test results************************")
                    print(
                        "Test Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                            acc,
                            kappa,
                            f1,
                        )
                    )
                    print(cm)
                    with open(self.params.file_name, "a") as file:
                        file.write("***************************Test results************************\n")
                        file.write(
                            "Test Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}\n".format(
                                acc,
                                kappa,
                                f1,
                            )
                        )
                        file.write(str(cm) + "\n")
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            with open(self.params.file_name, "a") as file:
                file.write("***************************Test************************\n")
            acc, kappa, f1, cm = self.test_eval.get_metrics_for_multiclass(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                    acc,
                    kappa,
                    f1,
                )
            )
            print(cm)
            self._wlog({
                'final_test/acc': float(acc), 'final_test/kappa': float(kappa), 'final_test/f1': float(f1),
            })
            with open(self.params.file_name, "a") as file:
                file.write("***************************Test results************************\n")
                file.write(
                "Test Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}\n".format(
                    acc,
                    kappa,
                    f1,
                )
            )
                file.write(str(cm) + "\n")
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_kappa_{:.5f}_f1_{:.5f}.pth".format(best_f1_epoch, acc, kappa, f1)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)
            with open(self.params.file_name, "a") as file:
                file.write("model save in " + model_path)

    def train_for_binaryclass(self):
        acc_best = 0
        roc_auc_best = 0
        pr_auc_best = 0
        kappa_best = 0
        cm_best = None
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)
                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, kappa, pr_auc, roc_auc, cm = self.val_eval.get_metrics_for_binaryclass(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, kappa: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1, np.mean(losses), acc, kappa, pr_auc, roc_auc,
                        optim_state['param_groups'][0]['lr'], (timer() - start_time) / 60
                    )
                )
                print(cm)
                # Test eval every epoch
                test_acc, test_kappa, test_pr_auc, test_roc_auc, test_cm = self.test_eval.get_metrics_for_binaryclass(self.model)
                print("  Test: acc: {:.5f}, kappa: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(test_acc, test_kappa, test_pr_auc, test_roc_auc))
                log_dict = {
                    'train/loss': float(np.mean(losses)),
                    'val/acc': float(acc), 'val/kappa': float(kappa), 'val/pr_auc': float(pr_auc), 'val/roc_auc': float(roc_auc),
                    'test/acc': float(test_acc), 'test/kappa': float(test_kappa), 'test/pr_auc': float(test_pr_auc), 'test/roc_auc': float(test_roc_auc),
                    'lr': float(optim_state['param_groups'][0]['lr']),
                    'epoch': epoch + 1,
                }
                print(f"  [WandB] logging: {log_dict}")
                self._wlog(log_dict, step=epoch + 1)
                with open(self.params.file_name, "a") as file:
                    file.write(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, kappa: {:.5f}, pr_auc: {:.5f}, "
                    "roc_auc: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins \n".format(
                        epoch + 1, np.mean(losses), acc, kappa, pr_auc, roc_auc,
                        optim_state['param_groups'][0]['lr'], (timer() - start_time) / 60
                    )
                )
                    file.write(str(cm) + "\n")
                if roc_auc > roc_auc_best:
                    print("auroc increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, kappa: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(acc, kappa, pr_auc, roc_auc))
                    with open(self.params.file_name, "a") as file:
                        file.write("auroc increasing....saving weights !! \n")
                        file.write("Val Evaluation: acc: {:.5f}, kappa: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}\n".format(acc, kappa, pr_auc, roc_auc))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    kappa_best = kappa
                    pr_auc_best = pr_auc
                    roc_auc_best = roc_auc
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Final Test************************")
            with open(self.params.file_name, "a") as file:
                file.write("***************************Final Test************************\n")
            acc, kappa, pr_auc, roc_auc, cm = self.test_eval.get_metrics_for_binaryclass(self.model)
            print(
                "Test Evaluation: acc: {:.5f}, kappa: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(acc, kappa, pr_auc, roc_auc)
            )
            print(cm)
            self._wlog({
                'final_test/acc': float(acc), 'final_test/kappa': float(kappa),
                'final_test/pr_auc': float(pr_auc), 'final_test/roc_auc': float(roc_auc),
            })
            with open(self.params.file_name, "a") as file:
                file.write("***************************Final Test************************\n")
                file.write(
                    "Test Evaluation: acc: {:.5f}, kappa: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f} \n".format(acc, kappa, pr_auc, roc_auc)
                )
                file.write(str(cm) + "\n")
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_acc_{:.5f}_kappa_{:.5f}_roc_{:.5f}.pth".format(best_f1_epoch, acc, kappa, roc_auc)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)
            with open(self.params.file_name, "a") as file:
                file.write("model save in " + model_path)