import argparse
import random
import numpy as np
import torch
from timm.models import create_model
from torch.utils.data import DataLoader
from Datasets.pretraining_dataset import PretrainingDataset
import Models.modeling_tokenizer
from Pretrain.Trainer import TFDual_Trainer


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_args():
    parser = argparse.ArgumentParser(description = 'CodeBrain Pre-Training Stage 1: TFDual-Tokenizer')
    parser.add_argument('--seed', type = int, default = 42, help = 'random seed (default: 0)')
    parser.add_argument('--cuda', type = int, default = 0, help = 'cuda number (default: 1)')
    parser.add_argument('--parallel', type = bool, default = True, help = 'parallel (default: True)')
    parser.add_argument('--epochs', type = int, default = 20, help = 'number of epochs (default: 20)')
    parser.add_argument('--batch_size', type = int, default = 256, help='batch size for training (default: 256)')
    parser.add_argument('--lr', type = float, default = 1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--lr_scheduler', type = str, default = 'CosineAnnealingLR',
                        help='LR scheduler (default: CosineAnnealingLR)')
    parser.add_argument('--dropout', type = float, default = 0.1, help = 'dropout (default: 0.1)')
    parser.add_argument('--in_dim', type = int, default = 200, help = 'in_dim (default: 200)')
    parser.add_argument('--out_dim', type = int, default = 200, help='out_dim (default: 200)')
    parser.add_argument('--dim_feedforward', type = int, default = 800, help='dim_feedforward (default: 800)')
    parser.add_argument('--seq_len', type = int, default = 30, help='seq_len (default: 30)')
    parser.add_argument('--n_layer', type = int, default = 12, help='n_layer (default: 8)')
    parser.add_argument('--nhead', type = int, default = 8, help='nhead')
    parser.add_argument('--need_mask', type = bool, default = True, help='need_mask')
    parser.add_argument('--mask_ratio', type = float, default = 0.5, help='mask_ratio')

    # codebook parameters
    parser.add_argument('--model', default='tfdual_vq', type=str,
                        help='Name of model to train')
    parser.add_argument('--codebook_n_emd_t', default = 4096, type = int,
                        help='number of temporal codebook (default: 4096)')
    parser.add_argument('--codebook_n_emd_f', default = 4096, type=int,
                        help='number of frequency codebook (default: 4096)')
    parser.add_argument('--codebook_emd_dim', default = 32, type = int,
                        help='dimention of codebook (default: 32)')
    parser.add_argument('--ema_decay', default = 0.99, type = float,
                        help='ema decay for quantizer (default: 0.99)')
    parser.add_argument('--quantize_kmeans_init', action = 'store_true',
                        help = 'enable kmeans_init for quantizer')
    parser.add_argument('--input_size', default = 6000, type=int,
                        help='EEG input size for backbone (default: 6000)')
    parser.add_argument('--weight_decay', type=float, default = 1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6,
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--eval', action='store_true', default=False, help="Perform evaluation only")
    parser.add_argument('--warmup_epochs', type=int, default = 5,
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type = int, default = 1,
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='start epoch')
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--save_ckpt_freq', default = 5, type=int,
                        help='save interval (default: 5)')

    parser.add_argument('--dataset_dir', type = str, default = '',
                        help='dataset_dir')
    parser.add_argument('--output_dir', type = str, default = '',
                        help = 'model_dir')
    parser.add_argument('--log_dir', type=str, default='',
                        help='log_dir')
    return parser.parse_args()


def get_model(args):
    model = create_model(
        args.model,
        pretrained = False,
        as_tokenzer = False,
        n_code_t = args.codebook_n_emd_t,
        n_code_f = args.codebook_n_emd_f,
        code_dim = args.codebook_emd_dim,
        EEG_size = args.input_size,
        decay = args.ema_decay,
        quantize_kmeans_init = args.quantize_kmeans_init,
        batch_size = args.batch_size,
        device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    )
    return model

def main():
    args = get_args()
    print(args)
    setup_seed(args.seed)
    pretrained_dataset = PretrainingDataset(dataset_dir = args.dataset_dir)
    print(len(pretrained_dataset))
    data_loader = DataLoader(
        pretrained_dataset,
        batch_size = args.batch_size,
        num_workers = 8,
        shuffle = True,
    )

    model = get_model(args)
    trainer = TFDual_Trainer(args, data_loader, model)
    trainer.train_all()

    pretrained_dataset.db.close()


if __name__ == '__main__':
    main()