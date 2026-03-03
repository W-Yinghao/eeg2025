import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from timm.models import create_model
import Models.modeling_tokenizer
from Datasets.pretraining_dataset import PretrainingDataset
from Models.SSSM import SSSM
from Pretrain.Trainer import EEGSSM_Trainer
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_visual_tokenizer(args):
    print(f"Creating visual tokenizer: {args.tokenizer_model}")
    model = create_model(
            args.tokenizer_model,
            pretrained = True,
            pretrained_weight = args.tokenizer_weight,
            as_tokenzer = True,
            n_code_t = args.codebook_size_t,
            n_code_f = args.codebook_size_f,
            code_dim = args.codebook_dim,
        ).eval()
    return model

def main():
    parser = argparse.ArgumentParser(description='CodeBrain Pre-Training Stage 2: EEGSSM')
    parser.add_argument('--seed', type = int, default = 42, help = 'random seed (default: 0)')
    parser.add_argument('--cuda', type = int, default = 6, help='cuda number (default: 6)')
    parser.add_argument('--parallel', type = bool, default = True, help='parallel (default: True)')
    parser.add_argument('--epochs', type=int, default = 10,
                        help='number of epochs (default: 10)')
    parser.add_argument('--batch_size', type = int, default = 256,
                        help='batch size for training (default: 256)')
    parser.add_argument('--lr', type = float, default = 1e-4,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type = float, default = 5e-3,
                        help='weight_decay (default: 5e-3)')
    parser.add_argument('--clip_value', type = float, default = 5,
                        help='clip_value (default: 5)')
    parser.add_argument('--lr_scheduler', type = str, default='CosineAnnealingLR',
                        help='lr_scheduler (default: CosineAnnealingLR)')

    parser.add_argument('--dropout', type = float, default = 0.1, help='dropout (default: 0.1)')
    parser.add_argument('--in_dim', type = int, default = 200, help='in_dim (default: 200)')
    parser.add_argument('--out_dim', type = int, default = 200, help='out_dim (default: 200)')
    parser.add_argument('--d_model', type = int, default = 200, help = 'd_model (default: 200)')
    parser.add_argument('--dim_feedforward', type = int, default = 800,
                        help = 'dim_feedforward (default: 800)')
    parser.add_argument('--seq_len', type = int, default = 570, help = 'seq_len (default: 570)')
    parser.add_argument('--n_layer', type = int, default = 8, help = 'n_layer (default: 8)')
    parser.add_argument('--nhead', type = int, default = 8, help = 'nhead (default: 8)')
    parser.add_argument('--need_mask', type = bool, default = True, help = 'need_mask (default: True)')
    parser.add_argument('--mask_ratio', type = float, default = 0.5, help = 'mask_ratio (default: 0.5)')

    parser.add_argument('--dataset_dir', type=str,
                        default='',
                        help='dataset_dir')
    parser.add_argument('--model_dir', type=str,
                        default='',
                        help = 'model_dir')

    parser.add_argument("--tokenizer_weight", type = str,
                        default = "")
    parser.add_argument("--tokenizer_model", type = str,
                        default = "tfdual_vq")
    parser.add_argument('--codebook_size_t', default = 4096, type=int,
                        help='number of temporal codebook (default: 4096)')
    parser.add_argument('--codebook_size_f', default = 4096, type=int,
                        help='number of frequency codebook (default: 4096)')
    parser.add_argument('--codebook_dim', default = 32, type=int,
                        help='dimention of codebook (default: 32)')

    params = parser.parse_args()
    print(params)
    setup_seed(params.seed)
    pretrained_dataset = PretrainingDataset(dataset_dir=params.dataset_dir)
    print(len(pretrained_dataset))
    data_loader = DataLoader(
        pretrained_dataset,
        batch_size=params.batch_size,
        num_workers=0,
        shuffle=True,
    )

    model = SSSM(
        in_channels=params.in_dim, res_channels=params.d_model,
        skip_channels=params.d_model, out_channels=params.out_dim,
        num_res_layers=params.n_layer,
        diffusion_step_embed_dim_in=params.d_model,
        diffusion_step_embed_dim_mid =params.d_model,
        diffusion_step_embed_dim_out = params.d_model,
        s4_lmax=params.seq_len,
        s4_d_state=64,
        s4_dropout = 0.1,
        s4_bidirectional=True,
        s4_layernorm=True,
        codebook_size_t = params.codebook_size_t,
        codebook_size_f = params.codebook_size_f,
        if_codebook = True)

    tfdual_tokenizer = get_visual_tokenizer(params)
    trainer = EEGSSM_Trainer(params, data_loader, model, tfdual_tokenizer)
    trainer.train()
    pretrained_dataset.db.close()


if __name__ == '__main__':
    main()
