import argparse
import math
import os
import warnings

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import VCC16Dataset
from model import MBNet

def clipped_mse(y_hat, label, tau = 0.5):
    mse = F.mse_loss(y_hat, label, reduction = 'none')
    threshold = torch.abs(y_hat - label)>tau
    mse = torch.mean(threshold*mse)
    return mse

def valid(model, dataloader, save_dir):
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    model.eval()
    all_mean_scores = []
    all_bias_scores = []
    all_true_mean_scores = []
    all_true_bias_scores = []
    total_loss = 0
    num_data = len(dataloader.dataset)
    for i, batch in enumerate(tqdm(dataloader, ncols=0, unit=" step")):
        wavs = batch
        wavs = wavs.to(device)
        wavs = wavs.unsqueeze(1)
        mean_scores = model.only_mean_inference(spectrum = wavs)
        #mean_scores = mean_scores.squeeze(-1)
        #bias_scores = bias_scores.squeeze(-1)
        #seq_len = mean_scores.shape[1]
        #bsz = mean_scores.shape[0]
        #means_loss = means.unsqueeze(1).repeat(1, seq_len)
        #bias_loss = scores.unsqueeze(1).repeat(1, seq_len)
        #mean_loss = clipped_mse(mean_scores, means_loss)
        #bias_loss = clipped_mse(bias_scores, bias_loss)
        #loss = mean_loss + bias_weight*bias_loss
        #total_loss += (loss*bsz).item()

        #bias_scores = torch.mean(bias_scores, dim = -1)
        #mean_scores = torch.mean(mean_scores, dim = -1)
        #print(mean_scores)
        #print(means)
        all_mean_scores.extend(mean_scores.cpu().tolist())
        #all_bias_scores.extend(bias_scores.cpu().tolist())
        #all_true_mean_scores.extend(means.cpu().tolist())
        #all_true_bias_scores.extend(scores.cpu().tolist())
    all_mean_scores = np.array(all_mean_scores)
    #all_bias_scores = np.array(all_bias_scores)
    #all_true_mean_scores = np.array(all_true_mean_scores)
    #all_true_bias_scores = np.array(all_true_bias_scores)
    
    #MSE = np.mean((all_true_bias_scores - all_bias_scores) ** 2)
    #writer.add_scalar(f"{prefix}/MSE", MSE, global_step=steps)
    #LCC = np.corrcoef(all_true_bias_scores, all_bias_scores)
    #writer.add_scalar(f"{prefix}/LCC", LCC[0][1], global_step=steps)
    #SRCC = scipy.stats.spearmanr(all_true_bias_scores.T, all_bias_scores.T)
    #writer.add_scalar(f"{prefix}/SRCC", SRCC[0], global_step=steps)
    #writer.add_scalar(f"{prefix}/loss", total_loss/num_data)
    #print(
    #    f"\n[{prefix}][ MSE = {MSE:.4f} | LCC = {LCC[0][1]:.4f} | SRCC = {SRCC[0]:.4f}" #| loss = {total_loss/num_data:.4f} ]"
    #)
    #np.save('predict_bias', all_bias_scores)
    dirname = os.path.join(save_dir, f'{current_time}')
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    np.save(os.path.join(dirname, 'predict_mean'), all_mean_scores)

def main(
    data_path,
    model_path,
    save_dir
):

    dataset = VCC16Dataset(data_path)
    # training_set, valid_set, test_set = random_split(dataset, [13580, 3000, 4000])
    dataloader = torch.utils.data.DataLoader(dataset, collate_fn = dataset.collate_fn, batch_size=20, num_workers=1, shuffle = False)

    model = MBNet(num_judges = 5000).to(device)
    model.load_state_dict(torch.load(model_path))

    valid(model, dataloader, save_dir)
    #valid(model, test_loader, save_dir, global_step, "Test", lamb)

    
if __name__ == "__main__":
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type = str)
    parser.add_argument("--model_path", type = str)
    parser.add_argument("--save_dir", type = str)
    #parser.add_argument("--split", type = str, default="Valid")
    args = parser.parse_args()
    model_name = args.model_path.split('/')[-1]
    #split = args.split
    #writer = SummaryWriter(log_dir = f'runs/VCC2016/{current_time}_{model_name}_{split}/')
    #warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(**vars(args))
