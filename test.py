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
#from transformers import Wav2Vec2Tokenizer

from dataset import get_dataloader, get_dataset
from model import MBNet
from utils import get_linear_schedule_with_warmup

def clipped_mse(y_hat, label, tau = 0.5):
    mse = F.mse_loss(y_hat, label, reduction = 'none')
    threshold = torch.abs(y_hat - label)>tau
    mse = torch.mean(threshold*mse)
    return mse

def valid(model, dataloader, steps, prefix, bias_weight):
    model.eval()
    all_mean_scores = []
    all_bias_scores = []
    all_true_mean_scores = []
    all_true_bias_scores = []
    total_loss = 0
    num_data = len(dataloader.dataset)
    for i, batch in enumerate(tqdm(dataloader, ncols=0, desc=prefix, unit=" step")):
        wavs, judge_ids, means, scores = batch
        wavs = wavs.to(device)
        wavs = wavs.unsqueeze(1)
        means = means.to(device)
        scores = scores.to(device)
        judge_ids = judge_ids.to(device).long()
        mean_scores, bias_scores = model(spectrum = wavs, 
                                         judge_id = judge_ids)
        mean_scores = mean_scores.squeeze(-1)
        bias_scores = bias_scores.squeeze(-1)
        bias_scores = torch.mean(bias_scores, dim = -1)
        mean_scores = torch.mean(mean_scores, dim = -1)
        all_mean_scores.extend(mean_scores.cpu().tolist())
        all_bias_scores.extend(bias_scores.cpu().tolist())
        all_true_mean_scores.extend(means.cpu().tolist())
        all_true_bias_scores.extend(scores.cpu().tolist())
    all_mean_scores = np.array(all_mean_scores)
    all_bias_scores = np.array(all_bias_scores)
    all_true_mean_scores = np.array(all_true_mean_scores)
    all_true_bias_scores = np.array(all_true_bias_scores)
    
    MSE = np.mean((all_true_mean_scores - all_mean_scores) ** 2)
    writer.add_scalar(f"{prefix}/MSE", MSE, global_step=steps)
    LCC = np.corrcoef(all_true_mean_scores, all_mean_scores)
    writer.add_scalar(f"{prefix}/LCC", LCC[0][1], global_step=steps)
    SRCC = scipy.stats.spearmanr(all_true_mean_scores.T, all_mean_scores.T)
    writer.add_scalar(f"{prefix}/SRCC", SRCC[0], global_step=steps)
    #writer.add_scalar(f"{prefix}/loss", total_loss/num_data)
    print(
        f"\n[{prefix}][ MSE = {MSE:.4f} | LCC = {LCC[0][1]:.4f} | SRCC = {SRCC[0]:.4f}" #| loss = {total_loss/num_data:.4f} ]"
    )
    np.save('predict_bias', all_bias_scores)
    np.save('predict_mean', all_mean_scores)
    #torch.save(model.state_dict(), os.path.join(save_dir, f"model-{steps}.pt"))

    #model.train()

def main(
    data_path,
    model_path,
    idtable_path,
    step,
    split
):


    if split == 'Valid':
        dataset = get_dataset(data_path, "valid_data.csv", vcc18 = True, valid = True, idtable = idtable_path)

    elif split == 'Test':
        dataset = get_dataset(data_path, "testing_data.csv", vcc18 = True, valid = True, idtable = idtable_path)

    dataloader = get_dataloader(dataset, batch_size=20, num_workers=1, shuffle = False)

    model = MBNet(num_judges = 5000).to(device)
    model.load_state_dict(torch.load(model_path))

    lamb = 4
    valid(model, dataloader, step, split, lamb)
    #valid(model, test_loader, save_dir, global_step, "Test", lamb)

    
if __name__ == "__main__":
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--data_path", type=str)
    parser.add_argument('-m', "--model_path", type=str)
    parser.add_argument('-i',"--idtable_path", type = str)
    parser.add_argument("--step", type = int)
    parser.add_argument("--split", type = str, default="Valid")
    args = parser.parse_args()
    model_name = args.model_path.split('/')[-1]
    split = args.split
    writer = SummaryWriter(log_dir = f'runs/{current_time}_{model_name}_{split}/')
    warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(**vars(args))
