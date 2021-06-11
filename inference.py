import argparse
import math
import os
import warnings

import numpy as np
import pandas as pd
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
    all_names = []
    total_loss = 0
    num_data = len(dataloader.dataset)
    for i, (wavs, wav_names) in enumerate(tqdm(dataloader)):
        wavs = wavs.to(device)
        wavs = wavs.unsqueeze(1)
        mean_scores = model.only_mean_inference(spectrum = wavs)
        all_mean_scores.extend(mean_scores.cpu().tolist())
        all_names.extend(wav_names)
    all_mean_scores = np.array(all_mean_scores)
    dirname = os.path.join(save_dir, f'{current_time}')
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    to_save = pd.DataFrame({'WAV_NAME':all_names, 'PREDICTION':all_mean_scores})
    to_save.to_csv(os.path.join(dirname, 'predict_mean.csv'))

def main(
    data_path,
    model_path,
    save_dir
):

    dataset = VCC16Dataset(data_path)
    dataloader = torch.utils.data.DataLoader(dataset, collate_fn = dataset.collate_fn, batch_size=20, num_workers=1, shuffle = False)
    model = MBNet(num_judges = 5000).to(device)
    model.load_state_dict(torch.load(model_path))
    valid(model, dataloader, save_dir)

    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type = str)
    parser.add_argument("--model_path", type = str)
    parser.add_argument("--save_dir", type = str)
    args = parser.parse_args()
    model_name = args.model_path.split('/')[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(**vars(args))
