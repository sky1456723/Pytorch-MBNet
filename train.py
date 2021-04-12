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

from dataset import get_dataloader, get_dataset
from model import MBNet

writer = SummaryWriter()
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def valid(model, dataloader, save_dir, steps, prefix):
    model.eval()
    predict_mean_scores = []
    #all_bias_scores = []
    true_mean_scores = []
    #all_true_bias_scores = []
    for i, batch in enumerate(tqdm(dataloader, ncols=0, desc=prefix, unit=" step")):
        wavs, judge_ids, label_means, scores = batch
        wavs = wavs.to(device)
        wavs = wavs.unsqueeze(1)
        label_means = label_means.to(device)
        scores = scores.to(device)
        judge_ids = judge_ids.to(device).long()
        mean_scores, bias_scores = model(spectrum = wavs, 
                                         judge_id = judge_ids)
        mean_scores = mean_scores.squeeze(-1)
        mean_scores = torch.mean(mean_scores, dim = -1)
        predict_mean_scores.extend(mean_scores.cpu().tolist())
        true_mean_scores.extend(label_means.cpu().tolist())
    
    predict_mean_scores = np.array(predict_mean_scores)
    true_mean_scores = np.array(true_mean_scores)
    MSE = np.mean((true_mean_scores - predict_mean_scores) ** 2)
    writer.add_scalar(f"{prefix}/MSE", MSE, global_step=steps)
    LCC = np.corrcoef(true_mean_scores, predict_mean_scores)
    writer.add_scalar(f"{prefix}/LCC", LCC[0][1], global_step=steps)
    SRCC = scipy.stats.spearmanr(true_mean_scores.T, predict_mean_scores.T)
    writer.add_scalar(f"{prefix}/SRCC", SRCC[0], global_step=steps)
    print(
        f"\n[{prefix}][ MSE = {MSE:.4f} | LCC = {LCC[0][1]:.4f} | SRCC = {SRCC[0]:.4f} ]"
    )

    torch.save(model.state_dict(), os.path.join(save_dir, f"model-{steps}.pt"))

    model.train()

def clipped_mse(y_hat, label, tau = 0.5):
    mse = F.mse_loss(y_hat, label, reduction = 'none')
    threshold = torch.abs(y_hat - label)>tau
    mse = torch.mean(threshold*mse)
    return mse

def main(
    data_path,
    save_dir,
    total_steps,
    valid_steps,
    log_steps,
    update_freq,
):

    os.makedirs(save_dir, exist_ok=True)

    train_set = get_dataset(data_path, "training_data.csv", vcc18 = True, idtable = os.path.join(save_dir, 'idtable.pkl'))
    valid_set = get_dataset(data_path, "valid_data.csv", vcc18 = True, valid = True, idtable = os.path.join(save_dir, 'idtable.pkl'))
    test_set = get_dataset(data_path, "testing_data.csv", vcc18 = True, valid = True, idtable = os.path.join(save_dir, 'idtable.pkl'))
    train_loader = get_dataloader(train_set, batch_size=64, num_workers=1)
    valid_loader = get_dataloader(valid_set, batch_size=32, num_workers=1)
    test_loader = get_dataloader(test_set, batch_size=32, num_workers=1)

    model = MBNet(num_judges = 5000).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    criterion = clipped_mse

    pbar = tqdm(total=total_steps, ncols=0, desc="Overall", unit=" step")

    backward_steps = 0
    all_loss = []
    mean_losses = []
    bias_losses = []
    lamb = 4

    model.train()
    while pbar.n < pbar.total:
        for i, batch in enumerate(
            tqdm(train_loader, ncols=0, desc="Train", unit=" step")
        ):
            try:
                if pbar.n >= pbar.total:
                    break
                global_step = pbar.n + 1

                wavs, judge_ids, means, scores = batch
                wavs = wavs.to(device)
                wavs = wavs.unsqueeze(1)
                judge_ids = judge_ids.to(device)
                means = means.to(device)
                scores = scores.to(device)
                mean_scores, bias_scores = model(spectrum = wavs, 
                                                 judge_id = judge_ids)
                mean_scores = mean_scores.squeeze()
                bias_scores = bias_scores.squeeze()
                seq_len = mean_scores.shape[1]
                bsz = mean_scores.shape[0]
                means = means.unsqueeze(1).repeat(1, seq_len)
                scores = scores.unsqueeze(1).repeat(1, seq_len)
                mean_loss = criterion(mean_scores, means)
                bias_loss = criterion(bias_scores, scores)
                loss = mean_loss + lamb*bias_loss

                (loss / update_freq).backward()

                pbar.set_postfix(
                    {
                        "loss": loss.item(),
                        "mean_loss": mean_loss.item(),
                        "bias_loss": bias_loss.item(),
                    }
                )

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"[Runner] - CUDA out of memory at step {global_step}")
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                else:
                    raise

            all_loss.append(loss.item())
            mean_losses.append(mean_loss.item())
            bias_losses.append(bias_loss.item())
            del loss

            # whether to accumulate gradient
            backward_steps += 1
            if backward_steps % update_freq > 0:
                continue

            # gradient clipping
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

            # optimize
            if math.isnan(grad_norm):
                print(f"[Runner] - grad norm is NaN at step {global_step}")
            else:
                optimizer.step()
            optimizer.zero_grad()

            # adjust learning rate
            # scheduler.step()

            # logging
            if global_step % log_steps == 0:
                # print("\nlogging....")
                average_loss = torch.FloatTensor(all_loss).mean().item()
                mean_losses = torch.FloatTensor(mean_losses).mean().item()
                bias_losses = torch.FloatTensor(bias_losses).mean().item()
                writer.add_scalar(
                    f"Training/Loss", average_loss, global_step=global_step
                )
                writer.add_scalar(
                    f"Training/Mean Loss", mean_losses, global_step=global_step
                )
                writer.add_scalar(
                    f"Training/Bias Loss", bias_losses, global_step=global_step
                )
                all_loss = []
                mean_losses = []
                bias_losses = []

            # evaluate
            if global_step % valid_steps == 0:
                valid(model, valid_loader, save_dir, global_step, "Valid")
                valid(model, test_loader, save_dir, global_step, "Test")
            pbar.update(1)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--total_steps", type=int, default=50000)
    parser.add_argument("--valid_steps", type=int, default=1000)
    parser.add_argument("--log_steps", type=int, default=500)
    parser.add_argument("--update_freq", type=int, default=1)

    main(**vars(parser.parse_args()))
