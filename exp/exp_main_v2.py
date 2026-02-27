import math
import os
import time
import json
import warnings
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.cuda.amp import autocast, GradScaler

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import xPatch, GLPatch
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

warnings.filterwarnings("ignore")


class Exp_Main_v2(Exp_Basic):
    """
    v2 variant of Exp_Main:
      - keeps best checkpoint (does NOT delete checkpoint.pth)
      - saves a stable copy: best.pth
      - can save predictions + ground truth for train/val/test as NPZ:
          {pred_save_root}/{setting}/{split}.npz  with arrays: preds, trues
        and metrics.json alongside
    """
    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        model_dict = {
            "xPatch": xPatch,
            "GLPatch": GLPatch,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        mse_criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()
        return mse_criterion, mae_criterion

    def vali(self, vali_data, vali_loader, criterion, is_test: bool = True):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input (kept for compatibility; not used by current model forward)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # forward (AMP for inference too)
                if self.args.use_amp:
                    with autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                if not is_test:
                    # Arctangent loss with weight decay (original repo behavior)
                    ratio = np.array([-1 * math.atan(i + 1) + math.pi / 4 + 1 for i in range(self.args.pred_len)])
                    ratio = torch.tensor(ratio).unsqueeze(-1).to(self.device)
                    pred = outputs * ratio
                    true = batch_y * ratio
                else:
                    pred = outputs
                    true = batch_y

                loss = criterion(pred, true)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    # ----------------------------
    # Prediction collection / saving
    # ----------------------------
    def _collect_preds_trues(self, data_loader) -> Tuple[np.ndarray, np.ndarray]:
        preds, trues = [], []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.use_amp:
                    with autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:]

                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        self.model.train()
        return preds, trues

    def predict_and_save(self, setting: str, split: str, save_root: str = "./predictions"):
        """
        Saves BOTH:
        - scaled arrays (what the model/dataset currently uses)
        - original-scale arrays (inverse StandardScaler)

        Output file:
        {save_root}/{setting}/{split}.npz

        Keys inside npz:
        preds, trues           -> scaled
        preds_orig, trues_orig -> original scale (inverse transformed)

        metrics.json will contain both scaled and original metrics.
        """
        data_set, data_loader = self._get_data(flag=split)
        preds, trues = self._collect_preds_trues(data_loader)  # these are SCALED in your current pipeline

        # ---------- inverse transform helper ----------
        def inverse_to_original(x_scaled: np.ndarray) -> np.ndarray:
            """
            x_scaled: [N, pred_len, C_sel]
            returns:  [N, pred_len, C_sel] in original units
            """
            if not hasattr(data_set, "scaler") or data_set.scaler is None:
                # No scaler available -> cannot invert
                return x_scaled

            scaler = data_set.scaler

            # If dataset provides inverse_transform (common in Informer-style datasets), prefer it.
            if hasattr(data_set, "inverse_transform"):
                try:
                    x2 = x_scaled.reshape(-1, x_scaled.shape[-1])
                    x2_inv = data_set.inverse_transform(x2)
                    return x2_inv.reshape(x_scaled.shape)
                except Exception:
                    pass

            # Fallback: manual inverse using scaler.mean_ and scaler.scale_
            if not (hasattr(scaler, "mean_") and hasattr(scaler, "scale_")):
                return x_scaled

            mean = np.asarray(scaler.mean_, dtype=np.float32)
            std = np.asarray(scaler.scale_, dtype=np.float32)

            C_sel = x_scaled.shape[-1]

            # Case A: scaler matches selected channel count
            if mean.shape[0] == C_sel:
                mu = mean
                sig = std

            # Case B: we only saved 1 channel, scaler has many -> assume target is last
            elif C_sel == 1 and mean.shape[0] > 1:
                target_idx = -1
                # Try to locate target column if dataset exposes cols/target
                if hasattr(data_set, "cols") and hasattr(self.args, "target"):
                    try:
                        target_idx = list(data_set.cols).index(self.args.target)
                    except Exception:
                        target_idx = -1
                mu = mean[target_idx:target_idx + 1]
                sig = std[target_idx:target_idx + 1]
            else:
                # Unknown mapping -> return scaled as-is rather than wrong inverse
                return x_scaled

            return x_scaled * sig.reshape(1, 1, -1) + mu.reshape(1, 1, -1)

        preds_orig = inverse_to_original(preds)
        trues_orig = inverse_to_original(trues)

        save_dir = os.path.join(save_root, setting)
        os.makedirs(save_dir, exist_ok=True)

        np.savez_compressed(
            os.path.join(save_dir, f"{split}.npz"),
            # Persist ONLY original scale arrays to keep artifacts small.
            preds=preds_orig,
            trues=trues_orig,
        )

        # metrics on scaled
        mae_s, mse_s = metric(preds, trues)
        # metrics on original
        mae_o, mse_o = metric(preds_orig, trues_orig)

        metrics_path = os.path.join(save_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        else:
            metrics = {}

        metrics[split] = {
            "scaled": {"mse": float(mse_s), "mae": float(mae_s)},
            "original": {"mse": float(mse_o), "mae": float(mae_o)},
        }

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(
            f"[predict_and_save] {setting} | {split}: "
            f"scaled(mse={mse_s:.6f}, mae={mae_s:.6f}) "
            f"orig(mse={mse_o:.6f}, mae={mae_o:.6f}) -> {save_dir}"
        )

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        mse_criterion, mae_criterion = self._select_criterion()

        scaler = GradScaler() if self.args.use_amp else None

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input (kept for compatibility; not used by current model forward)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with autocast():
                        outputs = self.model(batch_x)
                        f_dim = -1 if self.args.features == "MS" else 0
                        outputs = outputs[:, -self.args.pred_len :, f_dim:]
                        batch_y_cut = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                        ratio = np.array([-1 * math.atan(j + 1) + math.pi / 4 + 1 for j in range(self.args.pred_len)])
                        ratio = torch.tensor(ratio).unsqueeze(-1).to(self.device)
                        outputs = outputs * ratio
                        batch_y_cut = batch_y_cut * ratio

                        loss = mae_criterion(outputs, batch_y_cut)

                    train_loss.append(loss.item())

                    if (i + 1) % 100 == 0:
                        print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                        iter_count = 0
                        time_now = time.time()

                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    outputs = self.model(batch_x)
                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    batch_y_cut = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                    ratio = np.array([-1 * math.atan(j + 1) + math.pi / 4 + 1 for j in range(self.args.pred_len)])
                    ratio = torch.tensor(ratio).unsqueeze(-1).to(self.device)
                    outputs = outputs * ratio
                    batch_y_cut = batch_y_cut * ratio

                    loss = mae_criterion(outputs, batch_y_cut)
                    train_loss.append(loss.item())

                    if (i + 1) % 100 == 0:
                        print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                        iter_count = 0
                        time_now = time.time()

                    loss.backward()
                    model_optim.step()

            print(f"Epoch: {epoch+1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)

            vali_loss = self.vali(vali_data, vali_loader, mae_criterion, is_test=False)
            test_loss = self.vali(test_data, test_loader, mse_criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, "checkpoint.pth")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        # Keep checkpoint + write stable copy name
        torch.save(self.model.state_dict(), os.path.join(path, "best.pth"))

        return self.model

    def test(self, setting, test: int = 0):
        test_data, test_loader = self._get_data(flag="test")

        if test:
            print("loading model")
            ckpt_dir = os.path.join("./checkpoints", setting)
            best_path = os.path.join(ckpt_dir, "best.pth")
            ckpt_path = os.path.join(ckpt_dir, "checkpoint.pth")
            load_path = best_path if os.path.exists(best_path) else ckpt_path
            self.model.load_state_dict(torch.load(load_path, map_location=self.device))

        preds = []
        trues = []

        folder_path = os.path.join("./test_results", setting)
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input (kept for compatibility; not used by current model forward)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y_cut = batch_y[:, -self.args.pred_len :, f_dim:]

                outputs_np = outputs.detach().cpu().numpy()
                batch_y_np = batch_y_cut.detach().cpu().numpy()

                preds.append(outputs_np)
                trues.append(batch_y_np)

                if i % 20 == 0:
                    input_np = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input_np[0, :, -1], batch_y_np[0, :, -1]), axis=0)
                    pd = np.concatenate((input_np[0, :, -1], outputs_np[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))

        preds = np.array(preds)
        trues = np.array(trues)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse = metric(preds, trues)
        print(f"mse:{mse}, mae:{mae}")

        with open("result.txt", "a") as f:
            f.write(setting + " \n")
            f.write(f"mse:{mse}, mae:{mae}")
            f.write("\n\n")

        return
