# client.py
import torch
from tqdm import tqdm
import sys
import logging
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from utils import save_visual_results, CharbonnierLoss, EdgeAwareLoss
import config
from models import VQVAE2
import lpips
import piq
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
import torch.nn as nn


class Client:
    def __init__(self, client_id, train_loader, test_loader, val_loader, results_logger, device=None):
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.results_logger = results_logger
        self.logger = logging.getLogger()
        self.device = device if device is not None else config.DEVICE

        # --- Model & Optimizer ---
        self.model = VQVAE2().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)

        # --- Loss functions ---
        self.recon_loss_fn = CharbonnierLoss(eps=config.LOSS_CONFIG["charbonnier_eps"]).to(self.device)
        self.perceptual_loss_fn = lpips.LPIPS(net='vgg').to(self.device)
        self.structural_loss_fn = piq.MultiScaleSSIMLoss(data_range=1.0).to(self.device)
        self.edge_loss_fn = EdgeAwareLoss().to(self.device)

        # --- Metrics ---
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips_metric = lpips.LPIPS(net='vgg').to(self.device)
        self.mse_metric = nn.MSELoss()
        self.fid_metric = FrechetInceptionDistance(feature=2048).to(self.device)

        # --- Scheduler ---
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=config.LOCAL_EPOCHS * len(self.train_loader),  # Restart every round
            eta_min=1e-6
        )

    def get_codebook_vitality(self):
        return self.model.get_codebook_vitality()


    def get_shared_params(self):
        return self.model.get_shared_state_dict()

    def set_shared_params(self, global_params):
        self.model.load_shared_state_dict(global_params)

    #  NOVEL METHOD
    def get_codebook_vectors(self):
        """Get codebook vectors for similarity-based aggregation"""
        return self.model.get_codebook_vectors()
    

    def _to_lpips(self, x):
        """Convert 1-channel images to 3-channel [-1,1] for LPIPS."""
        x = torch.clamp(x, -1, 1) 
        if x.shape[1] == 1:         
            x = x.repeat(1, 3, 1, 1)
        return x

    def train_personalized(self, epochs=1, lr_factor=0.1):
        """Fine-tune decoder locally with full loss suite."""
        self.model.train()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.LEARNING_RATE * lr_factor
        )

        for epoch in range(epochs):
            for batch in self.train_loader:
                if batch is None:
                    continue
                lr_images = batch['lr'].to(config.DEVICE)
                hr_images = batch['hr'].to(config.DEVICE)

                optimizer.zero_grad()
                reconstructed, vq_loss = self.model(lr_images)

                rec_01 = torch.clamp((reconstructed + 1) / 2, 0.0, 1.0)
                hr_01 = torch.clamp((hr_images + 1) / 2, 0.0, 1.0)

                recon_loss = self.recon_loss_fn(reconstructed, hr_images)
                x_lpips = self._to_lpips(reconstructed)
                y_lpips = self._to_lpips(hr_images)
                perceptual_loss = self.perceptual_loss_fn(x_lpips, y_lpips).mean()
                structural_loss = torch.clamp(self.structural_loss_fn(rec_01, hr_01), min=0.0)
                edge_loss = self.edge_loss_fn(reconstructed, hr_images)

                loss = (config.LOSS_CONFIG["reconstruction_loss_weight"] * recon_loss +
                        config.LOSS_CONFIG["perceptual_loss_weight"] * perceptual_loss +
                        config.LOSS_CONFIG["structural_loss_weight"] * structural_loss +
                        config.LOSS_CONFIG["edge_loss_weight"] * edge_loss +
                        config.LOSS_CONFIG["vq_loss_weight"] * vq_loss)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            self.logger.info(f"Client {self.client_id} Personalization Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")


    def train(self, current_round, global_shared_params):
        self.model.train()
        is_interactive = sys.stdout.isatty()

        for epoch in range(config.LOCAL_EPOCHS):
            running_loss = 0.0
            progress_bar = tqdm(self.train_loader, desc=f"Client {self.client_id} Epoch {epoch+1}/{config.LOCAL_EPOCHS}", disable=not is_interactive)

            for batch in progress_bar:
                if batch is None:
                    continue
                lr_images = batch['lr'].to(config.DEVICE)
                hr_images = batch['hr'].to(config.DEVICE)

                self.optimizer.zero_grad()
                
                reconstructed_images, vq_loss = self.model(lr_images)
                
                reconstructed_images_0_1 = (reconstructed_images + 1) / 2
                hr_images_0_1 = (hr_images + 1) / 2

                reconstructed_images_0_1 = torch.clamp(reconstructed_images_0_1, 0.0, 1.0)
                hr_images_0_1 = torch.clamp(hr_images_0_1, 0.0, 1.0)
                
                recon_loss = self.recon_loss_fn(reconstructed_images, hr_images)
                x_lpips = self._to_lpips(reconstructed_images)
                y_lpips = self._to_lpips(hr_images)
                perceptual_loss = self.perceptual_loss_fn(x_lpips, y_lpips).mean()
                structural_loss = torch.clamp(self.structural_loss_fn(reconstructed_images_0_1, hr_images_0_1), min=0.0)
                edge_loss = self.edge_loss_fn(reconstructed_images, hr_images)  # NEW

                proximal_term = 0.0
                if config.AGGREGATOR == 'fedprox' and current_round > 1:
                    for name, local_param in self.model.named_parameters():
                        if name in global_shared_params:
                            global_param = global_shared_params[name]
                            proximal_term += torch.sum((local_param - global_param.to(config.DEVICE)) ** 2)
                
                total_loss = (config.LOSS_CONFIG["reconstruction_loss_weight"] * recon_loss + 
                              config.LOSS_CONFIG["perceptual_loss_weight"] * perceptual_loss +
                              config.LOSS_CONFIG["structural_loss_weight"] * structural_loss +
                              config.LOSS_CONFIG["edge_loss_weight"] * edge_loss +
                              config.LOSS_CONFIG["vq_loss_weight"] * vq_loss + 
                              (config.FEDPROX_MU / 2) * proximal_term)

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()

                running_loss += total_loss.item()
                if is_interactive:
                    current_lr = self.scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({"Loss": total_loss.item(), "LR": f"{current_lr:.1e}"})
            avg_epoch_loss = running_loss / len(self.train_loader)
            self.logger.info(
                f"Client {self.client_id} > Epoch {epoch+1}/{config.LOCAL_EPOCHS} | "
                f"Total Loss: {avg_epoch_loss:.4f} | "
                f"Components: Recon={recon_loss.item():.4f}, "
                f"Perceptual={perceptual_loss.item():.4f}, "
                f"Structural={structural_loss.item():.4f}, "
                f"Edge={edge_loss.item():.4f}, "
                f"VQ={vq_loss.item():.4f}"
            )

    def validate(self):
        """Validation with PSNR, SSIM, LPIPS, MSE, FID, and loss."""
        self.model.eval()
        all_psnr, all_ssim, all_lpips, all_mse, all_loss = [], [], [], [], []
        self.fid_metric.reset()
        total_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                if batch is None:
                    continue
                lr_images = batch['lr'].to(config.DEVICE)
                hr_images = batch['hr'].to(config.DEVICE)

                reconstructed_images, vq_loss = self.model(lr_images)

                reconstructed_images_0_1 = torch.clamp((reconstructed_images + 1) / 2, 0.0, 1.0)
                hr_images_0_1 = torch.clamp((hr_images + 1) / 2, 0.0, 1.0)

                # --- Losses ---
                recon_loss = self.recon_loss_fn(reconstructed_images, hr_images)
                x_lpips = self._to_lpips(reconstructed_images)
                y_lpips = self._to_lpips(hr_images)
                perceptual_loss = self.perceptual_loss_fn(x_lpips, y_lpips).mean()
                structural_loss = torch.clamp(self.structural_loss_fn(reconstructed_images_0_1, hr_images_0_1), min=0.0)

                loss = (
                    config.LOSS_CONFIG["reconstruction_loss_weight"] * recon_loss +
                    config.LOSS_CONFIG["perceptual_loss_weight"] * perceptual_loss +
                    config.LOSS_CONFIG["structural_loss_weight"] * structural_loss +
                    config.LOSS_CONFIG["vq_loss_weight"] * vq_loss
                )

                # --- Metrics ---
                psnr_val = self.psnr_metric(reconstructed_images_0_1, hr_images_0_1)
                ssim_val = self.ssim_metric(reconstructed_images_0_1, hr_images_0_1)
                lpips_val = self.lpips_metric(x_lpips, y_lpips)
                mse_val = self.mse_metric(reconstructed_images_0_1, hr_images_0_1)

                all_psnr.append(psnr_val.item())
                all_ssim.append(ssim_val.item())
                all_lpips.append(lpips_val.mean().item())
                all_mse.append(mse_val.item())
                all_loss.append(loss.item())

                total_samples += hr_images.shape[0]

                # FID update (convert to uint8 3-channel)
                def _prep_fid(x):
                    x = torch.clamp(x, 0.0, 1.0)
                    if x.shape[1] == 1:
                        x = x.repeat(1, 3, 1, 1)
                    return (x * 255).to(torch.uint8)
                self.fid_metric.update(_prep_fid(hr_images_0_1), real=True)
                self.fid_metric.update(_prep_fid(reconstructed_images_0_1), real=False)

        if total_samples < 2000:
            self.logger.warning(
                f"Client {self.client_id}: FID computed on only {total_samples} samples. "
                f"Results may be statistically unreliable (need 2000+)."
            )
        fid_val = self.fid_metric.compute().item()

        return {
            "psnr": np.mean(all_psnr),
            "ssim": np.mean(all_ssim),
            "lpips": np.mean(all_lpips),
            "mse": np.mean(all_mse),
            "fid": fid_val,
            "loss": np.mean(all_loss)
        }

    def evaluate(self, round_num, results_path):
        """Evaluation on test set with metrics + visual save."""
        self.model.eval()
        all_psnr, all_ssim, all_lpips, all_mse, all_loss = [], [], [], [], []
        self.fid_metric.reset()
        total_samples = 0

        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                if batch is None:
                    continue
                lr_images = batch['lr'].to(config.DEVICE)
                hr_images = batch['hr'].to(config.DEVICE)

                reconstructed_images, vq_loss = self.model(lr_images)

                reconstructed_images_0_1 = torch.clamp((reconstructed_images + 1) / 2, 0.0, 1.0)
                hr_images_0_1 = torch.clamp((hr_images + 1) / 2, 0.0, 1.0)

                # Losses
                recon_loss = self.recon_loss_fn(reconstructed_images, hr_images)
                x_lpips = self._to_lpips(reconstructed_images)
                y_lpips = self._to_lpips(hr_images)
                perceptual_loss = self.perceptual_loss_fn(x_lpips, y_lpips).mean()
                structural_loss = torch.clamp(self.structural_loss_fn(reconstructed_images_0_1, hr_images_0_1), min=0.0)

                loss = (
                    config.LOSS_CONFIG["reconstruction_loss_weight"] * recon_loss +
                    config.LOSS_CONFIG["perceptual_loss_weight"] * perceptual_loss +
                    config.LOSS_CONFIG["structural_loss_weight"] * structural_loss +
                    config.LOSS_CONFIG["vq_loss_weight"] * vq_loss
                )

                # Metrics
                psnr_val = self.psnr_metric(reconstructed_images_0_1, hr_images_0_1)
                ssim_val = self.ssim_metric(reconstructed_images_0_1, hr_images_0_1)
                lpips_val = self.lpips_metric(x_lpips, y_lpips)
                mse_val = self.mse_metric(reconstructed_images_0_1, hr_images_0_1)

                all_psnr.append(psnr_val.item())
                all_ssim.append(ssim_val.item())
                all_lpips.append(lpips_val.mean().item())  # .item() converts tensor to float
                all_mse.append(mse_val.item())
                all_loss.append(loss.item())

                total_samples += hr_images.shape[0]

                # FID update (convert to uint8 3-channel)
                def _prep_fid(x):
                    x = torch.clamp(x, 0.0, 1.0)
                    if x.shape[1] == 1:
                        x = x.repeat(1, 3, 1, 1)
                    return (x * 255).to(torch.uint8)
                self.fid_metric.update(_prep_fid(hr_images_0_1), real=True)
                self.fid_metric.update(_prep_fid(reconstructed_images_0_1), real=False)

                # Log first batch loss components
                if i == 0:
                    self.logger.info(
                        f"Client {self.client_id} Test Loss Components (1st batch): "
                        f"Recon={recon_loss.item():.4f}, "
                        f"Perceptual={perceptual_loss.item():.4f}, "
                        f"Structural={structural_loss.item():.4f}, "
                        f"VQ={vq_loss.item():.4f}"
                    )

                # Save first visual example
                if i == 0:
                    save_visual_results(
                        lr_images[0], hr_images[0], reconstructed_images[0],
                        results_path, round_num, self.client_id, sample_idx=0
                    )

        if total_samples < 2000:
            self.logger.warning(
                f"Client {self.client_id}: FID computed on only {total_samples} samples. "
                f"Results may be statistically unreliable (need 2000+)."
            )
        avg_psnr = np.mean(all_psnr)
        avg_ssim = np.mean(all_ssim)
        avg_lpips = np.mean(all_lpips)
        avg_mse = np.mean(all_mse)
        avg_loss = np.mean(all_loss)
        fid_val = self.fid_metric.compute().item()

        self.results_logger.add_round_results(
            round_num, self.client_id, avg_psnr, avg_ssim, avg_loss, 
            lpips_val=avg_lpips, mse_val=avg_mse, fid_val=fid_val, is_global=False
        )

        return {
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "lpips": avg_lpips,
            "mse": avg_mse,
            "fid": fid_val,
            "loss": avg_loss
        }