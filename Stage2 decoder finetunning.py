"""
Stage-2: Local Decoder Fine-Tuning
====================================
Run this AFTER Stage-1 federated training is complete.
Loads the best shared encoder+quantizer from best_model.pth,
freezes it, and fine-tunes each client's decoder independently
using the full loss suite.

Usage:
    python stage2_decoder_finetune.py --gpu 0 --epochs 10 --clients 4
"""

import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import lpips
import piq
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance

import config
from data import get_dataloaders
from models import VQVAE2
from utils import (
    setup_logger,
    ResultsLogger,
    CharbonnierLoss,
    EdgeAwareLoss,
    save_visual_results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_lpips(x: torch.Tensor) -> torch.Tensor:
    """Convert 1-channel [-1,1] tensor to 3-channel [-1,1] for LPIPS."""
    x = torch.clamp(x, -1, 1)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return x


def _prep_fid(x: torch.Tensor) -> torch.Tensor:
    """Convert [0,1] tensor to uint8 3-channel for FID."""
    x = torch.clamp(x, 0.0, 1.0)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return (x * 255).to(torch.uint8)


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_stage1_model(client_id: int, device: torch.device, logger: logging.Logger) -> VQVAE2:
    """
    Load best Stage-1 shared encoder+quantizer.
    Falls back to best_decoder_client_{id}.pth for decoder init.
    """
    model = VQVAE2().to(device)

    # Load shared encoder + quantizer (mandatory)
    shared_path = os.path.join(config.RESULTS_DIR, config.AGGREGATOR, "best_model.pth")
    if not os.path.exists(shared_path):
        raise FileNotFoundError(
            f"Stage-1 shared model not found at: {shared_path}\n"
            f"Run Stage-1 training first (main.py)."
        )
    shared_state = torch.load(shared_path, map_location=device)
    model.load_state_dict(shared_state, strict=False)
    logger.info(f"Client {client_id}: Loaded shared encoder+quantizer from {shared_path}")

    # Load best Stage-1 decoder for this client (optional but recommended)
    decoder_path = os.path.join(
        config.RESULTS_DIR, config.AGGREGATOR,
        f"best_decoder_client_{client_id}.pth"
    )
    if os.path.exists(decoder_path):
        decoder_state = torch.load(decoder_path, map_location=device)
        model.load_state_dict(decoder_state, strict=False)
        logger.info(f"Client {client_id}: Loaded Stage-1 decoder from {decoder_path}")
    else:
        logger.warning(f"Client {client_id}: No Stage-1 decoder found, using default init.")

    return model


def freeze_encoder_quantizer(model: VQVAE2, logger: logging.Logger, client_id: int):
    """Freeze all parameters except decoder."""
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        if not name.startswith("decoder."):
            param.requires_grad = False
            frozen_count += 1
        else:
            param.requires_grad = True
            trainable_count += 1
    logger.info(
        f"Client {client_id}: Frozen {frozen_count} param groups (encoder+quantizer), "
        f"{trainable_count} trainable (decoder only)."
    )


def unfreeze_all(model: VQVAE2):
    """Unfreeze all parameters after fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def finetune_decoder(
    model: VQVAE2,
    train_loader,
    client_id: int,
    epochs: int,
    lr: float,
    device: torch.device,
    logger: logging.Logger,
    results_dir: str,
):
    """Fine-tune decoder with full loss suite."""

    # Loss functions
    recon_loss_fn   = CharbonnierLoss(eps=config.LOSS_CONFIG["charbonnier_eps"]).to(device)
    perceptual_fn   = lpips.LPIPS(net="vgg").to(device)
    structural_fn   = piq.MultiScaleSSIMLoss(data_range=1.0).to(device)
    edge_fn         = EdgeAwareLoss().to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=len(train_loader) * epochs,
        eta_min=1e-7,
    )

    best_loss = float("inf")
    best_state = None

    model.train()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        running_recon = 0.0
        running_perceptual = 0.0
        running_structural = 0.0
        running_edge = 0.0
        running_vq = 0.0

        for batch in train_loader:
            if batch is None:
                continue

            lr_images = batch["lr"].to(device)
            hr_images = batch["hr"].to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                # Encoder + quantizer are frozen — no grad needed
                pass

            reconstructed, vq_loss = model(lr_images)

            rec_01 = torch.clamp((reconstructed + 1) / 2, 0.0, 1.0)
            hr_01  = torch.clamp((hr_images  + 1) / 2, 0.0, 1.0)

            recon_loss      = recon_loss_fn(reconstructed, hr_images)
            x_lpips         = _to_lpips(reconstructed)
            y_lpips         = _to_lpips(hr_images)
            perceptual_loss = perceptual_fn(x_lpips, y_lpips).mean()
            structural_loss = torch.clamp(structural_fn(rec_01, hr_01), min=0.0)
            edge_loss       = edge_fn(reconstructed, hr_images)

            total_loss = (
                config.LOSS_CONFIG["reconstruction_loss_weight"] * recon_loss
                + config.LOSS_CONFIG["perceptual_loss_weight"]   * perceptual_loss
                + config.LOSS_CONFIG["structural_loss_weight"]   * structural_loss
                + config.LOSS_CONFIG["edge_loss_weight"]         * edge_loss
                + config.LOSS_CONFIG["vq_loss_weight"]           * vq_loss
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss        += total_loss.item()
            running_recon       += recon_loss.item()
            running_perceptual  += perceptual_loss.item()
            running_structural  += structural_loss.item()
            running_edge        += edge_loss.item()
            running_vq          += vq_loss.item()

        n_batches = len(train_loader)
        avg_loss = running_loss / n_batches

        logger.info(
            f"Client {client_id} | Stage-2 Epoch {epoch}/{epochs} | "
            f"Loss={avg_loss:.4f} | "
            f"Recon={running_recon/n_batches:.4f}, "
            f"Perceptual={running_perceptual/n_batches:.4f}, "
            f"Structural={running_structural/n_batches:.4f}, "
            f"Edge={running_edge/n_batches:.4f}, "
            f"VQ={running_vq/n_batches:.4f}"
        )

        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            logger.info(f"Client {client_id}: New best loss {best_loss:.4f} — snapshot saved.")

    # Restore best weights from this fine-tuning run
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"Client {client_id}: Restored best Stage-2 weights (loss={best_loss:.4f}).")

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: VQVAE2,
    test_loader,
    client_id: int,
    device: torch.device,
    logger: logging.Logger,
    results_dir: str,
    tag: str = "stage2",
) -> dict:
    """Full evaluation: PSNR, SSIM, LPIPS, MSE, FID + visual save."""

    model.eval()

    recon_loss_fn  = CharbonnierLoss(eps=config.LOSS_CONFIG["charbonnier_eps"]).to(device)
    perceptual_fn  = lpips.LPIPS(net="vgg").to(device)
    structural_fn  = piq.MultiScaleSSIMLoss(data_range=1.0).to(device)

    psnr_metric  = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = lpips.LPIPS(net="vgg").to(device)
    mse_metric   = nn.MSELoss()
    fid_metric   = FrechetInceptionDistance(feature=2048).to(device)

    all_psnr, all_ssim, all_lpips, all_mse, all_loss = [], [], [], [], []
    total_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if batch is None:
                continue

            lr_images = batch["lr"].to(device)
            hr_images = batch["hr"].to(device)

            reconstructed, vq_loss = model(lr_images)

            rec_01 = torch.clamp((reconstructed + 1) / 2, 0.0, 1.0)
            hr_01  = torch.clamp((hr_images  + 1) / 2, 0.0, 1.0)

            recon_loss      = recon_loss_fn(reconstructed, hr_images)
            x_lpips         = _to_lpips(reconstructed)
            y_lpips         = _to_lpips(hr_images)
            perceptual_loss = perceptual_fn(x_lpips, y_lpips).mean()
            structural_loss = torch.clamp(structural_fn(rec_01, hr_01), min=0.0)

            loss = (
                config.LOSS_CONFIG["reconstruction_loss_weight"] * recon_loss
                + config.LOSS_CONFIG["perceptual_loss_weight"]   * perceptual_loss
                + config.LOSS_CONFIG["structural_loss_weight"]   * structural_loss
                + config.LOSS_CONFIG["vq_loss_weight"]           * vq_loss
            )

            psnr_val  = psnr_metric(rec_01, hr_01)
            ssim_val  = ssim_metric(rec_01, hr_01)
            lpips_val = lpips_metric(x_lpips, y_lpips)
            mse_val   = mse_metric(rec_01, hr_01)

            all_psnr.append(psnr_val.item())
            all_ssim.append(ssim_val.item())
            all_lpips.append(lpips_val.mean().item())
            all_mse.append(mse_val.item())
            all_loss.append(loss.item())

            fid_metric.update(_prep_fid(hr_01),  real=True)
            fid_metric.update(_prep_fid(rec_01), real=False)
            total_samples += hr_images.shape[0]

            # Save visual for first batch
            if i == 0:
                save_visual_results(
                    lr_images[0], hr_images[0], reconstructed[0],
                    results_dir, round_num=tag, client_id=client_id, sample_idx=0
                )

    if total_samples < 2000:
        logger.warning(
            f"Client {client_id}: FID computed on {total_samples} samples — "
            f"need 2000+ for reliable results."
        )

    fid_val = fid_metric.compute().item()

    metrics = {
        "psnr":  float(np.mean(all_psnr)),
        "ssim":  float(np.mean(all_ssim)),
        "lpips": float(np.mean(all_lpips)),
        "mse":   float(np.mean(all_mse)),
        "fid":   fid_val,
        "loss":  float(np.mean(all_loss)),
    }

    logger.info(
        f"Client {client_id} [{tag}] | "
        f"PSNR={metrics['psnr']:.4f} dB | "
        f"SSIM={metrics['ssim']:.4f} | "
        f"LPIPS={metrics['lpips']:.4f} | "
        f"MSE={metrics['mse']:.6f} | "
        f"FID={metrics['fid']:.4f} | "
        f"Loss={metrics['loss']:.4f}"
    )

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    torch.manual_seed(42)
    np.random.seed(42)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        config.DEVICE = torch.device(f"cuda:{args.gpu}")

    device = config.DEVICE

    # Setup logging
    logger = setup_logger()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    refined_dir = os.path.join(config.RESULTS_DIR, "DecoderFineTuning_Local")
    os.makedirs(refined_dir, exist_ok=True)

    log_path = os.path.join(refined_dir, f"stage2_finetune_{timestamp}.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    logger.info("=" * 80)
    logger.info("STAGE-2: LOCAL DECODER FINE-TUNING")
    logger.info("=" * 80)
    logger.info(f"Device     : {device}")
    logger.info(f"Epochs     : {args.epochs}")
    logger.info(f"LR         : {args.lr:.2e}")
    logger.info(f"Clients    : {args.clients}")
    logger.info(f"Output dir : {refined_dir}")

    client_ids = list(config.CLIENT_DATASETS.keys())[: args.clients]
    all_metrics_before = {}
    all_metrics_after  = {}

    for client_id in client_ids:
        logger.info(f"\n{'='*60}")
        logger.info(f"CLIENT {client_id} — {config.CLIENT_DATASETS[client_id]['name']}")
        logger.info(f"{'='*60}")

        train_loader, val_loader, test_loader = get_dataloaders(client_id)

        # ---- Load Stage-1 model ----
        model = load_stage1_model(client_id, device, logger)

        # ---- Evaluate BEFORE fine-tuning ----
        logger.info(f"Client {client_id}: Evaluating Stage-1 baseline...")
        metrics_before = evaluate_model(
            model, test_loader, client_id, device, logger,
            results_dir=refined_dir, tag="stage1_baseline"
        )
        all_metrics_before[client_id] = metrics_before

        # ---- Freeze encoder + quantizer ----
        freeze_encoder_quantizer(model, logger, client_id)

        # ---- Fine-tune decoder ----
        logger.info(f"Client {client_id}: Starting decoder fine-tuning for {args.epochs} epochs...")
        model = finetune_decoder(
            model, train_loader, client_id,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            logger=logger,
            results_dir=refined_dir,
        )

        # ---- Unfreeze for saving ----
        unfreeze_all(model)

        # ---- Save refined decoder ----
        save_path = os.path.join(
            refined_dir, f"best_refined_decoder_client_{client_id}.pth"
        )
        torch.save(model.state_dict(), save_path)
        logger.info(f"Client {client_id}: Refined decoder saved to {save_path}")

        # ---- Evaluate AFTER fine-tuning ----
        logger.info(f"Client {client_id}: Evaluating Stage-2 refined model...")
        metrics_after = evaluate_model(
            model, test_loader, client_id, device, logger,
            results_dir=refined_dir, tag="stage2_refined"
        )
        all_metrics_after[client_id] = metrics_after

    # ---- Final Summary ----
    logger.info("\n" + "=" * 80)
    logger.info("STAGE-2 COMPLETE — SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'Client':<10} {'Metric':<10} {'Stage-1':>12} {'Stage-2':>12} {'Delta':>10}")
    logger.info("-" * 56)

    for client_id in client_ids:
        b = all_metrics_before[client_id]
        a = all_metrics_after[client_id]
        name = config.CLIENT_DATASETS[client_id]["name"]
        for metric in ["psnr", "ssim", "lpips", "fid"]:
            delta = a[metric] - b[metric]
            sign  = "+" if delta >= 0 else ""
            logger.info(
                f"{f'C{client_id} ({name})':<10} {metric.upper():<10} "
                f"{b[metric]:>12.4f} {a[metric]:>12.4f} {sign+f'{delta:.4f}':>10}"
            )
        logger.info("-" * 56)

    # Global averages
    avg_before_psnr = np.mean([all_metrics_before[c]["psnr"] for c in client_ids])
    avg_after_psnr  = np.mean([all_metrics_after[c]["psnr"]  for c in client_ids])
    avg_before_ssim = np.mean([all_metrics_before[c]["ssim"] for c in client_ids])
    avg_after_ssim  = np.mean([all_metrics_after[c]["ssim"]  for c in client_ids])
    avg_before_fid  = np.mean([all_metrics_before[c]["fid"]  for c in client_ids])
    avg_after_fid   = np.mean([all_metrics_after[c]["fid"]   for c in client_ids])

    logger.info(
        f"\nGlobal Average | "
        f"PSNR: {avg_before_psnr:.4f} → {avg_after_psnr:.4f} "
        f"(+{avg_after_psnr - avg_before_psnr:.4f} dB) | "
        f"SSIM: {avg_before_ssim:.4f} → {avg_after_ssim:.4f} | "
        f"FID: {avg_before_fid:.4f} → {avg_after_fid:.4f}"
    )
    logger.info(f"Log saved to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage-2 Local Decoder Fine-Tuning")
    parser.add_argument("--gpu",     type=int,   default=0,    help="GPU ID")
    parser.add_argument("--epochs",  type=int,   default=10,   help="Fine-tuning epochs per client")
    parser.add_argument("--clients", type=int,   default=4,    help="Number of clients to fine-tune")
    parser.add_argument("--lr",      type=float, default=5e-6, help="Learning rate for decoder fine-tuning")
    main(parser.parse_args())