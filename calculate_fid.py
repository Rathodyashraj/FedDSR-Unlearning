# calculate_fid.py
"""
Compute FID for each client using test sets.

Loads the shared encoder+quantizer from best_model.pth and the best
available decoder per client (refined if present, otherwise Stage-1).
"""

import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

import config
from data import get_dataloaders
from models import VQVAE2
from utils import setup_logger


def _prepare_fid_images(x: torch.Tensor) -> torch.Tensor:
    """Prepare images for FID (uint8, 3-channel, [0,255])."""
    if x.min() < 0:
        x = (x + 1) / 2
    x = torch.clamp(x, 0.0, 1.0)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return (x * 255).to(torch.uint8)


def _load_model_for_client(client_id: int, device: torch.device) -> VQVAE2:
    model = VQVAE2().to(device)

    # Load shared encoder+quantizer
    shared_path = os.path.join(config.RESULTS_DIR, config.AGGREGATOR, "best_model.pth")
    if os.path.exists(shared_path):
        shared_state = torch.load(shared_path, map_location=device)
        model.load_state_dict(shared_state, strict=False)
    else:
        raise FileNotFoundError(f"Shared model not found: {shared_path}")

    # Prefer refined decoder (Stage-2), fallback to best Stage-1 decoder
    refined_path = os.path.join(config.RESULTS_DIR, "DecoderFineTuning_Local", f"best_refined_decoder_client_{client_id}.pth")
    stage1_path = os.path.join(config.RESULTS_DIR, config.AGGREGATOR, f"best_decoder_client_{client_id}.pth")

    if os.path.exists(refined_path):
        decoder_state = torch.load(refined_path, map_location=device)
        model.load_state_dict(decoder_state, strict=False)
        logging.getLogger().info(f"Client {client_id}: loaded refined decoder")
    elif os.path.exists(stage1_path):
        decoder_state = torch.load(stage1_path, map_location=device)
        model.load_state_dict(decoder_state, strict=False)
        logging.getLogger().info(f"Client {client_id}: loaded Stage-1 decoder")
    else:
        logging.getLogger().warning(f"Client {client_id}: decoder not found, using default decoder weights")

    model.eval()
    return model


def _compute_fid_for_loader(model: VQVAE2, loader, device: torch.device, max_batches: int | None) -> float:
    fid = FrechetInceptionDistance(feature=2048).to(device)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if batch is None:
                continue
            if max_batches is not None and i >= max_batches:
                break

            lr_images = batch["lr"].to(device)
            hr_images = batch["hr"].to(device)

            sr_images, _ = model(lr_images)

            real_uint8 = _prepare_fid_images(hr_images)
            fake_uint8 = _prepare_fid_images(sr_images)

            fid.update(real_uint8, real=True)
            fid.update(fake_uint8, real=False)

    return fid.compute().item()


def main(args):
    torch.manual_seed(42)
    np.random.seed(42)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        config.DEVICE = torch.device(f"cuda:{args.gpu}")

    logger = setup_logger()

    log_filename = f"fid_calculation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(config.RESULTS_DIR, "FID", log_filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)

    logger.info("=" * 80)
    logger.info("FID CALCULATION (Per-Client Test Sets)")
    logger.info("=" * 80)
    logger.info(f"Using device: {config.DEVICE}")

    client_ids_to_run = list(config.CLIENT_DATASETS.keys())[:args.clients]
    fid_scores = []

    for client_id in client_ids_to_run:
        logger.info(f"--- Client {client_id} ({config.CLIENT_DATASETS[client_id]['name']}) ---")
        _, _, test_loader = get_dataloaders(client_id)

        model = _load_model_for_client(client_id, config.DEVICE)
        fid = _compute_fid_for_loader(model, test_loader, config.DEVICE, args.max_batches)

        fid_scores.append(fid)
        logger.info(f"Client {client_id} FID: {fid:.4f}")

    if fid_scores:
        avg_fid = float(np.mean(fid_scores))
        logger.info(f"Global Average FID: {avg_fid:.4f}")

    logger.info(f"FID log saved to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FID for FedMedSR models")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--clients", type=int, default=4, help="Number of clients to evaluate")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches for quick check")
    main(parser.parse_args())
