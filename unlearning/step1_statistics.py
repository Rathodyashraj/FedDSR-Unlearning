# unlearning/step1_statistics.py
"""
Step 1: statistics pass at each remaining client.

One pass of the old encoder over the client's LR training slices gives, per level:
  n[j]  number of positions whose nearest old-codebook row is j
  s[j]  sum of the latents of those positions
  m     median margin (second-smallest minus smallest squared distance), from a fixed-seed sample

    python -m unlearning.step1_statistics --fixture --withdraw 3
"""
import argparse
import logging
import os
import sys

import numpy as np
import torch

from unlearning import settings
from unlearning import helpers

LEVEL_KEYS = {"top": "top", "bottom": "bot"}  # level -> suffix used in the saved file


def stats_path(withdrawn, k):
    return os.path.join(settings.step_dir(withdrawn, 1), f"stats_client_{k}.pt")


# -----------------
# Core arithmetic (also used by the toy check)
# -----------------
def nearest_and_margin(z_flat, codebook):
    """
    Nearest row (ties to the lowest index) and margin per position, computed in blocks of positions.
    z_flat: [P, D]; codebook: [K, D]. Returns idx [P] (long) and margin [P] (float).
    """
    idx_parts, margin_parts = [], []
    for start in range(0, z_flat.shape[0], settings.DISTANCE_CHUNK_POSITIONS):
        dist = helpers.squared_distances(z_flat[start:start + settings.DISTANCE_CHUNK_POSITIONS], codebook)
        idx_parts.append(torch.argmin(dist, dim=1))
        two = torch.topk(dist, 2, dim=1, largest=False).values
        margin_parts.append((two[:, 1] - two[:, 0]).clamp(min=0))
    return torch.cat(idx_parts), torch.cat(margin_parts)


class LevelAccumulator:
    """Counts, sums (float64) and a margin sample for one level."""

    def __init__(self, rows, dim, device):
        self.n = torch.zeros(rows, dtype=torch.float64, device=device)
        self.s = torch.zeros(rows, dim, dtype=torch.float64, device=device)
        self.margins = []
        self.positions = 0

    def add(self, z_flat, codebook, generator=None, sample_size=None):
        idx, margin = nearest_and_margin(z_flat, codebook)
        self.n.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float64))
        self.s.index_add_(0, idx, z_flat.double())
        if sample_size is not None and margin.numel() > sample_size:
            pick = torch.randperm(margin.numel(), generator=generator)[:sample_size].to(margin.device)
            margin = margin[pick]
        self.margins.append(margin.detach().double().cpu())
        self.positions += z_flat.shape[0]

    def median_margin(self):
        return float(np.median(torch.cat(self.margins).numpy()))


# -----------------
# Pass over one client
# -----------------
@torch.no_grad()
def client_statistics(ref_model, k):
    """One pass over client k's training split: no shuffling, no dropped batch, every slice once."""
    device = next(ref_model.parameters()).device
    loader = helpers.make_loader(k, "train", shuffle=False, drop_last=False)
    codebooks = {level: helpers.quantizer(ref_model, level)._embedding.weight for level in helpers.LEVELS}
    acc = {level: LevelAccumulator(cb.shape[0], cb.shape[1], device) for level, cb in codebooks.items()}
    generator = torch.Generator().manual_seed(settings.SEED + 1000 * k)
    slices_seen = 0
    for batch in loader:
        if batch is None:
            continue
        lr = batch["lr"].to(device)
        latents = dict(zip(("top", "bottom"), ref_model.encode(lr)))  # never forward(): no counters, no vitalization
        for level in helpers.LEVELS:
            acc[level].add(helpers.flatten_latent(latents[level]), codebooks[level],
                           generator=generator, sample_size=settings.MARGIN_SAMPLE_PER_BATCH)
        slices_seen += lr.shape[0]

    out = {
        "client": k,
        "num_train_slices": len(loader.dataset),
        "slices_seen": slices_seen,
    }
    for level, suffix in LEVEL_KEYS.items():
        a = acc[level]
        out[f"n_{suffix}"] = a.n.cpu()
        out[f"s_{suffix}"] = a.s.cpu()
        out[f"m_{suffix}"] = a.median_margin()
        out[f"positions_{suffix}"] = a.positions
        out[f"margin_sample_{suffix}"] = int(sum(m.numel() for m in a.margins))
    return out


def run_step1(withdrawn, overwrite=False):
    logger = logging.getLogger()
    helpers.set_withdrawn(withdrawn)
    clients = helpers.remaining_clients(withdrawn)
    if not overwrite:
        existing = [stats_path(withdrawn, k) for k in clients if os.path.exists(stats_path(withdrawn, k))]
        if existing:
            raise FileExistsError(f"Step 1 output exists ({existing[0]}); pass --overwrite to redo it")

    ref = helpers.load_reference_model()
    logger.info(f"Step 1: withdrawn client {withdrawn}; statistics at clients {clients}")
    results = {}
    for k in clients:
        stats = client_statistics(ref, k)
        if stats["slices_seen"] != stats["num_train_slices"]:
            logger.warning(f"Client {k}: {stats['num_train_slices'] - stats['slices_seen']} training slices "
                           f"were dropped by the loader (unreadable or blank images)")
        for level, suffix in LEVEL_KEYS.items():
            n = stats[f"n_{suffix}"]
            logger.info(f"Client {k} {level}: {stats[f'positions_{suffix}']:,} positions, "
                        f"{int((n > 0).sum())}/{n.numel()} rows used, median margin {stats[f'm_{suffix}']:.6g} "
                        f"(sample of {stats[f'margin_sample_{suffix}']:,})")
        helpers.atomic_save(stats, stats_path(withdrawn, k))
        logger.info(f"Client {k}: {stats['slices_seen']} slices -> {stats_path(withdrawn, k)}")
        results[k] = stats
    return results


def main():
    parser = argparse.ArgumentParser(description="Step 1: statistics pass at each remaining client")
    settings.add_common_args(parser)
    parser.add_argument("--withdraw", type=int, required=True, help="Withdrawn client id")
    parser.add_argument("--trial", action="store_true", help="Real model, first slices only")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    settings.configure(results_dir=args.results_dir, fixture=args.fixture, trial=args.trial)
    helpers.setup_logging(settings.logs_dir(args.withdraw), "step1_stats")
    try:
        run_step1(args.withdraw, overwrite=args.overwrite)
    except FileExistsError as e:
        logging.getLogger().error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
