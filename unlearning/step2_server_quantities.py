# unlearning/step2_server_quantities.py
"""
Step 2: server quantities from the remaining clients' statistics.

Reads only stats_client_k.pt of the remaining clients and writes server_quantities.pt:
aggregation weights, clean codebook and active-row mask per level, and the margin per level.

    python -m unlearning.step2_server_quantities --fixture --withdraw 3
"""
import argparse
import logging
import os
import sys

import torch
import torch.nn.functional as F

from unlearning import settings
from unlearning import helpers
from unlearning.step1_statistics import LEVEL_KEYS, stats_path


def quantities_path(withdrawn):
    return os.path.join(settings.step_dir(withdrawn, 2), "server_quantities.pt")



def client_descriptor(stats):
    """c_k = 1/2 (mean top latent + mean bottom latent)."""
    means = [stats[f"s_{sfx}"].sum(0) / stats[f"n_{sfx}"].sum() for sfx in LEVEL_KEYS.values()]
    if means[0].shape != means[1].shape:
        raise ValueError("top and bottom latents have different widths; the descriptor needs equal widths")
    return 0.5 * (means[0] + means[1])


def similarity_weights(descriptors):
    """Mean cosine to the other clients; negatives set to 0; normalised. All zero -> equal weights."""
    stacked = torch.stack(descriptors)
    normed = F.normalize(stacked, p=2, dim=1)
    sim = normed @ normed.t()
    n = len(descriptors)
    off_diag = ~torch.eye(n, dtype=torch.bool)
    scores = torch.stack([sim[i][off_diag[i]].mean() for i in range(n)])
    clipped = scores.clamp(min=0)
    if float(clipped.sum()) == 0:
        return scores, torch.full((n,), 1.0 / n, dtype=scores.dtype)
    return scores, clipped / clipped.sum()


def compute_server_quantities(stats_by_client, gamma):
    """stats_by_client: {k: stats dict from Step 1}. All arithmetic in float64."""
    clients = sorted(stats_by_client)
    stats = [stats_by_client[k] for k in clients]

    descriptors = [client_descriptor(st) for st in stats]
    sim_scores, w_sim = similarity_weights(descriptors)
    sizes = torch.tensor([float(st["num_train_slices"]) for st in stats], dtype=torch.float64)
    w_data = sizes / sizes.sum()
    w = gamma * w_sim + (1 - gamma) * w_data
    if abs(float(w.sum()) - 1.0) > 1e-9:
        raise AssertionError(f"weights sum to {float(w.sum())}, not 1")

    out = {
        "clients": clients,
        "weights": {k: float(w[i]) for i, k in enumerate(clients)},
        "w_sim": {k: float(w_sim[i]) for i, k in enumerate(clients)},
        "w_data": {k: float(w_data[i]) for i, k in enumerate(clients)},
        "similarity_scores": {k: float(sim_scores[i]) for i, k in enumerate(clients)},
        "descriptors": {k: descriptors[i] for i, k in enumerate(clients)},
        "gamma": gamma,
    }
    for sfx in LEVEL_KEYS.values():
        n = torch.stack([st[f"n_{sfx}"] for st in stats])          # [C, K]
        s = torch.stack([st[f"s_{sfx}"] for st in stats])          # [C, K, D]
        used = n >= 1                                              # client k used row j
        mask = used.any(dim=0)                                     # summed count >= 1
        wk = w.view(-1, 1) * used                                  # w_k where used, else 0
        means = s / n.clamp(min=1).unsqueeze(-1)                   # s_kj / n_kj (0 where unused)
        num = (wk.unsqueeze(-1) * means).sum(0)
        den = wk.sum(0)
        V = torch.zeros_like(num)
        V[mask] = num[mask] / den[mask].unsqueeze(-1)
        margins = torch.tensor([float(st[f"m_{sfx}"]) for st in stats], dtype=torch.float64)
        out[f"V_{sfx}_f64"] = V
        out[f"V_{sfx}"] = V.float()                                # used as a codebook, same dtype as the model
        out[f"mask_{sfx}"] = mask
        out[f"delta_{sfx}"] = float(0.5 * (w * margins).sum())
        out[f"active_rows_{sfx}"] = int(mask.sum())
    return out


# -----------------
# Step 2
# -----------------
def load_remaining_stats(withdrawn):
    helpers.set_withdrawn(withdrawn)
    stats = {}
    for k in helpers.remaining_clients(withdrawn):
        path = stats_path(withdrawn, k)
        helpers.check_client_access(k, "stats", path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing: run Step 1 first")
        stats[k] = helpers.load_file(path)
    return stats


def run_step2(withdrawn, overwrite=False):
    logger = logging.getLogger()
    out_path = quantities_path(withdrawn)
    if os.path.exists(out_path) and not overwrite:
        raise FileExistsError(f"{out_path} exists; pass --overwrite to redo Step 2")

    q = compute_server_quantities(load_remaining_stats(withdrawn), settings.AGGREGATION_GAMMA)
    q["withdrawn"] = withdrawn

    logger.info(f"Step 2: withdrawn client {withdrawn}; remaining {q['clients']}; gamma {q['gamma']}")
    for k in q["clients"]:
        logger.info(f"  Client {k}: similarity {q['similarity_scores'][k]:+.4f} -> w_sim {q['w_sim'][k]:.4f}, "
                    f"w_data {q['w_data'][k]:.4f}, weight {q['weights'][k]:.4f}")
    logger.info(f"  Sum of weights: {sum(q['weights'].values()):.12f}")
    for level, sfx in LEVEL_KEYS.items():
        rows = q[f"mask_{sfx}"].numel()
        logger.info(f"  {level}: {q[f'active_rows_{sfx}']}/{rows} rows active, margin delta {q[f'delta_{sfx}']:.6g}")

    helpers.atomic_save(q, out_path)
    logger.info(f"Wrote {out_path}")
    return q


def main():
    parser = argparse.ArgumentParser(description="Step 2: server quantities")
    settings.add_common_args(parser)
    parser.add_argument("--withdraw", type=int, required=True, help="Withdrawn client id")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    settings.configure(results_dir=args.results_dir, fixture=args.fixture)
    helpers.setup_logging(settings.logs_dir(args.withdraw), "step2_server")
    try:
        run_step2(args.withdraw, overwrite=args.overwrite)
    except (FileExistsError, FileNotFoundError) as e:
        logging.getLogger().error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
