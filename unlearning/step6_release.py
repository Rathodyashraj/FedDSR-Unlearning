# unlearning/step6_release.py
"""
Step 6: release the unlearned model, and list (or, with --purge, delete) what still holds the old model.

release/ uses the training code's layout, so existing loading code works on it unchanged:
  best_model.pth            shared part (unlearned encoder, clean codebooks) plus the two mask buffers
  unlearned_client_k.pth    full fine-tuned model of each remaining client
  phi0.pth                  initial encoder weights (no data; needed for any future withdrawal)
  summary.json

Clean-up is off by default: the files are only listed. --purge deletes inside the unlearning
output folder; --purge-training-files also deletes in the training folders.

    python -m unlearning.step6_release --fixture --withdraw 3
"""
import argparse
import glob
import json
import logging
import os
import shutil
import sys

import torch

from unlearning import settings
from unlearning import helpers
from unlearning.step0_initial_weights import phi0_path
from unlearning.step2_server_quantities import quantities_path
from unlearning.step3_shrink_repair import LEVEL_SUFFIX
from unlearning.step4_decoder_fit import unlearned_encoder_path
from unlearning.step5_finetune import finetuned_path


def release_dir(withdrawn):
    return settings.step_dir(withdrawn, 6)


def released_client_path(withdrawn, k):
    return os.path.join(release_dir(withdrawn), f"unlearned_client_{k}.pth")


# -----------------
# Release
# -----------------
def build_shared(withdrawn):
    """best_model.pth content: unlearned encoder, clean codebooks, and the two masks."""
    shared = dict(helpers.load_file(unlearned_encoder_path(withdrawn)))
    q = helpers.load_file(quantities_path(withdrawn))
    for (level, sfx), cb_key, mask_key in zip(LEVEL_SUFFIX.items(), helpers.CODEBOOK_KEYS, helpers.MASK_KEYS):
        shared[cb_key] = q[f"V_{sfx}"].clone()
        shared[mask_key] = q[f"mask_{sfx}"].clone()
    return shared, q


def release(withdrawn, overwrite=False):
    logger = logging.getLogger()
    helpers.set_withdrawn(withdrawn)
    clients = helpers.remaining_clients(withdrawn)
    out = release_dir(withdrawn)
    if os.path.exists(os.path.join(out, "summary.json")) and not overwrite:
        raise FileExistsError(f"{out} already holds a release; pass --overwrite to redo it")
    step3_summary_path = os.path.join(settings.step_dir(withdrawn, 3), "step3_summary.json")
    needed = [phi0_path(), quantities_path(withdrawn), unlearned_encoder_path(withdrawn), step3_summary_path] + \
             [finetuned_path(withdrawn, k) for k in clients]
    missing = [p for p in needed if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{missing[0]} missing: run the earlier steps first")

    shared, q = build_shared(withdrawn)
    files = {}
    for k in clients:
        sd = helpers.load_file(finetuned_path(withdrawn, k))
        mismatch = [key for key in shared if not torch.equal(sd[key], shared[key])]
        if mismatch:
            raise AssertionError(f"Client {k}: fine-tuned model differs from the shared unlearned part: {mismatch[:3]}")
        helpers.atomic_save(sd, released_client_path(withdrawn, k))
        files[f"unlearned_client_{k}.pth"] = released_client_path(withdrawn, k)
    helpers.atomic_save(shared, os.path.join(out, "best_model.pth"))
    helpers.atomic_save(helpers.load_file(phi0_path()), os.path.join(out, "phi0.pth"))
    files["best_model.pth"] = os.path.join(out, "best_model.pth")
    files["phi0.pth"] = os.path.join(out, "phi0.pth")

    with open(step3_summary_path) as f:
        s3 = json.load(f)
    summary = {
        "withdrawn": withdrawn,
        "withdrawn_name": settings.client_name(withdrawn),
        "remaining_clients": clients,
        "weights": {int(k): v for k, v in q["weights"].items()},
        "margins": {"top": q["delta_top"], "bottom": q["delta_bot"]},
        "active_rows": {"top": int(q["mask_top"].sum()), "bottom": int(q["mask_bot"].sum()),
                        "of": int(q["mask_top"].numel())},
        "shrinks_done": {int(k): v for k, v in s3["shrinks"].items()},
        "stalled": {int(k): v for k, v in s3["stalled"].items()},
        "stopped_early": s3["stopped_early"],
        "leftover_rho": s3["leftover_rho"],
        "final_flip_rates": {int(k): v for k, v in s3["final_flip_rates"].items()},
        "finetune": {"epochs": settings.FINETUNE_EPOCHS, "lr": settings.FINETUNE_LR},
        "fixture": settings.FIXTURE,
        "files": sorted(files),
    }
    helpers.atomic_save_json(summary, os.path.join(out, "summary.json"))
    logger.info(f"Step 6: released {sorted(files)} to {out}")
    logger.info(f"  Withdrawn client {withdrawn} ({summary['withdrawn_name']}): no file written for it")
    logger.info(f"  Active rows: top {summary['active_rows']['top']}, bottom {summary['active_rows']['bottom']} "
                f"of {summary['active_rows']['of']}; leftover rho {summary['leftover_rho']:.4g}")
    return summary


# -----------------
# Clean-up
# -----------------
def cleanup_list(withdrawn):
    """
    Everything that still holds the old model or data-derived intermediates.
    'unlearning': inside this withdrawal's output folder; 'training': in the training folders.
    """
    w_dir = settings.withdraw_dir(withdrawn)
    unlearning = []
    for step in (1, 2):
        unlearning.append(settings.step_dir(withdrawn, step))                      # statistics
    unlearning += sorted(glob.glob(os.path.join(settings.step_dir(withdrawn, 3), "round_*")))  # round encoders, optimizer states
    training = []
    train = settings.training_dir()
    training.append(os.path.join(train, "best_model.pth"))
    training += sorted(glob.glob(os.path.join(train, "best_decoder_client_*.pth")))
    training += sorted(glob.glob(os.path.join(train, "round_*" + os.sep)))       # global models and sample images
    stage2 = settings.stage2_dir()
    training += sorted(glob.glob(os.path.join(stage2, "best_refined_decoder_client_*.pth")))
    training += sorted(glob.glob(os.path.join(stage2, "*.png")))                   # sample images
    unlearning = [p for p in unlearning if os.path.exists(p) and os.path.realpath(p).startswith(os.path.realpath(w_dir))]
    training = [p.rstrip(os.sep) for p in training if os.path.exists(p)]
    return {"unlearning": unlearning, "training": training}


def purge(withdrawn, include_training=False):
    logger = logging.getLogger()
    items = cleanup_list(withdrawn)
    targets = items["unlearning"] + (items["training"] if include_training else [])
    for path in targets:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        logger.info(f"  deleted {path}")
    return targets


def report_cleanup(withdrawn):
    logger = logging.getLogger()
    items = cleanup_list(withdrawn)
    logger.info("Clean-up (not performed; --purge deletes the first list, --purge-training-files also the second):")
    logger.info(f"  In the unlearning output folder ({len(items['unlearning'])}):")
    for p in items["unlearning"]:
        logger.info(f"    {p}")
    logger.info(f"  In the training folders ({len(items['training'])}):")
    for p in items["training"]:
        logger.info(f"    {p}")
    logger.info("  Kept: round_0_global_model.pth and phi0.pth (no data), metrics, logs, and release/")
    return items


def main():
    parser = argparse.ArgumentParser(description="Step 6: release the unlearned model; optional clean-up")
    settings.add_common_args(parser)
    parser.add_argument("--withdraw", type=int, required=True, help="Withdrawn client id")
    parser.add_argument("--overwrite", action="store_true", help="Redo an existing release")
    parser.add_argument("--purge", action="store_true",
                        help="Delete the statistics and per-round states in the unlearning output folder")
    parser.add_argument("--purge-training-files", action="store_true",
                        help="With --purge: also delete the old model files and sample images in the training folders")
    args = parser.parse_args()
    if args.purge_training_files and not args.purge:
        parser.error("--purge-training-files needs --purge")
    settings.configure(results_dir=args.results_dir, fixture=args.fixture)
    helpers.setup_logging(settings.logs_dir(args.withdraw), "step6_release")
    try:
        release(args.withdraw, overwrite=args.overwrite)
    except FileExistsError as e:
        logging.getLogger().info(f"{e}; keeping it")
    except FileNotFoundError as e:
        logging.getLogger().error(str(e))
        sys.exit(1)
    report_cleanup(args.withdraw)
    if args.purge:
        purge(args.withdraw, include_training=args.purge_training_files)


if __name__ == "__main__":
    main()
