# unlearning/inventory.py
"""
Lists what a results folder holds and what the later steps will find there.
Reads only; writes one inventory.json into the unlearning output folder.

    python -m unlearning.inventory --fixture
    python -m unlearning.inventory --results-dir FedDSR_results
"""
import argparse
import glob
import logging
import os
import re

import torch

from unlearning import settings
from unlearning import helpers
from models import VQVAE2


def expected_files():
    """Name -> path of every file the unlearning steps may read."""
    train = settings.training_dir()
    files = {
        "best_model.pth": os.path.join(train, "best_model.pth"),
        "round_1/global_model.pth": os.path.join(train, "round_1", "global_model.pth"),
        "round_0_global_model.pth": os.path.join(train, "round_0_global_model.pth"),
    }
    for k in helpers.all_clients():
        files[f"best_decoder_client_{k}.pth"] = helpers.best_decoder_path(k)
    for k in helpers.all_clients():
        files[f"best_refined_decoder_client_{k}.pth"] = helpers.refined_decoder_path(k)
    return files


def saved_rounds():
    rounds = []
    for path in glob.glob(os.path.join(settings.training_dir(), "round_*", "global_model.pth")):
        m = re.search(r"round_(\d+)", path)
        if m:
            rounds.append(int(m.group(1)))
    return sorted(rounds)


def rows_differing(deployed_sd, best_sd):
    """Per level, how many codebook rows of the deployed file differ from best_model.pth."""
    out = {}
    for level, key in zip(("top", "bottom"), helpers.CODEBOOK_KEYS):
        a, b = deployed_sd[key], best_sd[key]
        out[level] = int((a != b).any(dim=1).sum())
    return out


def run_inventory():
    logger = logging.getLogger()
    files = {name: {"path": path, "exists": os.path.exists(path)} for name, path in expected_files().items()}

    deployed = {}
    for k in helpers.all_clients():
        path, source = helpers.choose_deployed(k)
        deployed[str(k)] = {"path": path, "source": source, "exists": os.path.exists(path)}

    step0_case = 1 if files["round_0_global_model.pth"]["exists"] else 2

    sizes, differing = None, {}
    if files["best_model.pth"]["exists"]:
        best_sd = helpers.load_file(files["best_model.pth"]["path"])
        model = VQVAE2()
        model.load_state_dict(best_sd, strict=False)
        sizes = helpers.model_sizes(model)
        for k, d in deployed.items():
            if d["exists"]:
                differing[k] = rows_differing(helpers.load_file(d["path"]), best_sd)

    inventory = {
        "results_dir": settings.RESULTS_DIR,
        "fixture": settings.FIXTURE,
        "files": files,
        "saved_rounds": saved_rounds(),
        "deployed": deployed,
        "step0_case": step0_case,
        "codebook_rows_differing_from_best_model": differing,
        "sizes": sizes,
    }

    # ---- Report ----
    logger.info(f"Inventory of {settings.RESULTS_DIR}")
    for name, f in files.items():
        logger.info(f"  [{'x' if f['exists'] else ' '}] {name:40s} {f['path']}")
    logger.info(f"  Saved rounds: {inventory['saved_rounds']}")
    for k, d in deployed.items():
        diff = differing.get(k)
        diff_txt = f"; codebook rows differing from best_model: top {diff['top']}, bottom {diff['bottom']}" if diff else ""
        logger.info(f"  Client {k} deployed: {d['source']} ({d['path']}){'' if d['exists'] else ' MISSING'}{diff_txt}")
    logger.info(f"  Step 0: Case {step0_case} "
                f"({'round-0 file present' if step0_case == 1 else 'regenerate from the seed and verify'})")
    if sizes:
        for level in ("top", "bottom"):
            s = sizes[level]
            logger.info(f"  Sizes {level}: codebook {s['codebook_rows']} x {s['latent_dim']}, grid {s['grid']}, "
                        f"decoder entry {s['decoder_in']}->{s['decoder_out']} k{s['kernel']}, patch {s['patch_length']}")
        logger.info(f"  Encoder weights: {sizes['encoder_weights']:,}")
    else:
        logger.warning("  best_model.pth missing: sizes not read")

    helpers.atomic_save_json(inventory, helpers.inventory_path())
    logger.info(f"Wrote {helpers.inventory_path()}")
    return inventory


def main():
    parser = argparse.ArgumentParser(description="Inventory of a trained model's results folder")
    settings.add_common_args(parser)
    args = parser.parse_args()
    settings.configure(results_dir=args.results_dir, fixture=args.fixture)
    helpers.setup_logging(settings.logs_dir(), "inventory")
    run_inventory()


if __name__ == "__main__":
    main()
