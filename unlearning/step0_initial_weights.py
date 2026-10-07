# unlearning/step0_initial_weights.py
"""
Step 0: the encoder weights exactly as they were before round 1 (phi0.pth).

Case 1: round_0_global_model.pth exists -> keep its encoder part.
Case 2: it does not -> regenerate the initial weights from the seed by repeating the start
        of main.py, then verify them against round_1/global_model.pth.

Runs once per trained model, not once per withdrawn client.

    python -m unlearning.step0_initial_weights --fixture
    python -m unlearning.step0_initial_weights --results-dir FedDSR_results
"""
import argparse
import logging
import math
import os
import sys

import numpy as np
import torch

import config
from unlearning import settings
from unlearning import helpers
from models import VQVAE2


def round0_path():
    return os.path.join(settings.training_dir(), "round_0_global_model.pth")


def round1_path():
    return os.path.join(settings.training_dir(), "round_1", "global_model.pth")


def phi0_path():
    return os.path.join(settings.phi0_dir(), "phi0.pth")


# -----------------
# Case 2: regeneration
# -----------------
def _client_loaders(i, use_data):
    """The loaders main.py passes to Client. Only their length is used at construction time."""
    from data import get_dataloaders
    if use_data:
        try:
            return get_dataloaders(i)  # lists file names only; no image is read
        except RuntimeError as e:
            logging.getLogger().warning(f"Client {i}: {e} Using a stand-in loader (consumes no random draws).")
    stand_in = [None]
    return stand_in, stand_in, stand_in


def regenerate_initial_shared(use_data=True):
    """
    Repeat the start of main.py line for line: seeds, the four Clients in order, then Server.
    Nothing is trained and nothing is written. Returns server.global_shared_params.
    """
    from client import Client
    from server import Server
    from utils import ResultsLogger

    # Built on CPU: every random draw happens on the CPU generator either way, and CPU avoids GPU memory.
    device = torch.device("cpu")
    torch.manual_seed(settings.TRAINING_SEED); np.random.seed(settings.TRAINING_SEED)
    results_logger = ResultsLogger(settings.training_dir())  # never saved
    clients = []
    for i in helpers.all_clients():
        train_loader, val_loader, test_loader = _client_loaders(i, use_data)
        clients.append(Client(client_id=i, train_loader=train_loader, val_loader=val_loader,
                              test_loader=test_loader, results_logger=results_logger, device=device))
    server = Server(clients, results_logger)
    shared = server.global_shared_params
    del server, clients
    return shared


def control_initial_shared(seed=None):
    """A control initialization from a different seed."""
    torch.manual_seed(settings.STEP0_CONTROL_SEED if seed is None else seed)
    return VQVAE2().get_shared_state_dict()


# -----------------
# Verification
# -----------------
def flat_encoder(state_dict):
    enc = helpers.encoder_state(state_dict)
    return torch.cat([enc[k].reshape(-1).double() for k in sorted(enc)])


def correlation_score(a_state, b_state):
    """Pearson r over all encoder weights, and r * sqrt(N)."""
    a, b = flat_encoder(a_state), flat_encoder(b_state)
    if a.numel() != b.numel():
        raise ValueError(f"encoder sizes differ: {a.numel()} vs {b.numel()}")
    a, b = a - a.mean(), b - b.mean()
    r = float((a @ b) / (a.norm() * b.norm()))
    n = a.numel()
    return {"r": r, "score": r * math.sqrt(n), "n": n}


def verify(candidate, round1, control):
    cand = correlation_score(candidate, round1)
    ctrl = correlation_score(control, round1)
    passed = cand["score"] > settings.STEP0_MIN_SCORE and abs(ctrl["score"]) < settings.STEP0_CONTROL_MAX_SCORE
    return {
        "n_weights": cand["n"],
        "regenerated_r": cand["r"], "regenerated_score": cand["score"],
        "control_r": ctrl["r"], "control_score": ctrl["score"],
        "control_seed": settings.STEP0_CONTROL_SEED,
        "min_score": settings.STEP0_MIN_SCORE, "control_max_score": settings.STEP0_CONTROL_MAX_SCORE,
        "passed": passed,
    }


# -----------------
# Step 0
# -----------------
def run_step0(force_case2=False, overwrite=False, use_data=True):
    logger = logging.getLogger()
    if os.path.exists(phi0_path()) and not overwrite:
        raise FileExistsError(f"{phi0_path()} already exists; pass --overwrite to redo Step 0")

    case = 1 if os.path.exists(round0_path()) and not force_case2 else 2
    info = {"case": case, "results_dir": settings.RESULTS_DIR}

    if case == 1:
        logger.info(f"Step 0 Case 1: loading {round0_path()}")
        shared = helpers.load_file(round0_path())
        info["source"] = round0_path()
    else:
        logger.info("Step 0 Case 2: regenerating the initial weights from the seed (nothing is trained or written)")
        shared = regenerate_initial_shared(use_data=use_data)
        info["source"] = f"regenerated (torch.manual_seed({settings.TRAINING_SEED}), clients {helpers.all_clients()}, then Server)"
        if os.path.exists(round0_path()):
            saved = helpers.encoder_state(helpers.load_file(round0_path()))
            regen = helpers.encoder_state(shared)
            info["equals_saved_round0"] = (set(saved) == set(regen)
                                           and all(torch.equal(saved[k], regen[k]) for k in saved))
            logger.info(f"Regenerated vs saved round-0 file: bit-for-bit equal = {info['equals_saved_round0']}")

    phi0 = {k: v.clone() for k, v in helpers.encoder_state(shared).items()}
    if not phi0:
        raise KeyError("no encoder weights found")
    info["n_weights"] = sum(v.numel() for v in phi0.values())
    info["n_tensors"] = len(phi0)

    if os.path.exists(round1_path()):
        round1 = helpers.load_file(round1_path())
        info["verification"] = verify(phi0, round1, control_initial_shared())
        v = info["verification"]
        logger.info(f"Against round 1 over {v['n_weights']:,} weights: "
                    f"regenerated r={v['regenerated_r']:.4f}, r*sqrt(N)={v['regenerated_score']:.1f}; "
                    f"control (seed {v['control_seed']}) r={v['control_r']:.5f}, r*sqrt(N)={v['control_score']:.2f}; "
                    f"pass rule: > {v['min_score']} and |control| < {v['control_max_score']} -> "
                    f"{'PASS' if v['passed'] else 'FAIL'}")
    else:
        info["verification"] = None
        logger.warning(f"{round1_path()} not found: no verification possible")

    if case == 2 and not (info["verification"] and info["verification"]["passed"]):
        helpers.atomic_save_json(info, os.path.join(settings.phi0_dir(), "phi0_failed_verification.json"))
        raise RuntimeError("Step 0 Case 2 verification failed: phi0.pth NOT written. "
                           "Do not continue with a wrong anchor (see Open decisions in the plan).")

    helpers.atomic_save(phi0, phi0_path())
    helpers.atomic_save_json(info, os.path.join(settings.phi0_dir(), "phi0_info.json"))
    logger.info(f"Wrote {phi0_path()} ({info['n_tensors']} tensors, {info['n_weights']:,} weights)")
    return phi0, info


def main():
    parser = argparse.ArgumentParser(description="Step 0: initial encoder weights (phi0.pth)")
    settings.add_common_args(parser)
    parser.add_argument("--force-case2", action="store_true",
                        help="Regenerate from the seed even if the round-0 file exists (testing)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing phi0.pth")
    args = parser.parse_args()
    settings.configure(results_dir=args.results_dir, fixture=args.fixture)
    helpers.setup_logging(settings.logs_dir(), "step0_phi0")
    try:
        run_step0(force_case2=args.force_case2, overwrite=args.overwrite)
    except (RuntimeError, FileExistsError) as e:
        logging.getLogger().error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
