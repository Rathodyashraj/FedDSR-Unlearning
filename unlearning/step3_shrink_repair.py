# unlearning/step3_shrink_repair.py
"""
Step 3: shrink and repair.

Each round, every remaining client loads the global encoder, shrinks it toward the initial
weights phi0 and repairs it so its latents keep their target rows in the clean codebook.
The server then aggregates the client encoders with the Step 2 weights. A closing round of
repairs without shrinking gives the unlearned encoder.

    python -m unlearning.step3_shrink_repair --fixture --withdraw 3
    python -m unlearning.step3_shrink_repair --fixture --withdraw 3 --resume
"""
import argparse
import copy
import json
import logging
import os
import shutil
import sys

import torch
import torch.nn as nn

from unlearning import settings
from unlearning import helpers
from unlearning.step0_initial_weights import phi0_path
from unlearning.step2_server_quantities import quantities_path

LEVEL_SUFFIX = {"top": "top", "bottom": "bot"}  # level -> suffix used in server_quantities.pt
CLOSING = "closing"


# -----------------
# Building blocks
# -----------------
class WorkingEncoder(nn.Module):
    """The four encoder groups, copied from a model; same parameter names as in VQVAE2. No decoder."""

    def __init__(self, model):
        super().__init__()
        for name in helpers.ENCODER_MODULES:
            setattr(self, name, copy.deepcopy(getattr(model, name)))
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, lr_image):
        # Same operations as VQVAE2.encode
        z_top = self.pre_quant_conv_top(self.encoder_top(lr_image))
        z_bottom = self.pre_quant_conv_bottom(self.encoder_bottom(lr_image))
        return z_top, z_bottom


@torch.no_grad()
def shrink(encoder, phi0, factor=None):
    """w = w0 + factor * (w - w0) for every encoder weight, in place. Never touches codebooks."""
    factor = settings.SHRINK_FACTOR if factor is None else factor
    for name, p in encoder.named_parameters():
        w0 = phi0[name]
        p.copy_(w0 + factor * (p - w0))


def hinge(dist, target, delta):
    """
    h_p = max(0, d_{p,t} - min_{j != t} d_{p,j} + delta).
    dist: [P, K] squared distances with inactive rows already +inf; target: [P].
    """
    t = target.view(-1, 1)
    d_t = dist.gather(1, t).squeeze(1)
    d_other = dist.scatter(1, t, float("inf")).min(dim=1).values
    return torch.relu(d_t - d_other + delta)


def aggregate(states, weights):
    """Global encoder = sum_k w_k * encoder_k (accumulated in float64)."""
    out = {}
    for key in states[0]:
        acc = sum(w * s[key].double() for s, w in zip(states, weights))
        out[key] = acc.to(states[0][key].dtype)
    return out


def leftover(shrink_counts, weights):
    """rho = prod_r sum_k w_k * factor^{n_{k,r}} over the shrink rounds."""
    rho = 1.0
    rounds = len(next(iter(shrink_counts.values())))
    for r in range(rounds):
        rho *= sum(weights[k] * settings.SHRINK_FACTOR ** shrink_counts[k][r] for k in shrink_counts)
    return rho


def level_hinge_grad(z, target, V, mask, delta):
    """
    Hinge loss of one level, computed in blocks of positions. Returns the gradient with respect
    to the latents, the mean hinge, and the number of flipped positions (nearest active row != target).
    """
    z_flat = helpers.flatten_latent(z.detach()).requires_grad_(True)
    P = z_flat.shape[0]
    chunk = settings.DISTANCE_CHUNK_POSITIONS
    loss, flips = 0.0, 0
    for start in range(0, P, chunk):
        zc = z_flat[start:start + chunk]
        dist = helpers.squared_distances(zc, V).masked_fill(~mask, float("inf"))
        h = hinge(dist, target[start:start + chunk], delta)
        (h.sum() / P).backward()
        loss += float(h.detach().sum()) / P
        flips += int((dist.detach().argmin(dim=1) != target[start:start + chunk]).sum())
    # back to [B, D, H, W]
    B, D, H, W = z.shape
    grad = z_flat.grad.view(B, H, W, D).permute(0, 3, 1, 2)
    return grad, loss, flips, P


# -----------------
# One client
# -----------------
class ClientWorker:
    """A remaining client's working encoder, its Adam optimizer (kept for all rounds) and its data."""

    def __init__(self, k, ref, phi0, quantities, records):
        self.k = k
        self.ref = ref
        self.phi0 = phi0
        self.q = quantities
        self.records = records
        self.device = next(ref.parameters()).device
        self.encoder = WorkingEncoder(ref).to(self.device)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=settings.REPAIR_LR)
        self.loader = helpers.make_loader(k, "train", shuffle=True, drop_last=True)
        if len(self.loader) == 0:
            raise RuntimeError(f"Client {k}: fewer training slices than one batch")
        self.batches_per_pass = len(self.loader)
        self._iter = None
        self.round_label = None

    # ---- data: an endless cycle, restarted with a fixed seed at the start of each round ----
    def start_round(self, round_label, round_index):
        self.round_label = round_label
        self.loader.generator.manual_seed(settings.SEED + 1000 * self.k + round_index)
        self._iter = iter(self.loader)

    def next_batch(self):
        while True:
            try:
                batch = next(self._iter)
            except StopIteration:
                self._iter = iter(self.loader)
                continue
            if batch is not None:
                return batch["lr"].to(self.device)

    # ---- weights ----
    def load_encoder(self, state):
        with torch.no_grad():
            for name, p in self.encoder.named_parameters():
                p.copy_(state[name])

    def encoder_state(self):
        return {name: p.detach().clone() for name, p in self.encoder.named_parameters()}

    # ---- repair ----
    @torch.no_grad()
    def targets(self, lr):
        z = dict(zip(("top", "bottom"), self.ref.encode(lr)))
        old = {lvl: helpers.quantizer(self.ref, lvl)._embedding.weight for lvl in helpers.LEVELS}
        return {lvl: helpers.nearest_rows(helpers.flatten_latent(z[lvl]), old[lvl]) for lvl in helpers.LEVELS}

    def repair(self, shrink_number):
        """Repair until the flip rate is at most the threshold at both levels, or one pass is used up."""
        block = {lvl: [0, 0] for lvl in helpers.LEVELS}   # flips, positions
        block_batches, batches, loss_sum = 0, 0, 0.0
        checks, done, stalled = [], False, False
        cap = self.batches_per_pass * settings.REPAIR_CAP_PASSES
        while True:
            lr = self.next_batch()
            t = self.targets(lr)
            for lvl in helpers.LEVELS:
                mask = self.q[f"mask_{LEVEL_SUFFIX[lvl]}"]
                if not bool(mask[t[lvl]].all()):
                    raise AssertionError(f"Client {self.k} {lvl}: a target row is inactive")

            self.optimizer.zero_grad(set_to_none=True)
            z_top, z_bottom = self.encoder(lr)
            grads, batch_loss = [], 0.0
            for lvl, z in (("top", z_top), ("bottom", z_bottom)):
                sfx = LEVEL_SUFFIX[lvl]
                g, loss, flips, P = level_hinge_grad(z, t[lvl], self.q[f"V_{sfx}"], self.q[f"mask_{sfx}"],
                                                     self.q[f"delta_{sfx}"])
                grads.append(g)
                batch_loss += loss
                block[lvl][0] += flips
                block[lvl][1] += P
            torch.autograd.backward([z_top, z_bottom], grads)
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=settings.GRAD_CLIP)
            self.optimizer.step()

            batches += 1
            block_batches += 1
            loss_sum += batch_loss
            at_check = block_batches == settings.FLIP_CHECK_BATCHES
            at_cap = batches >= cap
            if at_check or at_cap:
                rates = {lvl: block[lvl][0] / block[lvl][1] for lvl in helpers.LEVELS}
                checks.append({"batches": batches, "flip_top": rates["top"], "flip_bottom": rates["bottom"]})
                if all(r <= settings.FLIP_THRESHOLD for r in rates.values()):
                    done = True
                    break
                if at_cap:
                    stalled = True
                    break
                block = {lvl: [0, 0] for lvl in helpers.LEVELS}
                block_batches = 0

        record = {"round": self.round_label, "client": self.k, "shrink": shrink_number, "batches": batches,
                  "cap": cap, "checks": checks, "mean_loss": loss_sum / batches, "done": done, "stalled": stalled}
        self.records.write(record)
        logging.getLogger().info(
            f"  Client {self.k} round {self.round_label} shrink {shrink_number}: {batches} batches, "
            f"last flip rate top {checks[-1]['flip_top']:.4%} bottom {checks[-1]['flip_bottom']:.4%}, "
            f"mean loss {record['mean_loss']:.4g} -> {'done' if done else 'STALLED'}")
        return record

    def run_round(self, round_number, global_state, shrinks, initial_repair, repair_enabled=True):
        """One shrink round at this client. Returns (encoder state, shrinks done, stalled)."""
        self.start_round(round_number, round_number)
        self.load_encoder(global_state)
        if initial_repair and repair_enabled and self.repair(shrink_number=0)["stalled"]:
            # The alignment repair stalled: no shrink on top of an unaligned encoder
            return self.encoder_state(), 0, True
        done_shrinks, stalled = 0, False
        for s in range(1, shrinks + 1):
            shrink(self.encoder, self.phi0)
            done_shrinks += 1
            if repair_enabled and self.repair(shrink_number=s)["stalled"]:
                stalled = True
                break
        return self.encoder_state(), done_shrinks, stalled

    def run_closing(self, global_state, closing_index, repair_enabled=True):
        self.start_round(CLOSING, closing_index)
        self.load_encoder(global_state)
        if repair_enabled:
            self.repair(shrink_number=0)
        return self.encoder_state()


# -----------------
# Saving and resuming
# -----------------
def round_folder(out_dir, label):
    return os.path.join(out_dir, f"round_{label}")


def save_round(out_dir, label, global_state, workers, state):
    folder = round_folder(out_dir, label)
    helpers.atomic_save(global_state, os.path.join(folder, "global_encoder.pth"))
    for k, wk in workers.items():
        helpers.atomic_save(wk.optimizer.state_dict(), os.path.join(folder, f"optimizer_client_{k}.pt"))
    helpers.atomic_save_json(state, os.path.join(folder, "state.json"))  # written last: marks the round complete


def last_complete_round(out_dir):
    """The state of the last round whose state.json exists (the closing round counts as last)."""
    if not os.path.isdir(out_dir):
        return None
    states = []
    for name in os.listdir(out_dir):
        path = os.path.join(out_dir, name, "state.json")
        if name.startswith("round_") and os.path.exists(path):
            with open(path) as f:
                states.append(json.load(f))
    if not states:
        return None
    return max(states, key=lambda s: s["order"])


def load_round(out_dir, state, workers, device):
    folder = round_folder(out_dir, state["label"])
    global_state = {k: v.to(device) for k, v in helpers.load_file(os.path.join(folder, "global_encoder.pth")).items()}
    for k, wk in workers.items():
        wk.optimizer.load_state_dict(helpers.load_file(os.path.join(folder, f"optimizer_client_{k}.pt"), device))
    return global_state


# -----------------
# Final evaluation
# -----------------
@torch.no_grad()
def final_flip_rates(ref, encoder_state, quantities, k):
    """Run client k's full training split through the unlearned encoder; flip rate per level."""
    enc = WorkingEncoder(ref).to(next(ref.parameters()).device)
    for name, p in enc.named_parameters():
        p.copy_(encoder_state[name])
    flips = {lvl: 0 for lvl in helpers.LEVELS}
    total = {lvl: 0 for lvl in helpers.LEVELS}
    old = {lvl: helpers.quantizer(ref, lvl)._embedding.weight for lvl in helpers.LEVELS}
    for batch in helpers.make_loader(k, "train", shuffle=False, drop_last=False):
        if batch is None:
            continue
        lr = batch["lr"].to(next(ref.parameters()).device)
        z_old = dict(zip(("top", "bottom"), ref.encode(lr)))
        z_new = dict(zip(("top", "bottom"), enc(lr)))
        for lvl in helpers.LEVELS:
            sfx = LEVEL_SUFFIX[lvl]
            t = helpers.nearest_rows(helpers.flatten_latent(z_old[lvl]), old[lvl])
            got = helpers.nearest_rows(helpers.flatten_latent(z_new[lvl]), quantities[f"V_{sfx}"], quantities[f"mask_{sfx}"])
            flips[lvl] += int((got != t).sum())
            total[lvl] += t.numel()
    return {lvl: flips[lvl] / total[lvl] for lvl in helpers.LEVELS}


# -----------------
# Step 3
# -----------------
def load_inputs(withdrawn, device):
    for path, what in ((phi0_path(), "Step 0"), (quantities_path(withdrawn), "Step 2")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing: run {what} first")
    ref = helpers.load_reference_model(device)
    phi0 = {k: v.to(device) for k, v in helpers.load_file(phi0_path()).items()}
    q = helpers.load_file(quantities_path(withdrawn))
    quantities = {}
    for sfx in LEVEL_SUFFIX.values():
        quantities[f"V_{sfx}"] = q[f"V_{sfx}"].to(device)
        quantities[f"mask_{sfx}"] = q[f"mask_{sfx}"].to(device)
        quantities[f"delta_{sfx}"] = float(q[f"delta_{sfx}"])
    weights = {int(k): float(v) for k, v in q["weights"].items()}
    missing = set(helpers.encoder_state(dict(ref.named_parameters()))) ^ set(phi0)
    if missing:
        raise KeyError(f"phi0.pth does not match the encoder: {sorted(missing)[:3]}")
    return ref, phi0, quantities, weights


def run_step3(withdrawn, out_dir=None, resume=False, overwrite=False, repair_enabled=True,
              shrink_rounds=None, shrinks_per_round=None, stop_after_round=None):
    """
    Runs the round loop. Test hooks: repair_enabled=False skips every repair; shrink_rounds and
    shrinks_per_round override the schedule; stop_after_round stops after saving that round.
    """
    logger = logging.getLogger()
    helpers.set_withdrawn(withdrawn)
    out_dir = out_dir or settings.step_dir(withdrawn, 3)
    rounds = settings.SHRINK_ROUNDS if shrink_rounds is None else shrink_rounds
    per_round = settings.SHRINKS_PER_ROUND if shrinks_per_round is None else shrinks_per_round
    final_path = os.path.join(out_dir, "encoder_unlearned.pth")

    previous = last_complete_round(out_dir)
    if previous is not None and not resume:
        if not overwrite:
            raise FileExistsError(f"{out_dir} holds an earlier run; pass --resume to continue it or --overwrite")
        helpers.assert_writable(out_dir)
        shutil.rmtree(out_dir)
        previous = None

    device = settings.DEVICE
    ref, phi0, quantities, weights = load_inputs(withdrawn, device)
    clients = helpers.remaining_clients(withdrawn)
    if sorted(weights) != clients:
        raise ValueError(f"weights are for {sorted(weights)}, remaining clients are {clients}")
    records = helpers.JsonlWriter(os.path.join(settings.logs_dir(withdrawn), "repairs.jsonl"))
    workers = {k: ClientWorker(k, ref, phi0, quantities, records) for k in clients}
    w_list = [weights[k] for k in clients]

    global_state = {name: p.detach().clone() for name, p in ref.named_parameters() if helpers.is_encoder_key(name)}
    n = {k: [] for k in clients}           # shrinks done per client per round
    stalled = {k: [] for k in clients}     # stall flag per client per round
    next_round, closing_done = 1, False
    if previous is not None:
        global_state = load_round(out_dir, previous, workers, device)
        n = {int(k): v for k, v in previous["n"].items()}
        stalled = {int(k): v for k, v in previous["stalled"].items()}
        closing_done = previous["label"] == CLOSING
        next_round = previous.get("round", 0) + 1
        logger.info(f"Resuming after round {previous['label']}")

    logger.info(f"Step 3: withdrawn client {withdrawn}; clients {clients}; weights "
                + ", ".join(f"{k}: {weights[k]:.4f}" for k in clients)
                + f"; {rounds} rounds x {per_round} shrinks; delta top {quantities['delta_top']:.4g}, "
                  f"bottom {quantities['delta_bot']:.4g}")

    any_stall = any(any(v) for v in stalled.values())
    if not closing_done:
        for r in range(next_round, rounds + 1):
            if any_stall:
                break
            logger.info(f"--- Round {r}/{rounds} ---")
            states = []
            for k in clients:
                state_k, done_k, stall_k = workers[k].run_round(r, global_state, per_round, initial_repair=(r == 1),
                                                                repair_enabled=repair_enabled)
                states.append(state_k)
                n[k].append(done_k)
                stalled[k].append(stall_k)
            global_state = aggregate(states, w_list)
            any_stall = any(stalled[k][-1] for k in clients)
            save_round(out_dir, r, global_state, workers,
                       {"label": r, "round": r, "order": r, "n": n, "stalled": stalled, "any_stall": any_stall})
            logger.info(f"Round {r} aggregated; shrinks " + ", ".join(f"{k}: {n[k][-1]}" for k in clients)
                        + ("; a client stalled: skipping to the closing round" if any_stall else ""))
            if stop_after_round == r:
                return None

        logger.info("--- Closing round ---")
        states = [workers[k].run_closing(global_state, closing_index=100, repair_enabled=repair_enabled) for k in clients]
        global_state = aggregate(states, w_list)
        rounds_done = len(next(iter(n.values())))
        save_round(out_dir, CLOSING, global_state, workers,
                   {"label": CLOSING, "round": rounds_done, "order": 10 ** 6, "n": n, "stalled": stalled,
                    "any_stall": any_stall})

    # ---- final flips, leftover, outputs ----
    rho = leftover(n, weights)
    flip = {k: final_flip_rates(ref, global_state, quantities, k) for k in clients}
    for k in clients:
        logger.info(f"Final flip rate client {k}: top {flip[k]['top']:.4%}, bottom {flip[k]['bottom']:.4%}")
    total_shrinks = {k: sum(n[k]) for k in clients}
    logger.info(f"Leftover rho = {rho:.6g} (shrinks done per client: {total_shrinks})")

    summary = {
        "withdrawn": withdrawn, "clients": clients, "weights": weights,
        "schedule": {"rounds": rounds, "shrinks_per_round": per_round, "factor": settings.SHRINK_FACTOR,
                     "flip_threshold": settings.FLIP_THRESHOLD, "check_batches": settings.FLIP_CHECK_BATCHES,
                     "cap_passes": settings.REPAIR_CAP_PASSES, "repairs_enabled": repair_enabled},
        "delta": {"top": quantities["delta_top"], "bottom": quantities["delta_bot"]},
        "shrinks": n, "stalled": stalled, "stopped_early": any_stall,
        "shrink_rounds_done": len(next(iter(n.values()))),
        "leftover_rho": rho, "final_flip_rates": flip,
    }
    helpers.atomic_save({k: v.detach().cpu() for k, v in global_state.items()}, final_path)
    helpers.atomic_save_json(summary, os.path.join(out_dir, "step3_summary.json"))
    logger.info(f"Wrote {final_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Step 3: shrink and repair")
    settings.add_common_args(parser)
    parser.add_argument("--withdraw", type=int, required=True, help="Withdrawn client id")
    parser.add_argument("--trial", action="store_true", help="Real model, first slices only, short schedule")
    parser.add_argument("--resume", action="store_true", help="Continue from the last finished round")
    parser.add_argument("--overwrite", action="store_true", help="Discard an earlier Step 3 run")
    args = parser.parse_args()
    settings.configure(results_dir=args.results_dir, fixture=args.fixture, trial=args.trial)
    helpers.setup_logging(settings.logs_dir(args.withdraw), "step3_shrink_repair")
    try:
        run_step3(args.withdraw, resume=args.resume, overwrite=args.overwrite)
    except (FileExistsError, FileNotFoundError) as e:
        logging.getLogger().error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
