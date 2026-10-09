# unlearning/step4_decoder_fit.py
"""
Step 4: closed-form refit of the two decoder entry layers at each remaining client.

The decoder reads the codes only through decoder.pre_conv_top and decoder.pre_conv_bottom.
For each of them, W (kernel and bias as one matrix) is refit so the new code vectors give the
features the old ones gave:

    W' = W (Sigma_on + lambda I) (Sigma_nn + lambda I)^-1,   lambda = scale * tr(Sigma_nn) / patch_length

    python -m unlearning.step4_decoder_fit --fixture --withdraw 3
"""
import argparse
import logging
import os
import sys

import torch
import torch.nn.functional as F

from unlearning import settings
from unlearning import helpers
from unlearning.step2_server_quantities import quantities_path
from unlearning.step3_shrink_repair import LEVEL_SUFFIX, WorkingEncoder

def fit_path(withdrawn, k):
    return os.path.join(settings.step_dir(withdrawn, 4), f"fit_client_{k}.pth")


def unlearned_encoder_path(withdrawn):
    return os.path.join(settings.step_dir(withdrawn, 3), "encoder_unlearned.pth")


# -----------------
# Layer as a matrix
# -----------------
def layer_matrix(conv):
    """[out, in*kh*kw + 1]: the kernel flattened in unfold order, with the bias as the last column."""
    W = conv.weight.detach().reshape(conv.out_channels, -1)
    b = conv.bias.detach().view(-1, 1)
    return torch.cat([W, b], dim=1)


def set_layer_from_matrix(conv, W):
    with torch.no_grad():
        conv.weight.copy_(W[:, :-1].reshape(conv.weight.shape).to(conv.weight.dtype))
        conv.bias.copy_(W[:, -1].to(conv.bias.dtype))


def patches(code_map, conv):
    """3x3 patches of one slice [1, D, H, W] in the conv's own order, plus a constant 1: [H*W, D*kh*kw + 1]."""
    cols = F.unfold(code_map, kernel_size=conv.kernel_size, padding=conv.padding,
                    stride=conv.stride, dilation=conv.dilation)[0].t()
    return torch.cat([cols, torch.ones(cols.shape[0], 1, dtype=cols.dtype, device=cols.device)], dim=1)


def check_layer_supported(conv):
    if conv.stride != (1, 1) or conv.groups != 1 or conv.bias is None:
        raise ValueError(f"decoder entry conv must have stride 1, groups 1 and a bias; got {conv}")


# -----------------
# Moments and the solve
# -----------------
class PatchMoments:
    """Running sums of n^T n, o^T n and o^T o (float64) over the sampled patches of one level."""

    def __init__(self, length, device):
        self.S_nn = torch.zeros(length, length, dtype=torch.float64, device=device)
        self.S_on = torch.zeros(length, length, dtype=torch.float64, device=device)
        self.S_oo = torch.zeros(length, length, dtype=torch.float64, device=device)
        self.count = 0

    def add(self, old_map, new_map, conv, generator):
        """old_map, new_map: [B, D, H, W] code-vector maps. The same random positions are taken from both."""
        for i in range(old_map.shape[0]):
            o = patches(old_map[i:i + 1], conv)
            n = patches(new_map[i:i + 1], conv)
            keep = max(1, int(round(settings.DECODER_FIT_PATCH_FRACTION * o.shape[0])))
            pick = torch.randperm(o.shape[0], generator=generator)[:keep].to(o.device)
            o, n = o[pick].double(), n[pick].double()
            self.S_nn += n.t() @ n
            self.S_on += o.t() @ n
            self.S_oo += o.t() @ o
            self.count += keep

    def solve(self, W):
        """W' = W (Sigma_on + lambda I)(Sigma_nn + lambda I)^-1, without forming an inverse."""
        L = self.S_nn.shape[0]
        sig_nn, sig_on = self.S_nn / self.count, self.S_on / self.count
        lam = settings.DECODER_FIT_RIDGE_SCALE * float(torch.trace(sig_nn)) / L
        eye = torch.eye(L, dtype=torch.float64, device=sig_nn.device)
        W64 = W.double().to(sig_nn.device)
        # (Sigma_nn + lam I) is symmetric, so W' = [ (Sigma_nn + lam I)^-1 (Sigma_on + lam I)^T W^T ]^T
        rhs = (sig_on + lam * eye).t() @ W64.t()
        W_new = torch.linalg.solve(sig_nn + lam * eye, rhs).t()
        return W_new, lam

    def residual(self, A, W):
        """E||A n - W o||^2 / E||W o||^2 over the sampled patches (A = W before the fit, W' after)."""
        A, W = A.double().to(self.S_nn.device), W.double().to(self.S_nn.device)
        sig_nn, sig_on, sig_oo = self.S_nn / self.count, self.S_on / self.count, self.S_oo / self.count
        target = torch.trace(W @ sig_oo @ W.t())
        err = torch.trace(A @ sig_nn @ A.t()) - 2 * torch.trace(A @ sig_on.t() @ W.t()) + target
        return float(err / target)


# -----------------
# One client
# -----------------
def code_map(z, codebook, mask=None):
    """Nearest (active) codebook row at every position, as a [B, D, H, W] map of code vectors."""
    B, D, H, W = z.shape
    idx = helpers.nearest_rows(helpers.flatten_latent(z), codebook, mask)
    return codebook[idx].view(B, H, W, D).permute(0, 3, 1, 2)


@torch.no_grad()
def fit_client(k, ref, unlearned, quantities):
    """Refit client k's decoder entry layers. Returns (model with the new layers, report)."""
    device = next(ref.parameters()).device
    model = helpers.load_deployed_model(k, device)
    model.eval()
    deployed_cb = {lvl: helpers.quantizer(model, lvl)._embedding.weight.detach() for lvl in helpers.LEVELS}
    deployed_mask = {lvl: helpers.quantizer(model, lvl).active_mask for lvl in helpers.LEVELS}
    convs = {lvl: helpers.decoder_entry_conv(model, lvl) for lvl in helpers.LEVELS}
    for conv in convs.values():
        check_layer_supported(conv)
    moments = {lvl: PatchMoments(layer_matrix(convs[lvl]).shape[1], device) for lvl in helpers.LEVELS}
    generator = torch.Generator().manual_seed(settings.SEED + 1000 * k + 4)

    for batch in helpers.make_loader(k, "train", shuffle=False, drop_last=False):
        if batch is None:
            continue
        lr = batch["lr"].to(device)
        z_old = dict(zip(("top", "bottom"), ref.encode(lr)))
        z_new = dict(zip(("top", "bottom"), unlearned(lr)))
        for lvl in helpers.LEVELS:
            sfx = LEVEL_SUFFIX[lvl]
            old_map = code_map(z_old[lvl], deployed_cb[lvl], deployed_mask[lvl])
            new_map = code_map(z_new[lvl], quantities[f"V_{sfx}"], quantities[f"mask_{sfx}"])
            moments[lvl].add(old_map, new_map, convs[lvl], generator)

    report = {}
    for lvl in helpers.LEVELS:
        W = layer_matrix(convs[lvl])
        W_new, lam = moments[lvl].solve(W)
        report[lvl] = {
            "patches": moments[lvl].count,
            "patch_length": W.shape[1],
            "lambda": lam,
            "residual_before": moments[lvl].residual(W, W),
            "residual_after": moments[lvl].residual(W_new, W),
            "relative_change": float((W_new - W.double()).norm() / W.double().norm()),
        }
        set_layer_from_matrix(convs[lvl], W_new)

    # Unlearned encoder, clean codebooks and masks; usage counters of the old codebook are cleared
    unlearned_params = dict(unlearned.named_parameters())
    with torch.no_grad():
        for name, p in model.named_parameters():
            if helpers.is_encoder_key(name):
                p.copy_(unlearned_params[name])
        for lvl in helpers.LEVELS:
            q = helpers.quantizer(model, lvl)
            sfx = LEVEL_SUFFIX[lvl]
            q._embedding.weight.copy_(quantities[f"V_{sfx}"])
            q.active_mask.copy_(quantities[f"mask_{sfx}"])
            q.code_usage.zero_()
            q.batch_counter.zero_()
    return model, report


def run_step4(withdrawn, overwrite=False):
    logger = logging.getLogger()
    helpers.set_withdrawn(withdrawn)
    clients = helpers.remaining_clients(withdrawn)
    if not overwrite:
        existing = [fit_path(withdrawn, k) for k in clients if os.path.exists(fit_path(withdrawn, k))]
        if existing:
            raise FileExistsError(f"Step 4 output exists ({existing[0]}); pass --overwrite to redo it")
    for path, what in ((unlearned_encoder_path(withdrawn), "Step 3"), (quantities_path(withdrawn), "Step 2")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing: run {what} first")

    device = settings.DEVICE
    ref = helpers.load_reference_model(device)
    unlearned = WorkingEncoder(ref).to(device).eval()
    enc_state = helpers.load_file(unlearned_encoder_path(withdrawn), device)
    with torch.no_grad():
        for name, p in unlearned.named_parameters():
            p.copy_(enc_state[name])
            p.requires_grad = False
    q = helpers.load_file(quantities_path(withdrawn))
    quantities = {}
    for sfx in LEVEL_SUFFIX.values():
        quantities[f"V_{sfx}"] = q[f"V_{sfx}"].to(device)
        quantities[f"mask_{sfx}"] = q[f"mask_{sfx}"].to(device)

    logger.info(f"Step 4: withdrawn client {withdrawn}; decoder fit at clients {clients}")
    summary = {"withdrawn": withdrawn, "clients": clients, "patch_fraction": settings.DECODER_FIT_PATCH_FRACTION,
               "ridge_scale": settings.DECODER_FIT_RIDGE_SCALE, "per_client": {}}
    for k in clients:
        model, report = fit_client(k, ref, unlearned, quantities)
        for lvl, r in report.items():
            logger.info(f"  Client {k} {lvl}: {r['patches']:,} patches of {r['patch_length']}, lambda {r['lambda']:.3g}; "
                        f"feature error {r['residual_before']:.4f} -> {r['residual_after']:.4f}; "
                        f"layer changed by {r['relative_change']:.2%}")
        helpers.atomic_save({n: v.detach().cpu() for n, v in model.state_dict().items()}, fit_path(withdrawn, k))
        summary["per_client"][k] = report
        logger.info(f"  Client {k}: wrote {fit_path(withdrawn, k)}")
    helpers.atomic_save_json(summary, os.path.join(settings.step_dir(withdrawn, 4), "fit_summary.json"))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Step 4: closed-form fit of the decoder entry layers")
    settings.add_common_args(parser)
    parser.add_argument("--withdraw", type=int, required=True, help="Withdrawn client id")
    parser.add_argument("--trial", action="store_true", help="Real model, first slices only")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    settings.configure(results_dir=args.results_dir, fixture=args.fixture, trial=args.trial)
    helpers.setup_logging(settings.logs_dir(args.withdraw), "step4_decoder_fit")
    try:
        run_step4(args.withdraw, overwrite=args.overwrite)
    except (FileExistsError, FileNotFoundError) as e:
        logging.getLogger().error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
