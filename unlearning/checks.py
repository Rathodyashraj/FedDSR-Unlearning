# unlearning/checks.py
"""
Correctness checks for each phase of the unlearning code. These are not evaluation metrics.

    python -m unlearning.checks --phase 1
    python -m unlearning.checks --fixture            # every phase implemented so far
"""
import argparse
import contextlib
import copy
import glob
import importlib.util
import json
import logging
import os
import sys
import tempfile

import torch

import config
from unlearning import settings
from unlearning import helpers
import models



# -----------------
# Small harness
# -----------------
class CheckFailed(AssertionError):
    pass


class CheckSkipped(Exception):
    pass


def require(cond, msg):
    if not cond:
        raise CheckFailed(msg)


CHECKS = {}  # phase -> list of (name, fn)


def check(phase, name):
    def wrap(fn):
        CHECKS.setdefault(phase, []).append((name, fn))
        return fn
    return wrap


@contextlib.contextmanager
def config_override(**values):
    """Temporarily change config values (restored afterwards)."""
    old = {k: getattr(config, k) for k in values}
    try:
        for k, v in values.items():
            setattr(config, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(config, k, v)


def load_original_models(path):
    """Import an unmodified models.py (given with --original-models) under a different module name."""
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"--original-models: {path} not found")
    spec = importlib.util.spec_from_file_location("original_models", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_batch(batch_size, seed):
    """Smooth synthetic HR images in [-1, 1] and their 2x downsampled LR inputs."""
    g = torch.Generator().manual_seed(seed)
    size = config.IMAGE_SIZE
    coarse = torch.rand(batch_size, config.IMAGE_CHANNELS, size // 16, size // 16, generator=g)
    hr = torch.nn.functional.interpolate(coarse, size=(size, size), mode="bicubic", align_corners=False)
    hr = (hr.clamp(0, 1) * 2 - 1)
    lr = torch.nn.functional.avg_pool2d(hr, 2)
    return {"lr": lr, "hr": hr}


@contextlib.contextmanager
def record_argmin():
    """Record every torch.argmin result (used to read the index the original forward picks)."""
    calls = []
    real = torch.argmin

    def wrapped(*args, **kwargs):
        out = real(*args, **kwargs)
        calls.append(out.detach().clone())
        return out

    torch.argmin = wrapped
    try:
        yield calls
    finally:
        torch.argmin = real


def differing_keys(a, b, keys=None):
    keys = keys if keys is not None else a.keys()
    diff = [k for k in keys if not torch.equal(a[k], b[k])]
    return diff


# -----------------
# Phase 1
# -----------------
STATE = {}  # shared between checks


def trained_state():
    """A model state with non-trivial codebooks: one training-mode batch with vitalization firing."""
    if "trained_state" not in STATE:
        with config_override(CODEBOOK_RESET_INTERVAL=1):
            torch.manual_seed(42)
            m = models.VQVAE2()
            m.train()
            with torch.no_grad():
                m(synthetic_batch(batch_size=2, seed=7)["lr"])
        STATE["trained_state"] = copy.deepcopy(m.state_dict())
    return STATE["trained_state"]


def train_one_batch(model_cls):
    """One real Client.train step (full loss suite) on one synthetic batch, seeded like main.py."""
    import client as client_module
    batch = synthetic_batch(batch_size=2, seed=7)
    captured = {}

    real_cls = client_module.VQVAE2
    client_module.VQVAE2 = model_cls
    try:
        # Interval 1 makes vitalization fire on this batch, so that path is compared too.
        with config_override(LOCAL_EPOCHS=1, CODEBOOK_RESET_INTERVAL=1, DEVICE=torch.device("cpu")):
            torch.manual_seed(42)
            c = client_module.Client(client_id=0, train_loader=[batch], test_loader=[batch],
                                     val_loader=[batch], results_logger=None, device=torch.device("cpu"))
            initial = copy.deepcopy(c.model.state_dict())

            def hook(module, inputs, output):
                captured["reconstructed"] = output[0].detach().clone()
                captured["vq_loss"] = output[1].detach().clone()

            c.model.register_forward_hook(hook)
            log_lines = []
            handler = logging.Handler()
            handler.emit = lambda record: log_lines.append(record.getMessage())
            root = logging.getLogger()
            old_level = root.level
            root.addHandler(handler)
            root.setLevel(logging.INFO)  # Client logs the epoch loss at INFO
            try:
                c.train(current_round=1, global_shared_params=c.get_shared_params())
            finally:
                root.removeHandler(handler)
                root.setLevel(old_level)
    finally:
        client_module.VQVAE2 = real_cls
    captured["log"] = [l for l in log_lines if "Total Loss" in l]
    return initial, c.model.state_dict(), captured


@check(1, "Train one batch: changed models.py vs original, same seed")
def check_train_equivalence(args):
    original = STATE["original_models"]
    if original is None:
        raise CheckSkipped("needs an unmodified models.py: --original-models PATH")
    init_new, final_new, cap_new = train_one_batch(models.VQVAE2)
    init_old, final_old, cap_old = train_one_batch(original.VQVAE2)

    old_keys = list(final_old.keys())
    extra = sorted(set(final_new) - set(final_old))
    require(extra == sorted(helpers.MASK_KEYS),
            f"unexpected new state keys: {extra}")
    require(not differing_keys(init_new, init_old, old_keys), "initial weights differ")
    require(torch.equal(cap_new["vq_loss"], cap_old["vq_loss"]), "vq loss differs")
    require(torch.equal(cap_new["reconstructed"], cap_old["reconstructed"]), "model output differs")
    require(len(cap_new["log"]) == 1 and len(cap_old["log"]) == 1, "loss line was not captured")
    require(cap_new["log"] == cap_old["log"], f"logged loss differs: {cap_new['log']} vs {cap_old['log']}")
    diff = differing_keys(final_new, final_old, old_keys)
    require(not diff, f"{len(diff)} tensors differ after the step, e.g. {diff[:3]}")
    changed = [k for k in old_keys if not torch.equal(init_old[k], final_old[k])]
    require(any(k in changed for k in helpers.CODEBOOK_KEYS), "codebooks did not change, so the vitalized step was not exercised")
    for k in helpers.MASK_KEYS:
        require(bool(final_new[k].all()), f"{k} not all True after training")
    return (f"{len(old_keys)} tensors bit-identical after one step "
            f"({len(changed)} changed by the step, incl. vitalized codebooks); "
            f"{cap_new['log'][0].split('|')[1].strip()}")


@check(1, "Load old checkpoints into the changed model")
def check_load_old_checkpoints(args):
    model = models.VQVAE2()
    model.load_state_dict(trained_state())
    files = {}
    with tempfile.TemporaryDirectory() as tmp:
        # The two layouts the old training code writes: shared params only, and the full
        # state dict, which then had no mask buffers
        files["best_model.pth"] = os.path.join(tmp, "best_model.pth")
        torch.save(model.get_shared_state_dict(), files["best_model.pth"])
        files["best_decoder_client_k.pth"] = os.path.join(tmp, "best_decoder.pth")
        old_full = {k: v for k, v in model.state_dict().items() if k not in helpers.MASK_KEYS}
        torch.save(old_full, files["best_decoder_client_k.pth"])
        if args.results_dir:
            for path in sorted(glob.glob(os.path.join(args.results_dir, "**", "*.pth"), recursive=True)):
                files[path] = path
        lines = []
        for label, path in files.items():
            sd = helpers.load_file(path)
            m = models.VQVAE2()
            m.load_shared_state_dict(sd)  # the training code's loader (strict=False)
            new_sd = m.state_dict()
            mismatched = [k for k in sd if k in new_sd and not torch.equal(sd[k], new_sd[k])]
            require(not mismatched, f"{label}: {len(mismatched)} values not loaded")
            unknown = [k for k in sd if k not in new_sd]
            require(not unknown, f"{label}: keys not in the model: {unknown[:3]}")
            for k in helpers.MASK_KEYS:
                if k not in sd:
                    require(bool(new_sd[k].all()), f"{label}: {k} not all True")
            lines.append(f"{os.path.basename(path)}: {len(sd)} keys loaded")
    return "; ".join(lines) + "; all mask entries True"


@check(1, "nearest() vs the index forward picks")
def check_nearest_matches_forward(args):
    # forward of the original models.py if given, otherwise of the changed one with the default mask
    original = STATE["original_models"]
    old = (original.VQVAE2 if original is not None else models.VQVAE2)()
    old.load_state_dict(trained_state(), strict=False)
    old.eval()
    new = models.VQVAE2()
    new.load_state_dict(trained_state())
    new.eval()

    batch = synthetic_batch(batch_size=2, seed=11)
    totals = {}
    with torch.no_grad():
        # 1) Latents from the encoder, quantized by the original forward
        pre = {}
        hooks = [old.quantizer_top.register_forward_pre_hook(lambda m, i: pre.__setitem__("top", i[0].clone())),
                 old.quantizer_bottom.register_forward_pre_hook(lambda m, i: pre.__setitem__("bottom", i[0].clone()))]
        with record_argmin() as calls:
            old(batch["lr"])
        for h in hooks:
            h.remove()
        require(len(calls) == 2, f"expected 2 argmin calls in forward, got {len(calls)}")
        forward_idx = {"top": calls[0].view(-1), "bottom": calls[1].view(-1)}
        z_top, z_bottom = new.encode(batch["lr"])
        require(torch.equal(z_top, pre["top"]) and torch.equal(z_bottom, pre["bottom"]),
                "encode() differs from the latents forward passes to the quantizers")
        for level, z in (("top", z_top), ("bottom", z_bottom)):
            q = helpers.quantizer(new, level)
            usage_before = q.code_usage.clone(), q.batch_counter.clone()
            weight_before = q._embedding.weight.clone()
            idx = q.nearest(z).view(-1)
            require(torch.equal(idx, forward_idx[level]), f"{level}: nearest() differs from forward")
            require(torch.equal(q.code_usage, usage_before[0]) and torch.equal(q.batch_counter, usage_before[1]),
                    f"{level}: nearest() changed the usage counters")
            require(torch.equal(q._embedding.weight, weight_before), f"{level}: nearest() changed the codebook")
            totals[level] = (idx.numel(), idx.unique().numel())

        # 2) Latents spread over the codebook, so many rows are picked
        g = torch.Generator().manual_seed(3)
        for level in ("top", "bottom"):
            q_old = getattr(old, helpers.LEVELS[level][0])
            q_new = helpers.quantizer(new, level)
            cb = q_old._embedding.weight
            pick = torch.randint(0, cb.shape[0], (2 * 16 * 16,), generator=g)
            z = (cb[pick] + 0.02 * torch.randn(pick.shape[0], cb.shape[1], generator=g))
            z = z.view(2, 16, 16, cb.shape[1]).permute(0, 3, 1, 2).contiguous()
            with record_argmin() as calls:
                q_old(z)
            idx = q_new.nearest(z).view(-1)
            require(torch.equal(idx, calls[0].view(-1)), f"{level}: nearest() differs from forward on spread latents")
            # the changed forward with the default mask picks the same rows too
            with record_argmin() as calls_new:
                q_new(z)
            require(torch.equal(calls_new[0].view(-1), idx), f"{level}: changed forward differs")
            STATE.setdefault("spread", {})[level] = (z, idx)
            totals[level + " spread"] = (idx.numel(), idx.unique().numel())
    source = "original" if original is not None else "changed (default mask)"
    return f"forward of the {source} models.py; " + "; ".join(
        f"{k}: {n} positions, {u} distinct rows identical" for k, (n, u) in totals.items())


@check(1, "Masked rows are never returned")
def check_mask(args):
    new = models.VQVAE2()
    new.load_state_dict(trained_state())
    new.eval()
    lines = []
    with torch.no_grad():
        for level in ("top", "bottom"):
            q = helpers.quantizer(new, level)
            z, idx = STATE["spread"][level]
            rows = q._num_embeddings
            # Mask the most-used rows, so the mask has to change the answer
            counts = torch.bincount(idx, minlength=rows)
            masked = torch.argsort(counts, descending=True)[:8]
            mask = torch.ones(rows, dtype=torch.bool)
            mask[masked] = False

            got = q.nearest(z, mask=mask).view(-1)
            require(not torch.isin(got, masked).any(), f"{level}: nearest() returned a masked row")
            # Independent answer: argmin over the active rows only
            active = torch.nonzero(mask).view(-1)
            dist = helpers.squared_distances(helpers.flatten_latent(z), q._embedding.weight)
            expected = active[torch.argmin(dist[:, active], dim=1)]
            require(torch.equal(got, expected), f"{level}: masked nearest() is not the nearest active row")
            require(torch.equal(helpers.masked_nearest(dist, mask), expected), f"{level}: masked_nearest() differs")
            changed = int((got != idx).sum())
            require(changed > 0, f"{level}: masking did not change any index; test is not informative")

            # The mask buffer is honoured by forward too, and vitalize=False keeps the codebook fixed
            q.active_mask.copy_(mask)
            q.train()
            q.vitalize = False
            weight_before = q._embedding.weight.clone()
            usage_before = q.code_usage.clone()
            with config_override(CODEBOOK_RESET_INTERVAL=1):
                q(z)
            used = q.code_usage - usage_before
            require(float(used[masked].sum()) == 0, f"{level}: forward selected a masked row")
            require(torch.equal(q._embedding.weight, weight_before), f"{level}: codebook changed with vitalize=False")
            q.vitalize = True
            with config_override(CODEBOOK_RESET_INTERVAL=1):
                q(z)
            require(not torch.equal(q._embedding.weight, weight_before),
                    f"{level}: vitalize=True did not vitalize (control)")
            q.eval()
            lines.append(f"{level}: 8 rows masked, {changed} positions re-routed, none to a masked row")

    # Ties go to the lowest index (Appendix A toy codebook, latent (3, 3))
    toy = torch.tensor([[1., 1.], [5., 5.], [1., 5.], [5., 1.]])
    d = helpers.squared_distances(torch.tensor([[3., 3.]]), toy)
    require(int(helpers.masked_nearest(d, torch.ones(4, dtype=torch.bool))) == 0, "tie not broken to the lowest index")
    lines.append("tie goes to lowest index")
    return "; ".join(lines)


@check(1, "Helpers: sizes, parameter groups, client guard, write protection")
def check_helpers(args):
    m = models.VQVAE2()
    sizes = helpers.model_sizes(m)
    parts = helpers.split_state_dict(m.state_dict())
    n_enc = sum(v.numel() for v in parts["encoder"].values())
    require(n_enc == sizes["encoder_weights"], "encoder split disagrees with encoder_parameters")
    shared = m.get_shared_state_dict()
    require(set(shared) == set(parts["encoder"]) | set(parts["codebook"]), "split disagrees with get_shared_state_dict")
    require(set(parts["other"]) == {f"{q}.{b}" for q in ("quantizer_top", "quantizer_bottom")
                                   for b in ("code_usage", "batch_counter", "active_mask")},
            f"unexpected 'other' keys {sorted(parts['other'])}")

    clients = helpers.all_clients()
    for w in clients:
        rem = helpers.remaining_clients(w)
        require(w not in rem and len(rem) == len(clients) - 1, f"remaining_clients({w}) wrong")
    helpers.set_withdrawn(clients[-1])
    try:
        helpers.check_client_access(clients[-1], "decoder")
        raise CheckFailed("access to the withdrawn client was not refused")
    except PermissionError:
        pass
    finally:
        helpers._WITHDRAWN = None
        helpers._ACCESS_LOG.clear()

    saved_dir = settings.RESULTS_DIR
    settings.RESULTS_DIR = saved_dir or settings.FIXTURE_RESULTS_DIR
    try:
        for protected in (os.path.join(settings.training_dir(), "x.pth"),
                          os.path.join(settings.stage2_dir(), "x.pth"),
                          os.path.join("FedDSR_results", config.AGGREGATOR, "best_model.pth")):
            try:
                helpers.assert_writable(protected)
                raise CheckFailed(f"write to {protected} was not refused")
            except PermissionError:
                pass
        helpers.assert_writable(os.path.join(settings.withdraw_dir(clients[-1]), "logs", "run.log"))
    finally:
        settings.RESULTS_DIR = saved_dir

    t, b = sizes["top"], sizes["bottom"]
    return (f"encoder weights {sizes['encoder_weights']:,}; "
            f"top: {t['codebook_rows']}x{t['latent_dim']} codebook, grid {t['grid']}, patch {t['patch_length']}; "
            f"bottom: {b['codebook_rows']}x{b['latent_dim']} codebook, grid {b['grid']}, patch {b['patch_length']}; "
            f"guard and write protection work")


def setup_phase1(args):
    STATE["original_models"] = load_original_models(args.original_models)


# -----------------
# Phase 2
# -----------------
def require_results_dir():
    require(settings.RESULTS_DIR is not None, "run with --fixture or --results-dir")
    require(os.path.isdir(settings.training_dir()),
            f"{settings.training_dir()} not found" + (" (run: python -m unlearning.fixture --build)" if settings.FIXTURE else ""))


@check(2, "Results folder has the layout the training code writes")
def check_fixture_layout(args):
    from unlearning import inventory
    require_results_dir()
    ref = models.VQVAE2()
    shared_ref = {k: v.shape for k, v in ref.get_shared_state_dict().items()}
    full_ref = {k: v.shape for k, v in ref.state_dict().items()}

    def same_layout(path, reference, allow_missing=()):
        sd = helpers.load_file(path)
        missing = set(reference) - set(sd) - set(allow_missing)
        extra = set(sd) - set(reference)
        require(not missing and not extra, f"{path}: missing {sorted(missing)[:3]}, extra {sorted(extra)[:3]}")
        wrong = [k for k in sd if tuple(sd[k].shape) != tuple(reference[k])]
        require(not wrong, f"{path}: wrong shapes for {wrong[:3]}")
        return len(sd)

    files = inventory.expected_files()
    lines = []
    for name, path in files.items():
        if not settings.FIXTURE and not os.path.exists(path):
            lines.append(f"{name} absent")
            continue
        require(os.path.exists(path), f"missing {name} ({path})")
        if "decoder_client" in name:
            # Files written by the old code have no mask buffers
            n = same_layout(path, full_ref, allow_missing=helpers.MASK_KEYS)
        else:
            n = same_layout(path, shared_ref)
        lines.append(f"{name} ({n} keys)")
    rounds = inventory.saved_rounds()
    require(rounds and rounds[0] == 1 and rounds == list(range(1, rounds[-1] + 1)),
            f"round folders not contiguous from 1: {rounds}")
    require(os.path.exists(os.path.join(settings.training_dir(), "metrics.csv")), "metrics.csv missing")
    if settings.FIXTURE:
        from unlearning import fixture as fx
        for k in helpers.all_clients():
            n_png = len(glob.glob(os.path.join(fx.client_data_dir(settings.RESULTS_DIR, k), "*.png")))
            require(n_png == fx.IMAGES_PER_CLIENT, f"client {k}: {n_png} images")
    return f"rounds saved {rounds}; " + "; ".join(lines)


@check(2, "Inventory run on the results folder")
def check_inventory(args):
    import json
    from unlearning import inventory
    require_results_dir()
    inv = inventory.run_inventory()
    with open(helpers.inventory_path()) as f:
        on_disk = json.load(f)
    require(on_disk["step0_case"] == inv["step0_case"], "inventory.json differs from the returned inventory")
    if settings.FIXTURE:
        missing = [n for n, f in inv["files"].items() if not f["exists"]]
        require(not missing, f"files not found: {missing}")
        for k, d in inv["deployed"].items():
            require(d["source"] == "stage2_refined", f"client {k}: deployed is {d['source']}, expected the refined file")
        require(inv["step0_case"] == 1, f"Step 0 Case {inv['step0_case']} reported, expected Case 1")
    for k in helpers.all_clients():
        require(helpers.deployed_model_path(k) == inv["deployed"][str(k)]["path"],
                f"client {k}: deployed_model_path disagrees with the inventory")
    s = inv["sizes"]
    require(s is not None, "sizes not read")
    diff = inv["codebook_rows_differing_from_best_model"]
    return (f"{sum(f['exists'] for f in inv['files'].values())}/{len(inv['files'])} files found; "
            "deployed: " + ", ".join("%s=%s" % (k, d["source"]) for k, d in inv["deployed"].items()) + "; "
            f"Step 0 Case {inv['step0_case']}; "
            "rows differing (top/bottom): " + ", ".join("%s=%d/%d" % (k, v["top"], v["bottom"]) for k, v in diff.items()) + "; "
            f"sizes: top {s['top']['codebook_rows']}x{s['top']['latent_dim']} grid {s['top']['grid']}, "
            f"bottom {s['bottom']['codebook_rows']}x{s['bottom']['latent_dim']} grid {s['bottom']['grid']}, "
            f"patch {s['top']['patch_length']}")


# -----------------
# Phase 3 (Step 0)
# -----------------
def snapshot_tree(*dirs):
    """(path, size, mtime) of every file below the given folders."""
    out = {}
    for d in dirs:
        for root, _, names in os.walk(d):
            for n in names:
                p = os.path.join(root, n)
                st = os.stat(p)
                out[p] = (st.st_size, st.st_mtime_ns)
    return out


def same_tensors(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


@check(3, "Step 0 Case 1: phi0.pth is the encoder part of the round-0 file")
def check_step0_case1(args):
    import json
    from unlearning import step0_initial_weights as s0
    require_results_dir()
    require(os.path.exists(s0.round0_path()), "round-0 file missing; Case 1 cannot be tested")
    phi0, info = s0.run_step0(overwrite=True)
    require(info["case"] == 1, f"Case {info['case']} taken, expected Case 1")
    on_disk = helpers.load_file(s0.phi0_path())
    require(same_tensors(on_disk, phi0), "phi0.pth on disk differs from the returned weights")
    require(same_tensors(on_disk, helpers.encoder_state(helpers.load_file(s0.round0_path()))),
            "phi0.pth is not bit-for-bit the encoder part of the round-0 file")
    expected_keys = {k for k, _ in models.VQVAE2().named_parameters() if helpers.is_encoder_key(k)}
    require(set(on_disk) == expected_keys, "phi0.pth does not hold exactly the four encoder groups")
    n = sum(v.numel() for v in on_disk.values())
    require(n == helpers.model_sizes(models.VQVAE2())["encoder_weights"], "weight count disagrees with the model")
    with open(os.path.join(settings.phi0_dir(), "phi0_info.json")) as f:
        require(json.load(f)["case"] == 1, "phi0_info.json does not record Case 1")
    return f"phi0.pth = encoder part of round_0_global_model.pth, {len(on_disk)} tensors, {n:,} weights"


@check(3, "Step 0 Case 2: regenerated weights vs the saved round-0 file")
def check_step0_regenerate(args):
    from unlearning import step0_initial_weights as s0
    require_results_dir()
    saved = helpers.load_file(s0.round0_path())
    before = snapshot_tree(settings.training_dir(), settings.stage2_dir())
    lines = []
    for use_data, label in ((True, "with the real loaders"), (False, "with stand-in loaders")):
        regen = s0.regenerate_initial_shared(use_data=use_data)
        require(same_tensors(helpers.encoder_state(regen), helpers.encoder_state(saved)),
                f"regenerated encoder ({label}) is not bit-for-bit equal to the round-0 file")
        require(same_tensors(regen, saved), f"regenerated shared model ({label}) differs from the round-0 file")
        lines.append(f"bit-for-bit equal {label}")
    require(snapshot_tree(settings.training_dir(), settings.stage2_dir()) == before,
            "regeneration changed a file in the training folders")
    return "; ".join(lines) + f" ({len(saved)} shared tensors); training folders untouched"


@check(3, "Step 0: r x sqrt(N) against round 1, regenerated vs control")
def check_step0_verification(args):
    from unlearning import step0_initial_weights as s0
    require_results_dir()
    phi0, info = s0.run_step0(force_case2=True, overwrite=True)
    require(info["case"] == 2, "Case 2 not taken")
    require(info.get("equals_saved_round0") is True, "Case 2 result not equal to the saved round-0 file")
    v = info["verification"]
    require(v["regenerated_score"] > settings.STEP0_MIN_SCORE, f"regenerated score {v['regenerated_score']:.2f} too low")
    require(abs(v["control_score"]) < settings.STEP0_CONTROL_MAX_SCORE, f"control score {v['control_score']:.2f} not near 0")
    require(v["passed"], "verification did not pass")
    return (f"N={v['n_weights']:,}: regenerated r={v['regenerated_r']:.4f}, r*sqrt(N)={v['regenerated_score']:.1f} (> {settings.STEP0_MIN_SCORE}); "
            f"control r={v['control_r']:.5f}, r*sqrt(N)={v['control_score']:.2f} (|.| < {settings.STEP0_CONTROL_MAX_SCORE})")


@check(3, "Step 0 Case 2 refuses a wrong anchor")
def check_step0_wrong_anchor(args):
    from unlearning import step0_initial_weights as s0
    require_results_dir()
    before = snapshot_tree(settings.phi0_dir())
    phi0_before = before.get(s0.phi0_path())
    saved_seed = settings.TRAINING_SEED
    settings.TRAINING_SEED = saved_seed + 1  # simulate a different environment / wrong replay
    try:
        s0.run_step0(force_case2=True, overwrite=True)
        raise CheckFailed("a wrong anchor was accepted")
    except RuntimeError as e:
        msg = str(e)
    finally:
        settings.TRAINING_SEED = saved_seed
    after = snapshot_tree(settings.phi0_dir())
    require(after.get(s0.phi0_path()) == phi0_before, "phi0.pth was written despite the failed verification")
    failed = os.path.join(settings.phi0_dir(), "phi0_failed_verification.json")
    import json
    with open(failed) as f:
        v = json.load(f)["verification"]
    os.remove(failed)
    # Leave the fixture in its normal state: phi0 from Case 1
    s0.run_step0(overwrite=True)
    return (f"seed {saved_seed + 1}: r*sqrt(N)={v['regenerated_score']:.2f} -> refused, phi0.pth not written "
            f"('{msg.split(':')[0]}')")


# -----------------
# Phase 4 (Step 1)
# -----------------
@check(4, "Step 1: counts sum to slices x grid positions; sums match the latents")
def check_step1_counts(args):
    from unlearning import step1_statistics as s1
    require_results_dir()
    w = args.withdraw
    helpers._ACCESS_LOG.clear()
    results = s1.run_step1(w, overwrite=True)
    remaining = helpers.remaining_clients(w)
    require(sorted(results) == remaining, f"statistics for {sorted(results)}, expected {remaining}")
    files = sorted(os.listdir(settings.step_dir(w, 1)))
    require(files == sorted(f"stats_client_{k}.pt" for k in remaining), f"unexpected files {files}")
    require(all(k != w for _, k, _ in helpers.access_log()), f"client {w} was accessed")

    ref = helpers.load_reference_model()
    sizes = helpers.model_sizes(ref)
    lines = []
    for k in remaining:
        st = helpers.load_file(s1.stats_path(w, k))
        require(st["slices_seen"] == st["num_train_slices"], f"client {k}: {st['slices_seen']} of {st['num_train_slices']} slices seen")
        for level, suffix in s1.LEVEL_KEYS.items():
            per_slice = sizes[level]["positions_per_slice"]
            total = float(st[f"n_{suffix}"].sum())
            require(total == st["slices_seen"] * per_slice,
                    f"client {k} {level}: counts sum to {total}, expected {st['slices_seen']} x {per_slice}")
            require(st[f"n_{suffix}"].dtype == torch.float64 and st[f"s_{suffix}"].dtype == torch.float64, "not float64")
            # rows never chosen have zero sums
            require(bool((st[f"s_{suffix}"][st[f"n_{suffix}"] == 0] == 0).all()), f"client {k} {level}: sums on unused rows")
        # Independent check of the sums: total of all latents from a separate pass
        totals = {lvl: 0 for lvl in helpers.LEVELS}
        with torch.no_grad():
            for batch in helpers.make_loader(k, "train", shuffle=False, drop_last=False):
                zt, zb = ref.encode(batch["lr"].to(settings.DEVICE))
                totals["top"] += helpers.flatten_latent(zt).double().sum(0)
                totals["bottom"] += helpers.flatten_latent(zb).double().sum(0)
        for level, suffix in s1.LEVEL_KEYS.items():
            require(torch.allclose(st[f"s_{suffix}"].sum(0), totals[level].cpu(), rtol=1e-9, atol=1e-6),
                    f"client {k} {level}: sum of s differs from the sum of latents")
        lines.append(f"client {k}: {st['slices_seen']} slices -> top {int(st['n_top'].sum()):,} "
                     f"(= x{sizes['top']['positions_per_slice']}), bottom {int(st['n_bot'].sum()):,} "
                     f"(= x{sizes['bottom']['positions_per_slice']}), m_top={st['m_top']:.4g}, m_bot={st['m_bot']:.4g}")
    return f"withdrawn {w}, never accessed; " + "; ".join(lines)


@check(4, "Step 1: run twice gives identical counts, sums and margins; no quantizer side effects")
def check_step1_repeat(args):
    from unlearning import step1_statistics as s1
    require_results_dir()
    w = args.withdraw
    helpers.set_withdrawn(w)
    ref = helpers.load_reference_model()
    buffers = {k: v.clone() for k, v in ref.state_dict().items()}
    lines = []
    for k in helpers.remaining_clients(w):
        a = s1.client_statistics(ref, k)
        b = s1.client_statistics(ref, k)
        saved = helpers.load_file(s1.stats_path(w, k))
        for key in ("n_top", "s_top", "n_bot", "s_bot"):
            require(torch.equal(a[key], b[key]) and torch.equal(a[key], saved[key]), f"client {k}: {key} differs between runs")
        for key in ("m_top", "m_bot"):
            require(a[key] == b[key] == saved[key], f"client {k}: {key} differs between runs")
        lines.append(f"client {k} identical")
    require(same_tensors(buffers, ref.state_dict()), "the pass changed the reference model (weights, counters or codebook)")
    return "; ".join(lines) + " (3 runs incl. the saved file); reference model and its counters unchanged"


@check(4, "Step 1: results do not depend on the distance block size")
def check_step1_chunks(args):
    from unlearning import step1_statistics as s1
    require_results_dir()
    w = args.withdraw
    helpers.set_withdrawn(w)
    ref = helpers.load_reference_model()
    k = helpers.remaining_clients(w)[0]
    saved_chunk = settings.DISTANCE_CHUNK_POSITIONS
    runs = {}
    try:
        for chunk in (saved_chunk, 1000, 10 ** 9):
            settings.DISTANCE_CHUNK_POSITIONS = chunk
            runs[chunk] = s1.client_statistics(ref, k)
    finally:
        settings.DISTANCE_CHUNK_POSITIONS = saved_chunk
    base = runs[saved_chunk]
    for chunk, r in runs.items():
        for key in ("n_top", "s_top", "n_bot", "s_bot"):
            require(torch.equal(base[key], r[key]), f"block size {chunk}: {key} differs")
        require(base["m_top"] == r["m_top"] and base["m_bot"] == r["m_bot"], f"block size {chunk}: margins differ")
    return f"client {k}: identical with blocks of {', '.join(str(c) for c in runs)} positions"


@check(4, "Step 1: toy codebook from Appendix A")
def check_step1_toy(args):
    from unlearning import step1_statistics as s1
    codebook = torch.tensor([[1., 1.], [5., 5.], [1., 5.], [5., 1.]])
    z = torch.tensor([[1.0, 1.1], [4.8, 5.0], [3.0, 3.0]])
    acc = s1.LevelAccumulator(4, 2, "cpu")
    acc.add(z, codebook, generator=torch.Generator().manual_seed(0), sample_size=settings.MARGIN_SAMPLE_PER_BATCH)
    require(acc.n.tolist() == [2, 1, 0, 0], f"counts {acc.n.tolist()}")
    require(torch.allclose(acc.s[0], torch.tensor([4.0, 4.1], dtype=torch.float64), atol=1e-6), f"s0 {acc.s[0].tolist()}")
    require(torch.allclose(acc.s[1], torch.tensor([4.8, 5.0], dtype=torch.float64), atol=1e-6), f"s1 {acc.s[1].tolist()}")
    require(bool((acc.s[2:] == 0).all()), "s2, s3 not zero")
    margins = torch.cat(acc.margins).tolist()
    require(abs(margins[0] - 15.2) < 1e-4 and abs(margins[1] - 14.4) < 1e-4 and margins[2] == 0,
            f"margins {margins}")
    require(abs(acc.median_margin() - 14.4) < 1e-4, f"median margin {acc.median_margin()}")
    return (f"n0=2, s0=({acc.s[0][0]:.1f}, {acc.s[0][1]:.1f}), n1=1, s1=({acc.s[1][0]:.1f}, {acc.s[1][1]:.1f}), "
            f"n2=n3=0, margins {[round(m, 4) for m in margins]}, median {acc.median_margin():.4f}; z3 tie -> row 0")


# -----------------
# Phase 5 (Step 2)
# -----------------
def step2_quantities(args):
    from unlearning import step2_server_quantities as s2
    require_results_dir()
    path = s2.quantities_path(args.withdraw)
    if "step2" not in STATE:
        require(all(os.path.exists(os.path.join(settings.step_dir(args.withdraw, 1), f"stats_client_{k}.pt"))
                    for k in helpers.remaining_clients(args.withdraw)), "Step 1 output missing (run the Phase 4 checks)")
        helpers._ACCESS_LOG.clear()
        STATE["step2"] = s2.run_step2(args.withdraw, overwrite=True)
        STATE["step2_access"] = helpers.access_log()
    return STATE["step2"], path


@check(5, "Step 2: weights sum to 1")
def check_step2_weights(args):
    q, path = step2_quantities(args)
    w = args.withdraw
    remaining = helpers.remaining_clients(w)
    require(sorted(q["weights"]) == remaining, f"weights for {sorted(q['weights'])}, expected {remaining}")
    total = sum(q["weights"].values())
    require(abs(total - 1) < 1e-12, f"weights sum to {total!r}")
    require(all(v >= 0 for v in q["weights"].values()), "negative weight")
    require(abs(sum(q["w_sim"].values()) - 1) < 1e-12 and abs(sum(q["w_data"].values()) - 1) < 1e-12,
            "similarity or data weights do not sum to 1")
    stats_reads = [k for kind, k, _ in STATE["step2_access"] if kind == "stats"]
    require(sorted(stats_reads) == remaining, f"statistics read for {stats_reads}")
    saved = helpers.load_file(path)
    require(saved["weights"] == q["weights"] and saved["withdrawn"] == w, "saved file differs")
    return (f"sum = {total:.15f}; " + ", ".join(
        "client %d: sim %+.4f -> w_sim %.4f, w_data %.4f, w %.4f" % (k, q["similarity_scores"][k], q["w_sim"][k],
                                                                    q["w_data"][k], q["weights"][k]) for k in remaining)
            + f"; only the stats of {remaining} were read")


@check(5, "Step 2: nearest old-codebook row of every active clean row is the row itself")
def check_step2_nearest(args):
    from unlearning import step1_statistics as s1
    q, _ = step2_quantities(args)
    ref = helpers.load_reference_model()
    lines = []
    for level, sfx in s1.LEVEL_KEYS.items():
        old = helpers.quantizer(ref, level)._embedding.weight.detach().cpu()
        V, mask = q[f"V_{sfx}"], q[f"mask_{sfx}"]
        active = torch.nonzero(mask).view(-1)
        nearest = torch.argmin(helpers.squared_distances(V[active], old), dim=1)
        wrong = active[nearest != active]
        require(wrong.numel() == 0, f"{level}: rows {wrong.tolist()[:10]} have a different nearest old row")
        lines.append(f"{level}: {active.numel()}/{mask.numel()} active rows, all map to themselves")
    return "; ".join(lines)


@check(5, "Step 2: inactive rows are all zeros and never selected")
def check_step2_inactive(args):
    from unlearning import step1_statistics as s1
    q, _ = step2_quantities(args)
    w = args.withdraw
    stats = {k: helpers.load_file(s1.stats_path(w, k)) for k in helpers.remaining_clients(w)}
    ref = helpers.load_reference_model()
    g = torch.Generator().manual_seed(5)
    lines = []
    for level, sfx in s1.LEVEL_KEYS.items():
        V, mask = q[f"V_{sfx}"], q[f"mask_{sfx}"]
        expected = sum(st[f"n_{sfx}"] for st in stats.values()) >= 1
        require(torch.equal(mask, expected), f"{level}: mask is not 'summed count >= 1'")
        require(bool((V[~mask] == 0).all()), f"{level}: an inactive row is not zero")
        require(bool((V[mask].abs().sum(1) > 0).all()), f"{level}: an active row is zero")
        # Latents of every remaining client, plus random latents and latents placed at the zero vector
        picks = []
        with torch.no_grad():
            for k in stats:
                batch = next(iter(helpers.make_loader(k, "train", shuffle=False, drop_last=False)))
                zt, zb = ref.encode(batch["lr"].to(settings.DEVICE))
                z = helpers.flatten_latent(zt if level == "top" else zb).cpu()
                picks.append(helpers.masked_nearest(helpers.squared_distances(z, V), mask))
            rand = torch.randn(4096, V.shape[1], generator=g) * V[mask].std()
            picks.append(helpers.masked_nearest(helpers.squared_distances(rand, V), mask))
            picks.append(helpers.masked_nearest(helpers.squared_distances(torch.zeros(16, V.shape[1]), V), mask))
        picks = torch.cat(picks)
        require(bool(mask[picks].all()), f"{level}: an inactive row was selected")
        n_inactive = int((~mask).sum())
        lines.append(f"{level}: {n_inactive} inactive row(s), zero, never selected in {picks.numel():,} lookups")

    # Same statistics with the most-used rows removed, as if only the withdrawn client had used them
    from unlearning import step2_server_quantities as s2
    removed = {}
    edited = {k: dict(st) for k, st in stats.items()}
    for sfx in s1.LEVEL_KEYS.values():
        total = sum(st[f"n_{sfx}"] for st in stats.values())
        rows = torch.argsort(total, descending=True)[:4]
        removed[sfx] = rows
        for st in edited.values():
            st[f"n_{sfx}"] = st[f"n_{sfx}"].clone(); st[f"n_{sfx}"][rows] = 0
            st[f"s_{sfx}"] = st[f"s_{sfx}"].clone(); st[f"s_{sfx}"][rows] = 0
    q2 = s2.compute_server_quantities(edited, settings.AGGREGATION_GAMMA)
    for level, sfx in s1.LEVEL_KEYS.items():
        V, mask, rows = q2[f"V_{sfx}"], q2[f"mask_{sfx}"], removed[sfx]
        require(not mask[rows].any() and int((~mask).sum()) == rows.numel(), f"{level}: removed rows not inactive")
        require(bool((V[rows] == 0).all()), f"{level}: removed rows not zero")
        with torch.no_grad():
            z = []
            for k in stats:
                batch = next(iter(helpers.make_loader(k, "train", shuffle=False, drop_last=False)))
                zt, zb = ref.encode(batch["lr"].to(settings.DEVICE))
                z.append(helpers.flatten_latent(zt if level == "top" else zb).cpu())
            z = torch.cat(z + [torch.zeros(16, V.shape[1])])
            unmasked = torch.argmin(helpers.squared_distances(z, V), dim=1)
            picks = helpers.masked_nearest(helpers.squared_distances(z, V), mask)
        require(not torch.isin(picks, rows).any(), f"{level}: a removed row was selected")
        lines.append(f"{level} with the 4 most-used rows removed: inactive, zero, never selected "
                     f"({int(torch.isin(unmasked, rows).sum()):,} lookups would have picked them without the mask)")
    return "; ".join(lines)


def _reference_step2(stats, gamma):
    """Independent plain-Python implementation of the plan's formulas, for the toy check."""
    import math
    clients = sorted(stats)
    def mean_latent(st, sfx):
        n = sum(st[f"n_{sfx}"])
        return [sum(row[d] for row in st[f"s_{sfx}"]) / n for d in range(len(st[f"s_{sfx}"][0]))]
    desc = {k: [0.5 * (a + b) for a, b in zip(mean_latent(stats[k], "top"), mean_latent(stats[k], "bot"))] for k in clients}
    def cos(a, b):
        return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))
    sim = {k: sum(cos(desc[k], desc[i]) for i in clients if i != k) / (len(clients) - 1) for k in clients}
    clipped = {k: max(sim[k], 0.0) for k in clients}
    tot = sum(clipped.values())
    w_sim = {k: (clipped[k] / tot if tot > 0 else 1 / len(clients)) for k in clients}
    size = sum(stats[k]["num_train_slices"] for k in clients)
    w = {k: gamma * w_sim[k] + (1 - gamma) * stats[k]["num_train_slices"] / size for k in clients}
    out = {"weights": w}
    for sfx in ("top", "bot"):
        K, D = len(stats[clients[0]][f"n_{sfx}"]), len(stats[clients[0]][f"s_{sfx}"][0])
        V, mask = [], []
        for j in range(K):
            users = [k for k in clients if stats[k][f"n_{sfx}"][j] >= 1]
            mask.append(bool(users))
            if not users:
                V.append([0.0] * D)
                continue
            den = sum(w[k] for k in users)
            V.append([sum(w[k] * stats[k][f"s_{sfx}"][j][d] / stats[k][f"n_{sfx}"][j] for k in users) / den
                      for d in range(D)])
        out[f"V_{sfx}"], out[f"mask_{sfx}"] = V, mask
        out[f"delta_{sfx}"] = 0.5 * sum(w[k] * stats[k][f"m_{sfx}"] for k in clients)
    return out


@check(5, "Step 2: toy statistics against an independent implementation")
def check_step2_toy(args):
    from unlearning import step2_server_quantities as s2
    def as_tensors(raw):
        return {k: {key: (torch.tensor(v, dtype=torch.float64) if isinstance(v, list) else v)
                    for key, v in st.items()} for k, st in raw.items()}
    # Three clients, 4 rows of 2 numbers; row 3 is used by nobody at the top level
    raw = {
        0: {"n_top": [2, 0, 0, 0], "s_top": [[2, 4], [0, 0], [0, 0], [0, 0]], "m_top": 0.4,
            "n_bot": [1, 0, 3, 0], "s_bot": [[1, 1], [0, 0], [3, 6], [0, 0]], "m_bot": 0.2, "num_train_slices": 10},
        1: {"n_top": [1, 1, 0, 0], "s_top": [[3, 0], [0, 5], [0, 0], [0, 0]], "m_top": 0.8,
            "n_bot": [0, 2, 0, 1], "s_bot": [[0, 0], [4, 2], [0, 0], [5, 5]], "m_bot": 0.6, "num_train_slices": 30},
        2: {"n_top": [0, 2, 1, 0], "s_top": [[0, 0], [2, 2], [1, 3], [0, 0]], "m_top": 1.0,
            "n_bot": [2, 1, 0, 0], "s_bot": [[2, 4], [1, 0], [0, 0], [0, 0]], "m_bot": 0.1, "num_train_slices": 60},
    }
    got = s2.compute_server_quantities(as_tensors(raw), 0.5)
    exp = _reference_step2(raw, 0.5)
    for k in raw:
        require(abs(got["weights"][k] - exp["weights"][k]) < 1e-12, f"weight of client {k}")
    for sfx in ("top", "bot"):
        require(got[f"mask_{sfx}"].tolist() == exp[f"mask_{sfx}"], f"mask_{sfx}")
        require(torch.allclose(got[f"V_{sfx}_f64"], torch.tensor(exp[f"V_{sfx}"], dtype=torch.float64), atol=1e-12), f"V_{sfx}")
        require(abs(got[f"delta_{sfx}"] - exp[f"delta_{sfx}"]) < 1e-12, f"delta_{sfx}")
    # All similarity scores <= 0 -> equal similarity weights
    angles = [0.0, 2.0944, 4.18879]  # three descriptors 120 degrees apart: every cosine is -0.5
    import math
    neg = {}
    for k, a in enumerate(angles):
        v = [math.cos(a), math.sin(a)]
        neg[k] = {"n_top": [1], "s_top": [v], "m_top": 1.0, "n_bot": [1], "s_bot": [v], "m_bot": 1.0,
                  "num_train_slices": 10 * (k + 1)}
    gneg = s2.compute_server_quantities(as_tensors(neg), 0.5)
    require(all(abs(v - 1 / 3) < 1e-12 for v in gneg["w_sim"].values()), f"w_sim {gneg['w_sim']}")
    return ("weights " + ", ".join("%.6f" % got["weights"][k] for k in raw)
            + f"; masks top {got['mask_top'].tolist()}, bottom {got['mask_bot'].tolist()}; "
            f"V and delta match to 1e-12; all-negative similarities -> equal w_sim")


# -----------------
# Phase 6 (Step 3)
# -----------------
def step3_inputs(args):
    from unlearning import step3_shrink_repair as s3
    require_results_dir()
    helpers.set_withdrawn(args.withdraw)
    try:
        return s3.load_inputs(args.withdraw, settings.DEVICE)
    except FileNotFoundError as e:
        raise CheckFailed(f"{e} (run the Phase 3 and Phase 5 checks first)")


def old_codebook_targets(ref, quantities):
    """Quantities whose clean codebook is the old codebook itself, all rows active."""
    q = dict(quantities)
    for level, sfx in (("top", "top"), ("bottom", "bot")):
        q[f"V_{sfx}"] = helpers.quantizer(ref, level)._embedding.weight.detach().clone()
        q[f"mask_{sfx}"] = torch.ones_like(quantities[f"mask_{sfx}"])
    return q


def count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


@contextlib.contextmanager
def settings_override(**values):
    old = {k: getattr(settings, k) for k in values}
    try:
        for k, v in values.items():
            setattr(settings, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(settings, k, v)


@check(6, "Step 3: hinge on the toy example of Appendix A")
def check_step3_hinge_toy(args):
    from unlearning import step3_shrink_repair as s3
    V = torch.tensor([[2., 2.], [5., 5.], [1., 5.]])
    z = torch.tensor([[3.6, 3.6], [2.1, 2.2]])
    dist = helpers.squared_distances(z, V)
    h = s3.hinge(dist, torch.tensor([0, 0]), delta=1.0)
    require(abs(float(dist[0, 0]) - 5.12) < 1e-5 and abs(float(dist[0, 1]) - 3.92) < 1e-5, f"distances {dist[0].tolist()}")
    require(abs(float(h[0]) - 2.2) < 1e-5, f"hinge of (3.6, 3.6) is {float(h[0])}, expected 2.2")
    require(float(h[1]) == 0.0, f"hinge of (2.1, 2.2) is {float(h[1])}, expected 0")
    # an inactive row is ignored even when it is the closest
    mask = torch.tensor([True, False, True])
    hm = s3.hinge(dist.masked_fill(~mask, float("inf")), torch.tensor([0, 0]), delta=1.0)
    require(abs(float(hm[0]) - max(0.0, 5.12 - float(dist[0, 2]) + 1.0)) < 1e-5, "masked row was not ignored")
    return (f"(3.6, 3.6): d_t=5.12, nearest other 3.92 -> h={float(h[0]):.4f}; (2.1, 2.2): h={float(h[1]):.1f}; "
            f"with row 1 inactive: h={float(hm[0]):.4f}")


@check(6, "Step 3: aggregation of identical client encoders returns the same encoder")
def check_step3_aggregate_identity(args):
    from unlearning import step3_shrink_repair as s3
    ref, phi0, q, weights = step3_inputs(args)
    enc = {k: v for k, v in ref.state_dict().items() if helpers.is_encoder_key(k)}
    w = [weights[k] for k in sorted(weights)]
    out = s3.aggregate([enc, enc, enc], w)
    diff = [k for k in enc if not torch.equal(out[k], enc[k])]
    require(not diff, f"{len(diff)} tensors changed, e.g. {diff[:2]}")
    return f"{len(enc)} tensors bit-identical with weights {[round(x, 4) for x in w]}"


@check(6, "Step 3: shrink and repair alone on one client, one shrink")
def check_step3_single(args):
    import tempfile as tf
    from unlearning import step3_shrink_repair as s3
    ref, phi0, q, weights = step3_inputs(args)
    k = helpers.remaining_clients(args.withdraw)[0]
    with tf.TemporaryDirectory() as tmp:
        records = helpers.JsonlWriter(os.path.join(tmp, "repairs.jsonl"))
        wk = s3.ClientWorker(k, ref, phi0, q, records)
        wk.start_round(1, 1)
        before = wk.encoder_state()
        s3.shrink(wk.encoder, phi0)
        after_shrink = wk.encoder_state()
        for name in before:
            expected = phi0[name] + settings.SHRINK_FACTOR * (before[name] - phi0[name])
            require(torch.allclose(after_shrink[name], expected, rtol=0, atol=1e-7), f"shrink wrong for {name}")
        rec = wk.repair(shrink_number=1)
        with open(os.path.join(tmp, "repairs.jsonl")) as f:
            saved = json.loads(f.readline())
    require(saved == json.loads(json.dumps(rec)), "repair record not written")
    require(rec["batches"] <= rec["cap"] and rec["done"] != rec["stalled"], "inconsistent repair record")
    require(all(set(c) == {"batches", "flip_top", "flip_bottom"} for c in rec["checks"]), "bad check entries")
    changed = any(not torch.equal(after_shrink[n], v) for n, v in wk.encoder_state().items())
    require(changed, "repair did not update the encoder")
    last = rec["checks"][-1]
    return (f"client {k}: shrink exact; repair used {rec['batches']}/{rec['cap']} batches, "
            f"flip rate at checks " + ", ".join("%.1f%%/%.1f%%" % (100 * c["flip_top"], 100 * c["flip_bottom"]) for c in rec["checks"])
            + f" (top/bottom) -> {'done' if rec['done'] else 'stalled'}")


@check(6, "Step 3: during a repair, codebooks, masks, reference model and decoders are unchanged")
def check_step3_repair_side_effects(args):
    import tempfile as tf
    from unlearning import step3_shrink_repair as s3
    ref, phi0, q, weights = step3_inputs(args)
    k = helpers.remaining_clients(args.withdraw)[0]
    ref_before = {n: v.clone() for n, v in ref.state_dict().items()}
    q_before = {n: (v.clone() if torch.is_tensor(v) else v) for n, v in q.items()}
    phi0_before = {n: v.clone() for n, v in phi0.items()}
    helpers._ACCESS_LOG.clear()
    with tf.TemporaryDirectory() as tmp:
        wk = s3.ClientWorker(k, ref, phi0, q, helpers.JsonlWriter(os.path.join(tmp, "r.jsonl")))
        wk.start_round(1, 1)
        start = wk.encoder_state()
        wk.repair(shrink_number=0)
    require(same_tensors(ref_before, ref.state_dict()), "reference model changed")
    for n, v in q_before.items():
        require(torch.equal(v, q[n]) if torch.is_tensor(v) else v == q[n], f"{n} changed")
    require(same_tensors(phi0_before, phi0), "phi0 changed")
    require(not any(n.startswith(helpers.DECODER_PREFIX) for n, _ in wk.encoder.named_parameters()), "decoder in the working model")
    kinds = {kind for kind, _, _ in helpers.access_log()}
    require(not any("decoder" in kd for kd in kinds), f"a decoder was opened: {kinds}")
    require(any(not torch.equal(start[n], v) for n, v in wk.encoder_state().items()), "the working encoder did not change")
    return ("reference model, clean codebooks, masks, margins and phi0 bit-identical after a repair; "
            "no decoder built or opened; only the working encoder changed")


@check(6, "Step 3: repair with no shrink, old encoder and old codebook as target")
def check_step3_repair_zero_flips(args):
    import tempfile as tf
    from unlearning import step3_shrink_repair as s3
    ref, phi0, q, weights = step3_inputs(args)
    q_old = old_codebook_targets(ref, q)
    q_old["delta_top"] = q_old["delta_bot"] = 0.0
    lines = []
    with tf.TemporaryDirectory() as tmp:
        for k in helpers.remaining_clients(args.withdraw):
            wk = s3.ClientWorker(k, ref, phi0, q_old, helpers.JsonlWriter(os.path.join(tmp, "r.jsonl")))
            wk.start_round(1, 1)
            start = wk.encoder_state()
            rec = wk.repair(shrink_number=0)
            first = rec["checks"][0]
            require(first["flip_top"] == 0 and first["flip_bottom"] == 0, f"client {k}: flips {first}")
            require(rec["done"] and len(rec["checks"]) == 1, f"client {k}: not done at the first check")
            require(same_tensors(start, wk.encoder_state()), f"client {k}: encoder moved although nothing flipped")
            lines.append(f"client {k}: 0 flips, done after {rec['batches']} batches")
    return "margin 0: " + "; ".join(lines) + "; encoder unchanged (zero hinge, zero gradient)"


@check(6, "Step 3: thirty shrinks with repairs switched off")
def check_step3_thirty_shrinks(args):
    import tempfile as tf
    from unlearning import step3_shrink_repair as s3
    ref, phi0, q, weights = step3_inputs(args)
    old = {n: v.detach().clone() for n, v in ref.named_parameters() if helpers.is_encoder_key(n)}
    with tf.TemporaryDirectory() as tmp:
        summary = s3.run_step3(args.withdraw, out_dir=tmp, repair_enabled=False, shrink_rounds=6, shrinks_per_round=5)
        got = helpers.load_file(os.path.join(tmp, "encoder_unlearned.pth"))
        rounds_saved = sorted(d for d in os.listdir(tmp) if d.startswith("round_"))
    factor = settings.SHRINK_FACTOR ** 30
    # "To float precision": at most one float32 rounding of the weight size per operation
    # (30 shrinks + 6 aggregations), measured against the exact value in float64.
    eps, ops = torch.finfo(torch.float32).eps, 36
    worst = 0.0
    w_list = [weights[k] for k in sorted(weights)]
    for n in old:
        p0, o = phi0[n].cpu(), old[n].cpu()
        expected = p0.double() + factor * (o.double() - p0.double())
        scale = float(torch.maximum(p0.abs().max(), o.abs().max()))
        worst = max(worst, float((got[n].double() - expected).abs().max()) / (eps * scale))
        # The same 30 shrinks and 6 aggregations replayed directly in float32 give the identical result
        x = o.clone()
        for _ in range(6):
            for _ in range(5):
                x = p0 + settings.SHRINK_FACTOR * (x - p0)
            x = s3.aggregate([{"x": x}] * len(w_list), w_list)["x"]
        require(torch.equal(x, got[n]), f"{n}: loop result differs from a direct float32 replay")
    require(worst <= ops, f"encoder - phi0 differs from 0.9^30 (old - phi0) by {worst:.1f} roundings (bound {ops})")
    require(all(v == [5] * 6 for v in summary["shrinks"].values()), f"shrink counts {summary['shrinks']}")
    require(abs(summary["leftover_rho"] - factor) < 1e-12, f"rho {summary['leftover_rho']} != 0.9^30")
    require(rounds_saved == sorted([f"round_{r}" for r in range(1, 7)] + ["round_closing"]), f"rounds saved {rounds_saved}")
    return (f"6 rounds x 5 shrinks at {len(summary['clients'])} clients, aggregated each round: "
            f"encoder - phi0 = 0.9^30 (old - phi0) within {worst:.1f} float32 roundings of the weights (bound {ops}); "
            f"bit-identical to a direct float32 replay; rho = {summary['leftover_rho']:.6f}")


@check(6, "Step 3: round loop with the real settings (fixture schedule)")
def check_step3_real_run(args):
    from unlearning import step3_shrink_repair as s3
    step3_inputs(args)
    w = args.withdraw
    out = settings.step_dir(w, 3)
    jsonl = os.path.join(settings.logs_dir(w), "repairs.jsonl")
    n_before = count_lines(jsonl)
    helpers._ACCESS_LOG.clear()
    summary = s3.run_step3(w, overwrite=True)
    STATE["step3_summary"] = summary
    clients = helpers.remaining_clients(w)
    require(all(k != w for _, k, _ in helpers.access_log()), f"client {w} was accessed")
    for name in ("encoder_unlearned.pth", "step3_summary.json"):
        require(os.path.exists(os.path.join(out, name)), f"{name} missing")
    rounds = sorted(d for d in os.listdir(out) if d.startswith("round_"))
    for r in rounds:
        files = set(os.listdir(os.path.join(out, r)))
        require(files == {"global_encoder.pth", "state.json"} | {f"optimizer_client_{k}.pt" for k in clients},
                f"{r}: files {sorted(files)}")
    n = summary["shrinks"]
    rho = 1.0
    for r in range(summary["shrink_rounds_done"]):
        rho *= sum(summary["weights"][k] * settings.SHRINK_FACTOR ** n[k][r] for k in clients)
    require(abs(rho - summary["leftover_rho"]) < 1e-12, "rho inconsistent with the shrink counts")
    total = sum(sum(v) for v in n.values())
    require(set(summary["final_flip_rates"]) == set(clients), "final flip rates missing")
    repairs = count_lines(jsonl) - n_before
    require(repairs > 0, "no repair records written")
    stall_rounds = [r + 1 for r in range(summary["shrink_rounds_done"]) if any(summary["stalled"][k][r] for k in clients)]
    return (f"rounds saved {rounds}; shrinks per client {n}; stalls in round(s) {stall_rounds or 'none'}; "
            f"{repairs} repair records; rho = {summary['leftover_rho']:.4f} (= 0.9^shrinks done, weighted); final flip rates "
            + ", ".join("client %d %.1f%%/%.1f%%" % (k, 100 * v["top"], 100 * v["bottom"]) for k, v in summary["final_flip_rates"].items()))


@check(6, "Step 3: loop bookkeeping when every repair succeeds (path test, threshold raised in-process)")
def check_step3_success_path(args):
    import tempfile as tf
    from unlearning import step3_shrink_repair as s3
    step3_inputs(args)
    with settings_override(FLIP_THRESHOLD=1.0), tf.TemporaryDirectory() as tmp:
        summary = s3.run_step3(args.withdraw, out_dir=tmp)
        rounds = sorted(d for d in os.listdir(tmp) if d.startswith("round_"))
        enc = helpers.load_file(os.path.join(tmp, "encoder_unlearned.pth"))
        STATE["success_path"] = (summary, enc)
    R, S = settings.SHRINK_ROUNDS, settings.SHRINKS_PER_ROUND
    require(all(v == [S] * R for v in summary["shrinks"].values()), f"shrink counts {summary['shrinks']}")
    require(not summary["stopped_early"], "stopped early")
    require(abs(summary["leftover_rho"] - settings.SHRINK_FACTOR ** (R * S)) < 1e-12, f"rho {summary['leftover_rho']}")
    require(rounds == sorted([f"round_{r}" for r in range(1, R + 1)] + ["round_closing"]), f"rounds {rounds}")
    return f"{R} rounds x {S} shrinks, no stall, rounds saved {rounds}, rho = 0.9^{R * S} = {summary['leftover_rho']:.4f}"


@check(6, "Step 3: --resume reproduces an uninterrupted run")
def check_step3_resume(args):
    import tempfile as tf
    from unlearning import step3_shrink_repair as s3
    step3_inputs(args)
    lines = []
    # (a) success path: stop after round 1, resume
    require("success_path" in STATE, "needs the success-path check")
    full_summary, full_enc = STATE["success_path"]
    with settings_override(FLIP_THRESHOLD=1.0), tf.TemporaryDirectory() as tmp:
        require(s3.run_step3(args.withdraw, out_dir=tmp, stop_after_round=1) is None, "did not stop after round 1")
        require(not os.path.exists(os.path.join(tmp, "encoder_unlearned.pth")), "finished although stopped")
        try:
            s3.run_step3(args.withdraw, out_dir=tmp)
            raise CheckFailed("an earlier run was overwritten without --resume")
        except FileExistsError:
            pass
        resumed = s3.run_step3(args.withdraw, out_dir=tmp, resume=True)
        enc = helpers.load_file(os.path.join(tmp, "encoder_unlearned.pth"))
    require(same_tensors(enc, full_enc), "resumed encoder differs from the uninterrupted run")
    require(resumed["shrinks"] == full_summary["shrinks"] and resumed["leftover_rho"] == full_summary["leftover_rho"],
            "resumed bookkeeping differs")
    lines.append("success path: stopped after round 1, resumed -> encoder bit-identical, same shrink counts and rho")
    # (b) real settings: stop after round 1, resume; compare with the real run
    require("step3_summary" in STATE, "needs the real-run check")
    real_enc = helpers.load_file(os.path.join(settings.step_dir(args.withdraw, 3), "encoder_unlearned.pth"))
    with tf.TemporaryDirectory() as tmp:
        s3.run_step3(args.withdraw, out_dir=tmp, stop_after_round=1)
        resumed = s3.run_step3(args.withdraw, out_dir=tmp, resume=True)
        enc = helpers.load_file(os.path.join(tmp, "encoder_unlearned.pth"))
    require(same_tensors(enc, real_enc), "resumed encoder differs from the real run")
    require(resumed["final_flip_rates"] == STATE["step3_summary"]["final_flip_rates"], "final flip rates differ")
    lines.append("real settings: stopped after round 1, resumed -> bit-identical to the uninterrupted run")
    lines.append("a rerun without --resume is refused")
    return "; ".join(lines)


# -----------------
# Phase 7 (Step 4)
# -----------------
def step4_client_maps(args, k):
    """One batch of client k: old code maps (reference encoder, deployed codebook) per level, and the deployed model."""
    from unlearning import step4_decoder_fit as s4
    require_results_dir()
    helpers.set_withdrawn(args.withdraw)
    ref = helpers.load_reference_model()
    model = helpers.load_deployed_model(k)
    batch = next(iter(helpers.make_loader(k, "train", shuffle=False, drop_last=False)))
    with torch.no_grad():
        z = dict(zip(("top", "bottom"), ref.encode(batch["lr"].to(settings.DEVICE))))
        maps = {lvl: s4.code_map(z[lvl], helpers.quantizer(model, lvl)._embedding.weight.detach())
                for lvl in helpers.LEVELS}
    return model, maps


@check(7, "Step 4: W times an unfolded patch equals the conv layer's output")
def check_step4_patch_order(args):
    from unlearning import step4_decoder_fit as s4
    k = helpers.remaining_clients(args.withdraw)[0]
    model, maps = step4_client_maps(args, k)
    g = torch.Generator().manual_seed(9)
    lines = []
    with torch.no_grad():
        for lvl in helpers.LEVELS:
            conv = helpers.decoder_entry_conv(model, lvl)
            W = s4.layer_matrix(conv)
            for label, x in (("code map", maps[lvl][:2]),
                             ("random map", torch.randn(2, conv.in_channels, 11, 13, generator=g).to(W.device))):
                out = conv(x)
                worst = 0.0
                for i in range(x.shape[0]):
                    via_w = (W @ s4.patches(x[i:i + 1], conv).t()).view(out.shape[1:])
                    worst = max(worst, float((via_w - out[i]).abs().max() / out[i].abs().max()))
                require(worst < 1e-5, f"{lvl} {label}: W x patch differs from the conv output (relative {worst:.1e})")
            lines.append(f"{lvl}: W is {W.shape[0]}x{W.shape[1]}, matches the conv on code maps and random maps")
    return "; ".join(lines)


@check(7, "Step 4: fit with new input equal to old input gives W' = W")
def check_step4_identity_fit(args):
    from unlearning import step4_decoder_fit as s4
    k = helpers.remaining_clients(args.withdraw)[0]
    model, maps = step4_client_maps(args, k)
    lines = []
    for lvl in helpers.LEVELS:
        conv = helpers.decoder_entry_conv(model, lvl)
        W = s4.layer_matrix(conv)
        m = s4.PatchMoments(W.shape[1], W.device)
        m.add(maps[lvl], maps[lvl], conv, torch.Generator().manual_seed(0))
        W_new, lam = m.solve(W)
        rel = float((W_new - W.double()).norm() / W.double().norm())
        require(rel < 1e-9, f"{lvl}: W' differs from W by {rel:.1e}")
        require(m.residual(W_new, W) < 1e-9, f"{lvl}: residual {m.residual(W_new, W)}")
        lines.append(f"{lvl}: |W'-W|/|W| = {rel:.1e} over {m.count:,} patches (lambda {lam:.2e})")
    return "; ".join(lines)


@check(7, "Step 4: fit recovers an exact linear change of the code vectors")
def check_step4_linear_change(args):
    from unlearning import step4_decoder_fit as s4
    k = helpers.remaining_clients(args.withdraw)[0]
    model, maps = step4_client_maps(args, k)
    g = torch.Generator().manual_seed(1)
    lines = []
    for lvl in helpers.LEVELS:
        conv = helpers.decoder_entry_conv(model, lvl)
        W = s4.layer_matrix(conv)
        D = conv.in_channels
        M = (torch.eye(D) + 0.5 * torch.randn(D, D, generator=g) / D ** 0.5).to(W.device)  # invertible mix of channels
        new = torch.einsum("ij,bjhw->bihw", M, maps[lvl]) + 0.3                         # plus a shift
        m = s4.PatchMoments(W.shape[1], W.device)
        m.add(maps[lvl], new, conv, torch.Generator().manual_seed(0))
        W_new, _ = m.solve(W)
        before, after = m.residual(W, W), m.residual(W_new, W)
        require(after < 1e-3 * max(before, 1e-12) or after < 1e-6,
                f"{lvl}: feature error {before:.3e} -> {after:.3e}; the solve does not recover the change")
        lines.append(f"{lvl}: feature error {before:.3f} -> {after:.1e}")
    return "; ".join(lines) + " (channel mix + shift undone by the refit layer)"


@check(7, "Step 4: decoder fit on the fixture writes the expected models")
def check_step4_run(args):
    from unlearning import step4_decoder_fit as s4
    from unlearning.step2_server_quantities import quantities_path
    require_results_dir()
    w = args.withdraw
    if not os.path.exists(s4.unlearned_encoder_path(w)):
        raise CheckFailed("Step 3 output missing (run the Phase 6 checks first)")
    helpers._ACCESS_LOG.clear()
    summary = s4.run_step4(w, overwrite=True)
    clients = helpers.remaining_clients(w)
    require(all(k != w for _, k, _ in helpers.access_log()), f"client {w} was accessed")
    files = sorted(f for f in os.listdir(settings.step_dir(w, 4)) if f.endswith(".pth"))
    require(files == sorted(f"fit_client_{k}.pth" for k in clients), f"files {files}")
    enc = helpers.load_file(s4.unlearned_encoder_path(w))
    q = helpers.load_file(quantities_path(w))
    lines = []
    for k in clients:
        sd = helpers.load_file(s4.fit_path(w, k))
        m = models.VQVAE2()
        m.load_state_dict(sd)  # strict: the file holds the complete model
        deployed = helpers.load_file(helpers.deployed_model_path(k))
        for name, v in enc.items():
            require(torch.equal(sd[name], v), f"client {k}: {name} is not the unlearned encoder")
        for lvl, (cb_key, mask_key) in zip(("top", "bot"), zip(helpers.CODEBOOK_KEYS, helpers.MASK_KEYS)):
            require(torch.equal(sd[cb_key], q[f"V_{lvl}"]), f"client {k}: {cb_key} is not the clean codebook")
            require(torch.equal(sd[mask_key], q[f"mask_{lvl}"]), f"client {k}: {mask_key} is not the mask")
        entry = {f"decoder.{c}.{p}" for c in ("pre_conv_top", "pre_conv_bottom") for p in ("weight", "bias")}
        for name, v in sd.items():
            if name.startswith(helpers.DECODER_PREFIX):
                same = torch.equal(v, deployed[name])
                require(same != (name in entry), f"client {k}: {name} {'unchanged' if same else 'changed'}")
        require(all(float(sd[f"{qn}.code_usage"].abs().sum()) == 0 for qn in ("quantizer_top", "quantizer_bottom")),
                f"client {k}: usage counters not cleared")
        rep = summary["per_client"][k]
        lines.append(f"client {k}: feature error top {rep['top']['residual_before']:.3f}->{rep['top']['residual_after']:.3f}, "
                     f"bottom {rep['bottom']['residual_before']:.3f}->{rep['bottom']['residual_after']:.3f}")
    return (f"fit_client_k.pth for {clients} only; each = unlearned encoder + clean codebooks + masks + deployed decoder "
            f"with only the two entry layers changed; " + "; ".join(lines))


SETUP = {1: setup_phase1}


# -----------------
# Runner
# -----------------
def main():
    parser = argparse.ArgumentParser(description="Unlearning correctness checks")
    settings.add_common_args(parser)
    parser.add_argument("--phase", type=int, action="append",
                        help="Phase(s) to check; default: every phase implemented so far")
    parser.add_argument("--withdraw", type=int, default=3, help="Withdrawn client for the per-withdrawal checks")
    parser.add_argument("--original-models", default=None,
                        help="Path to an unmodified models.py; enables the Phase 1 equivalence check")
    args = parser.parse_args()
    if args.original_models and not os.path.exists(args.original_models):
        parser.error(f"--original-models: {args.original_models} not found")
    if args.fixture or args.results_dir:
        settings.configure(results_dir=args.results_dir, fixture=args.fixture)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    for h in logging.getLogger().handlers:
        h.setLevel(logging.WARNING)  # keep the console quiet when a check raises the root level
    phases = args.phase or sorted(CHECKS)
    failures, skipped = 0, 0
    for phase in phases:
        print(f"\n=== Phase {phase} checks ===", flush=True)
        if phase in SETUP:
            SETUP[phase](args)
        for name, fn in CHECKS.get(phase, []):
            try:
                detail = fn(args)
                print(f"PASS  {name}\n      {detail}", flush=True)
            except CheckSkipped as e:
                skipped += 1
                print(f"SKIP  {name}\n      {e}", flush=True)
            except Exception as e:  # report every failure, keep going
                failures += 1
                print(f"FAIL  {name}\n      {type(e).__name__}: {e}", flush=True)
    summary = "ALL CHECKS PASSED" if failures == 0 else f"{failures} CHECK(S) FAILED"
    print(f"\n{summary}" + (f" ({skipped} skipped)" if skipped else ""))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
