# unlearning/step5_finetune.py
"""
Step 5: decoder fine-tuning at each remaining client, with the existing Stage-2 code.

The model from Step 4 is fine-tuned with freeze_encoder_quantizer and finetune_decoder from
"Stage2 decoder finetunning.py". Vitalization is switched off first, so the Stage-2 call to
model.train() can neither overwrite a codebook row nor bring a masked row back.

    python -m unlearning.step5_finetune --fixture --withdraw 3
"""
import argparse
import logging
import os
import sys

import torch

from unlearning import settings
from unlearning import helpers
from unlearning.step4_decoder_fit import fit_path
from models import VQVAE2


def finetuned_path(withdrawn, k):
    return os.path.join(settings.step_dir(withdrawn, 5), f"finetuned_client_{k}.pth")


def frozen_part(model):
    """Everything fine-tuning must leave untouched: encoder weights, both codebooks, both masks."""
    sd = model.state_dict()
    keys = [k for k in sd if helpers.is_encoder_key(k)] + list(helpers.CODEBOOK_KEYS) + list(helpers.MASK_KEYS)
    return {k: sd[k].detach().clone() for k in keys}


def finetune_client(k, withdrawn, stage2):
    logger = logging.getLogger()
    device = settings.DEVICE
    model = VQVAE2()
    model.load_state_dict(helpers.load_file(fit_path(withdrawn, k)))  # strict: Step 4 saved the complete model
    model.to(device)
    for level in helpers.LEVELS:
        helpers.quantizer(model, level).vitalize = False

    before = frozen_part(model)
    stage2.freeze_encoder_quantizer(model, logger, k)
    loader = helpers.make_loader(k, "train", shuffle=True, drop_last=True)
    torch.manual_seed(settings.SEED + k)
    model = stage2.finetune_decoder(model, loader, k, epochs=settings.FINETUNE_EPOCHS, lr=settings.FINETUNE_LR,
                                    device=device, logger=logger, results_dir=settings.step_dir(withdrawn, 5))
    stage2.unfreeze_all(model)

    after = frozen_part(model)
    changed = [name for name in before if not torch.equal(before[name], after[name])]
    if changed:
        raise AssertionError(f"Client {k}: fine-tuning changed frozen tensors: {changed[:5]}")
    return model


def run_step5(withdrawn, overwrite=False):
    logger = logging.getLogger()
    helpers.set_withdrawn(withdrawn)
    clients = helpers.remaining_clients(withdrawn)
    if not overwrite:
        existing = [finetuned_path(withdrawn, k) for k in clients if os.path.exists(finetuned_path(withdrawn, k))]
        if existing:
            raise FileExistsError(f"Step 5 output exists ({existing[0]}); pass --overwrite to redo it")
    missing = [fit_path(withdrawn, k) for k in clients if not os.path.exists(fit_path(withdrawn, k))]
    if missing:
        raise FileNotFoundError(f"{missing[0]} missing: run Step 4 first")

    stage2 = helpers.load_stage2_module()
    logger.info(f"Step 5: withdrawn client {withdrawn}; fine-tuning at clients {clients}, "
                f"{settings.FINETUNE_EPOCHS} epoch(s) at lr {settings.FINETUNE_LR:g}, vitalization off")
    for k in clients:
        model = finetune_client(k, withdrawn, stage2)
        helpers.atomic_save({n: v.detach().cpu() for n, v in model.state_dict().items()}, finetuned_path(withdrawn, k))
        logger.info(f"  Client {k}: encoder, codebooks and masks unchanged; wrote {finetuned_path(withdrawn, k)}")
    return clients


def main():
    parser = argparse.ArgumentParser(description="Step 5: decoder fine-tuning with the Stage-2 code")
    settings.add_common_args(parser)
    parser.add_argument("--withdraw", type=int, required=True, help="Withdrawn client id")
    parser.add_argument("--trial", action="store_true", help="Real model, first slices only")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    settings.configure(results_dir=args.results_dir, fixture=args.fixture, trial=args.trial)
    helpers.setup_logging(settings.logs_dir(args.withdraw), "step5_finetune")
    try:
        run_step5(args.withdraw, overwrite=args.overwrite)
    except (FileExistsError, FileNotFoundError) as e:
        logging.getLogger().error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
