# unlearning/run.py
"""
Run Steps 1 to 6 in order for one withdrawn client, stopping at the first failure.
Step 0 (initial weights) is run separately, once per trained model.

    python -m unlearning.run --fixture --withdraw 3
    python -m unlearning.run --results-dir FedDSR_results --withdraw 3 --trial
    python -m unlearning.run --results-dir FedDSR_results --withdraw 3 --resume
    python -m unlearning.run --results-dir FedDSR_results --withdraw 3 --steps 4,5,6 --overwrite
"""
import argparse
import logging
import os
import sys
import time

from unlearning import settings
from unlearning import helpers
from unlearning.step0_initial_weights import phi0_path
from unlearning.step1_statistics import run_step1, stats_path
from unlearning.step2_server_quantities import run_step2, quantities_path
from unlearning.step3_shrink_repair import run_step3
from unlearning.step4_decoder_fit import run_step4, fit_path, unlearned_encoder_path
from unlearning.step5_finetune import run_step5, finetuned_path
from unlearning.step6_release import release, report_cleanup, release_dir, released_client_path

STEP_NAMES = {1: "statistics", 2: "server quantities", 3: "shrink and repair",
              4: "decoder fit", 5: "fine-tuning", 6: "release"}


def step_outputs(step, withdrawn):
    """The files a step must leave behind."""
    clients = helpers.remaining_clients(withdrawn)
    if step == 1:
        return [stats_path(withdrawn, k) for k in clients]
    if step == 2:
        return [quantities_path(withdrawn)]
    if step == 3:
        return [unlearned_encoder_path(withdrawn), os.path.join(settings.step_dir(withdrawn, 3), "step3_summary.json")]
    if step == 4:
        return [fit_path(withdrawn, k) for k in clients]
    if step == 5:
        return [finetuned_path(withdrawn, k) for k in clients]
    if step == 6:
        out = release_dir(withdrawn)
        return ([os.path.join(out, n) for n in ("best_model.pth", "phi0.pth", "summary.json")]
                + [released_client_path(withdrawn, k) for k in clients])
    raise ValueError(step)


def prerequisites(step, withdrawn):
    """Files that must exist before a step starts: the training output, phi0 and every earlier step's output."""
    needed = [os.path.join(settings.training_dir(), "best_model.pth")]
    if step >= 3:
        needed.append(phi0_path())
    for earlier in range(1, step):
        needed += step_outputs(earlier, withdrawn)
    return needed


def run_one(step, withdrawn, resume, overwrite):
    if step == 1:
        run_step1(withdrawn, overwrite=overwrite)
    elif step == 2:
        run_step2(withdrawn, overwrite=overwrite)
    elif step == 3:
        run_step3(withdrawn, resume=resume, overwrite=overwrite)
    elif step == 4:
        run_step4(withdrawn, overwrite=overwrite)
    elif step == 5:
        run_step5(withdrawn, overwrite=overwrite)
    elif step == 6:
        release(withdrawn, overwrite=overwrite)
        report_cleanup(withdrawn)


def run_pipeline(withdrawn, steps=(1, 2, 3, 4, 5, 6), resume=False, overwrite=False):
    """Runs the steps in order; raises on the first failure. Returns the time per step."""
    logger = logging.getLogger()
    helpers.set_withdrawn(withdrawn)
    steps = sorted(set(steps))
    finished = all(os.path.exists(p) for p in step_outputs(6, withdrawn))
    if finished and not (overwrite or resume):
        raise FileExistsError(f"{settings.withdraw_dir(withdrawn)} holds a finished run; pass --overwrite to redo it")

    logger.info(f"Unlearning run: withdrawn client {withdrawn} ({settings.client_name(withdrawn)}), "
                f"remaining {helpers.remaining_clients(withdrawn)}, steps {steps}, results {settings.RESULTS_DIR}"
                + (" [fixture]" if settings.FIXTURE else "") + (" [trial]" if settings.TRIAL else ""))
    timings = {}
    for step in steps:
        missing = [p for p in prerequisites(step, withdrawn) if not os.path.exists(p)]
        if missing:
            hint = "run Step 0 first" if phi0_path() in missing else "run the earlier steps first"
            raise FileNotFoundError(f"Step {step} cannot start, {missing[0]} is missing ({hint})")
        if resume and step != 3 and all(os.path.exists(p) for p in step_outputs(step, withdrawn)):
            logger.info(f"=== Step {step} ({STEP_NAMES[step]}): output complete, reused ===")
            continue
        logger.info(f"=== Step {step} ({STEP_NAMES[step]}) ===")
        start = time.time()
        run_one(step, withdrawn, resume=resume, overwrite=overwrite)
        missing = [p for p in step_outputs(step, withdrawn) if not os.path.exists(p)]
        if missing:
            raise RuntimeError(f"Step {step} finished without writing {missing[0]}")
        timings[step] = time.time() - start
        logger.info(f"=== Step {step} done in {timings[step]:.1f} s ===")
    logger.info("Unlearning run complete: " + ", ".join(f"step {s} {t:.0f} s" for s, t in timings.items()))
    return timings


def parse_steps(text):
    steps = [int(s) for s in text.split(",") if s.strip()]
    if not steps or any(s not in STEP_NAMES for s in steps):
        raise argparse.ArgumentTypeError("steps must be a comma-separated list of 1..6")
    return steps


def main():
    parser = argparse.ArgumentParser(description="Unlearn one withdrawn client: Steps 1 to 6")
    settings.add_common_args(parser)
    parser.add_argument("--withdraw", type=int, required=True, help="Withdrawn client id (0 to 3)")
    parser.add_argument("--trial", action="store_true",
                        help="Real model and data, but 64 slices per client and 1 round of 2 shrinks")
    parser.add_argument("--steps", type=parse_steps, default=[1, 2, 3, 4, 5, 6],
                        help="Comma-separated steps to run, reusing earlier outputs (e.g. 4,5,6)")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse finished steps and continue Step 3 from its last finished round")
    parser.add_argument("--overwrite", action="store_true", help="Redo steps whose output already exists")
    args = parser.parse_args()
    settings.configure(results_dir=args.results_dir, fixture=args.fixture, trial=args.trial)
    helpers.setup_logging(settings.logs_dir(args.withdraw), "run")
    try:
        run_pipeline(args.withdraw, steps=args.steps, resume=args.resume, overwrite=args.overwrite)
    except Exception as e:
        logging.getLogger().exception(f"Unlearning run stopped: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
