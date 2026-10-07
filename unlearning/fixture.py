# unlearning/fixture.py
"""
The fixture: a tiny stand-in for the whole training output, built by running the real
training code (main.py, then Stage 2) on synthetic images, on CPU, in minutes.

    python -m unlearning.fixture --build

Every unlearning script started with --fixture calls apply_fixture_mode() before any
model is built (through settings.configure). The fixture folder is disposable.
"""
import argparse
import os
import shutil
from argparse import Namespace

import numpy as np
from PIL import Image

import config
from unlearning import settings

MARKER_FILE = ".unlearn_fixture"     # only folders holding this marker are ever deleted
DATA_DIRNAME = "data"

IMAGES_PER_CLIENT = 40
TRAIN_ROUNDS = 2
TRAIN_LOCAL_EPOCHS = 1
STAGE2_EPOCHS = 1
STAGE2_LR = 5e-6

# Tiny model and training settings, applied to config in place
TINY_HIDDEN = 32
TINY_RESIDUAL_LAYERS = 1
TINY_RESIDUAL_HIDDENS = 16
TINY_CODEBOOK_ROWS = 64
TINY_CODEBOOK_DIM = 16
TINY_BATCH_SIZE = 4
TINY_RESET_INTERVAL = 2              # vitalization fires during the short run

# Shortened unlearning schedule
FIXTURE_SHRINK_ROUNDS = 2
FIXTURE_SHRINKS_PER_ROUND = 2
FIXTURE_FLIP_CHECK_BATCHES = 2
FIXTURE_FINETUNE_EPOCHS = 1

PATTERNS = {0: "blobs", 1: "stripes", 2: "checks", 3: "gradients"}


def data_dir(results_dir):
    return os.path.join(results_dir, DATA_DIRNAME)


def client_data_dir(results_dir, k):
    return os.path.join(data_dir(results_dir), f"client_{k}_{PATTERNS[k]}")


def apply_fixture_mode(results_dir=None):
    """Switch config to the tiny fixture settings and the unlearning settings to the short schedule."""
    results_dir = results_dir or settings.FIXTURE_RESULTS_DIR

    for enc in (config.ENCODER_TOP_CONFIG, config.ENCODER_BOTTOM_CONFIG):
        enc["hidden_channels"] = TINY_HIDDEN
        enc["num_residual_layers"] = TINY_RESIDUAL_LAYERS
        enc["num_residual_hiddens"] = TINY_RESIDUAL_HIDDENS
    for q in (config.QUANTIZER_TOP_CONFIG, config.QUANTIZER_BOTTOM_CONFIG):
        q["num_embeddings"] = TINY_CODEBOOK_ROWS
        q["embedding_dim"] = TINY_CODEBOOK_DIM
    # DECODER_CONFIG copied the embedding sizes at import time
    config.DECODER_CONFIG["top_embedding_dim"] = config.QUANTIZER_TOP_CONFIG["embedding_dim"]
    config.DECODER_CONFIG["bottom_embedding_dim"] = config.QUANTIZER_BOTTOM_CONFIG["embedding_dim"]
    config.DECODER_CONFIG["hidden_channels"] = TINY_HIDDEN
    config.DECODER_CONFIG["num_residual_layers"] = TINY_RESIDUAL_LAYERS
    config.DECODER_CONFIG["num_residual_hiddens"] = TINY_RESIDUAL_HIDDENS

    config.BATCH_SIZE = TINY_BATCH_SIZE
    config.NUM_WORKERS = 0
    config.CODEBOOK_RESET_INTERVAL = TINY_RESET_INTERVAL
    config.NUM_IMAGES_PER_CLIENT = None
    # IMAGE_SIZE stays 256: the multi-scale SSIM loss needs images larger than 160 pixels.

    for k in config.CLIENT_DATASETS:
        config.CLIENT_DATASETS[k]["path"] = client_data_dir(results_dir, k)
        config.CLIENT_DATASETS[k]["ext"] = "png"
    config.RESULTS_DIR = results_dir

    settings.BATCH_SIZE = TINY_BATCH_SIZE
    settings.SHRINK_ROUNDS = FIXTURE_SHRINK_ROUNDS
    settings.SHRINKS_PER_ROUND = FIXTURE_SHRINKS_PER_ROUND
    settings.FLIP_CHECK_BATCHES = FIXTURE_FLIP_CHECK_BATCHES
    settings.FINETUNE_EPOCHS = FIXTURE_FINETUNE_EPOCHS
    return results_dir


# -----------------
# Synthetic data
# -----------------
def _pattern(kind, rng, size):
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    if kind == "blobs":
        img = np.zeros((size, size), np.float32)
        for _ in range(rng.integers(3, 8)):
            cx, cy = rng.random(2)
            s = rng.uniform(0.04, 0.15)
            img += rng.uniform(0.4, 1.0) * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))
    elif kind == "stripes":
        theta = rng.uniform(0, np.pi)
        freq = rng.uniform(4, 16)
        img = 0.5 + 0.5 * np.sin(2 * np.pi * freq * (x * np.cos(theta) + y * np.sin(theta)) + rng.uniform(0, 6.3))
    elif kind == "checks":
        cells = rng.integers(3, 12)
        ox, oy = rng.random(2)
        img = ((np.floor((x + ox) * cells) + np.floor((y + oy) * cells)) % 2).astype(np.float32)
        img = 0.2 + 0.6 * img
    elif kind == "gradients":
        theta = rng.uniform(0, 2 * np.pi)
        img = x * np.cos(theta) + y * np.sin(theta)
        img = img + rng.normal(0, 0.08, size=img.shape).astype(np.float32)
    else:
        raise ValueError(kind)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    return (img * 255).astype(np.uint8)


def write_synthetic_data(results_dir):
    for k, kind in PATTERNS.items():
        out = client_data_dir(results_dir, k)
        os.makedirs(out, exist_ok=True)
        rng = np.random.default_rng(1000 + k)
        for i in range(IMAGES_PER_CLIENT):
            Image.fromarray(_pattern(kind, rng, config.IMAGE_SIZE), mode="L").save(
                os.path.join(out, f"{kind}_{i:03d}.png"))


# -----------------
# Build
# -----------------
def _remove_old_fixture(results_dir):
    if not os.path.exists(results_dir):
        return
    real = os.path.realpath(results_dir)
    if real == os.path.realpath("FedDSR_results") or not os.path.exists(os.path.join(results_dir, MARKER_FILE)):
        raise PermissionError(f"{results_dir} exists but is not a fixture folder; refusing to delete it")
    shutil.rmtree(results_dir)


def build(results_dir=None):
    import matplotlib
    matplotlib.use("Agg")  # the training code saves plots; no display needed

    results_dir = results_dir or settings.FIXTURE_RESULTS_DIR
    _remove_old_fixture(results_dir)
    apply_fixture_mode(results_dir)
    os.makedirs(results_dir)
    open(os.path.join(results_dir, MARKER_FILE), "w").close()
    write_synthetic_data(results_dir)

    # Stage 1: the same code main.py runs (gpu=None keeps it on the configured device).
    # main() sets rounds, clients, aggregator and patience from its arguments; local epochs come from config.
    config.LOCAL_EPOCHS = TRAIN_LOCAL_EPOCHS
    import main as training_main
    training_main.main(Namespace(aggregator=config.AGGREGATOR, patience=TRAIN_ROUNDS + 1,
                                 rounds=TRAIN_ROUNDS, clients=len(config.CLIENT_DATASETS), gpu=None))

    # Stage 2: the existing decoder fine-tuning script
    from unlearning import helpers
    stage2 = helpers.load_stage2_module()
    stage2.main(Namespace(gpu=None, epochs=STAGE2_EPOCHS, clients=len(config.CLIENT_DATASETS), lr=STAGE2_LR))
    return results_dir


def main():
    parser = argparse.ArgumentParser(description="Build the synthetic fixture for Stage I testing")
    parser.add_argument("--build", action="store_true", help="(Re)build the fixture")
    parser.add_argument("--results-dir", default=settings.FIXTURE_RESULTS_DIR)
    args = parser.parse_args()
    if not args.build:
        parser.error("nothing to do; pass --build")
    out = build(args.results_dir)
    print(f"Fixture built in {out}")


if __name__ == "__main__":
    main()
