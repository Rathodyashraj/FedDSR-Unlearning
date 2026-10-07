# unlearning/helpers.py
"""
Shared helpers for the unlearning steps: parameter groups, model loading,
distances, loaders, logging, and the guard that keeps the withdrawn client untouched.
"""
import json
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset

import config
from unlearning import settings
from data import get_dataloaders, safe_collate
from models import VQVAE2, VectorQuantizer

# -----------------
# Parameter groups
# -----------------
ENCODER_PREFIXES = (
    "encoder_top.",
    "pre_quant_conv_top.",
    "encoder_bottom.",
    "pre_quant_conv_bottom.",
)
CODEBOOK_KEYS = (
    "quantizer_top._embedding.weight",
    "quantizer_bottom._embedding.weight",
)
MASK_KEYS = (
    "quantizer_top.active_mask",
    "quantizer_bottom.active_mask",
)
DECODER_PREFIX = "decoder."

# Level name -> (quantizer attribute, decoder entry conv attribute)
LEVELS = {
    "top": ("quantizer_top", "pre_conv_top"),
    "bottom": ("quantizer_bottom", "pre_conv_bottom"),
}


def is_encoder_key(name):
    return name.startswith(ENCODER_PREFIXES)


def split_state_dict(state_dict):
    """Split a state dict into encoder, codebook, decoder and other (buffers such as counters and masks)."""
    parts = {"encoder": {}, "codebook": {}, "decoder": {}, "other": {}}
    for name, value in state_dict.items():
        if is_encoder_key(name):
            parts["encoder"][name] = value
        elif name in CODEBOOK_KEYS:
            parts["codebook"][name] = value
        elif name.startswith(DECODER_PREFIX):
            parts["decoder"][name] = value
        else:
            parts["other"][name] = value
    return parts


def encoder_state(state_dict):
    return split_state_dict(state_dict)["encoder"]


def encoder_parameters(model):
    """The encoder weights only (the four encoder groups, never the codebooks)."""
    return [p for name, p in model.named_parameters() if is_encoder_key(name)]


def quantizer(model, level):
    return getattr(model, LEVELS[level][0])


def decoder_entry_conv(model, level):
    return getattr(model.decoder, LEVELS[level][1])


# -----------------
# Clients and the withdrawn-client guard
# -----------------
_WITHDRAWN = None
_ACCESS_LOG = []  # (kind, client, path) for every client-specific read


def all_clients():
    """Client ids in training order, as main.py builds them."""
    return list(config.CLIENT_DATASETS.keys())[:config.NUM_CLIENTS]


def set_withdrawn(withdrawn):
    """Register the withdrawn client. From now on no data, decoder or statistics of it can be opened."""
    global _WITHDRAWN
    if withdrawn not in all_clients():
        raise ValueError(f"Withdrawn client must be one of {all_clients()}, got {withdrawn}")
    _WITHDRAWN = withdrawn


def remaining_clients(withdrawn):
    """Client ids that take part. Every loop over clients uses this."""
    if withdrawn not in all_clients():
        raise ValueError(f"Withdrawn client must be one of {all_clients()}, got {withdrawn}")
    return [k for k in all_clients() if k != withdrawn]


def check_client_access(k, kind, path=None):
    """Refuse any client-specific read for the withdrawn client, and record every allowed one."""
    if _WITHDRAWN is not None and k == _WITHDRAWN:
        raise PermissionError(f"Refused to open {kind} of withdrawn client {k}")
    _ACCESS_LOG.append((kind, k, path))
    logging.getLogger().info(f"[access] {kind} client={k}" + (f" path={path}" if path else ""))


def access_log():
    return list(_ACCESS_LOG)


# -----------------
# Paths, saving and write protection
# -----------------
def assert_writable(path):
    """Refuse to write inside the training output folders (FedMedSR/, DecoderFineTuning_Local/)."""
    target = os.path.realpath(path)
    roots = [settings.RESULTS_DIR, "FedDSR_results"] if settings.RESULTS_DIR else ["FedDSR_results"]
    for root in roots:
        for name in settings.PROTECTED_DIRNAMES:
            protected = os.path.realpath(os.path.join(root, name))
            if target == protected or target.startswith(protected + os.sep):
                raise PermissionError(f"Refused to write into protected training output: {path}")


def atomic_save(obj, path):
    """torch.save to a temporary file, then rename, so a file is only present when complete."""
    assert_writable(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_save_json(obj, path):
    assert_writable(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def load_file(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=False)


# -----------------
# Model loading
# -----------------
def _load_into_model(model, state_dict, path, require_decoder):
    """load_state_dict(strict=False), but fail loudly if an expected parameter is missing."""
    result = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in result.missing_keys
               if k in dict(model.named_parameters())
               and (require_decoder or not k.startswith(DECODER_PREFIX))]
    if missing:
        raise KeyError(f"{path} is missing {len(missing)} parameters, e.g. {missing[:3]}")
    if result.unexpected_keys:
        logging.getLogger().warning(f"{path}: ignored unexpected keys {result.unexpected_keys[:5]}")
    return model


def freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for level in LEVELS:
        quantizer(model, level).vitalize = False
    return model


def load_reference_model(device=None):
    """The old model: encoder and codebooks from best_model.pth. Eval mode, gradients off."""
    device = device or settings.DEVICE
    path = os.path.join(settings.training_dir(), "best_model.pth")
    model = VQVAE2()
    _load_into_model(model, load_file(path), path, require_decoder=False)
    logging.getLogger().info(f"Loaded reference model from {path}")
    return freeze(model.to(device))


def refined_decoder_path(k):
    return os.path.join(settings.stage2_dir(), f"best_refined_decoder_client_{k}.pth")


def best_decoder_path(k):
    return os.path.join(settings.training_dir(), f"best_decoder_client_{k}.pth")


def choose_deployed(k):
    """The rule: the Stage-2 refined file if it exists, otherwise best_decoder_client_k.pth."""
    refined = refined_decoder_path(k)
    if os.path.exists(refined):
        return refined, "stage2_refined"
    return best_decoder_path(k), "stage1_best_decoder"


def inventory_path():
    return os.path.join(settings.unlearning_root(), "inventory.json")


def deployed_model_path(k):
    """Client k's deployed file, as chosen in the inventory (or by the same rule if there is none)."""
    if os.path.exists(inventory_path()):
        with open(inventory_path()) as f:
            chosen = json.load(f).get("deployed", {}).get(str(k))
        if chosen:
            return chosen["path"]
    return choose_deployed(k)[0]


def load_deployed_model(k, device=None):
    """Client k's full deployed model (encoder, deployed codebooks, decoder)."""
    device = device or settings.DEVICE
    path = deployed_model_path(k)
    check_client_access(k, "decoder", path)
    model = VQVAE2()
    _load_into_model(model, load_file(path), path, require_decoder=True)
    logging.getLogger().info(f"Client {k}: loaded deployed model from {path}")
    return model.to(device)


def model_sizes(model):
    """Every size the steps need, read from the model (never typed in)."""
    device = next(model.parameters()).device
    lr_size = config.IMAGE_SIZE // 2
    dummy = torch.zeros(1, config.IMAGE_CHANNELS, lr_size, lr_size, device=device)
    with torch.no_grad():
        z_top, z_bottom = model.encode(dummy)
    sizes = {}
    for level, z in (("top", z_top), ("bottom", z_bottom)):
        q = quantizer(model, level)
        conv = decoder_entry_conv(model, level)
        kh, kw = conv.kernel_size
        sizes[level] = {
            "codebook_rows": q._num_embeddings,
            "latent_dim": q._embedding_dim,
            "grid": [z.shape[2], z.shape[3]],
            "positions_per_slice": z.shape[2] * z.shape[3],
            "decoder_in": conv.in_channels,
            "decoder_out": conv.out_channels,
            "kernel": [kh, kw],
            "padding": list(conv.padding),
            "patch_length": conv.in_channels * kh * kw + (1 if conv.bias is not None else 0),
        }
    sizes["lr_size"] = lr_size
    sizes["encoder_weights"] = sum(p.numel() for p in encoder_parameters(model))
    return sizes


# The training code lives one folder up from this package
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE2_FILE = "Stage2 decoder finetunning.py"


def load_stage2_module():
    """Import the Stage-2 script by path (its file name has spaces, so a plain import is impossible)."""
    import importlib.util
    path = os.path.join(PROJECT_DIR, STAGE2_FILE)
    spec = importlib.util.spec_from_file_location("stage2_decoder_finetunning", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -----------------
# Distances
# -----------------
def flatten_latent(z):
    """[B, D, H, W] -> [B*H*W, D], in the same position order as the quantizer."""
    return z.permute(0, 2, 3, 1).contiguous().view(-1, z.shape[1])


def squared_distances(z_flat, codebook):
    """Same formula as VectorQuantizer.forward, so nearest rows match exactly."""
    return VectorQuantizer.squared_distances(z_flat, codebook)


def masked_nearest(dist, mask):
    """Nearest active row per position. Inactive rows are +inf; ties go to the lowest index."""
    if not bool(mask.all()):
        dist = dist.masked_fill(~mask.to(dist.device), float("inf"))
    return torch.argmin(dist, dim=1)


# -----------------
# Loaders
# -----------------
_SPLITS = {"train": 0, "val": 1, "test": 2}


def make_loader(k, split, shuffle, drop_last, batch_size=None, limit=None, seed=None):
    """
    Loader over client k's split, built from the same file lists as get_dataloaders.
    In trial mode the split is cut to the first TRIAL_SLICES_PER_CLIENT slices.
    """
    check_client_access(k, f"data[{split}]", config.CLIENT_DATASETS[k]["path"])
    dataset = get_dataloaders(k)[_SPLITS[split]].dataset
    if limit is None and settings.TRIAL:
        limit = settings.TRIAL_SLICES_PER_CLIENT
    if limit is not None and limit < len(dataset):
        dataset = Subset(dataset, range(limit))
    generator = None
    if shuffle:
        generator = torch.Generator().manual_seed(settings.SEED + k if seed is None else seed)
    return DataLoader(
        dataset,
        batch_size=batch_size or settings.BATCH_SIZE,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=safe_collate,
        generator=generator,
    )


# -----------------
# Logging
# -----------------
def setup_logging(log_dir, run_name):
    """Root logger writing to the console and to log_dir/run_name.log."""
    assert_writable(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{run_name}.log"))
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


class JsonlWriter:
    """Appends one JSON record per line (used for per-repair records)."""

    def __init__(self, path):
        assert_writable(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path

    def write(self, record):
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
