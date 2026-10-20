# unlearning/settings.py
"""
All unlearning settings and paths in one place.

No model size is written here: codebook rows, latent width, grid sizes and the
decoder patch length are read from the model at run time (see helpers.model_sizes).
"""
import os
import config

# -----------------
# Method settings
# -----------------
SEED = 42                       # seed for all unlearning randomness

SHRINK_FACTOR = 0.9             # w = w0 + 0.9 * (w - w0), i.e. eta = 0.1
SHRINK_ROUNDS = 6
SHRINKS_PER_ROUND = 5           # 6 x 5 = 30 shrinks in total

# Repair
REPAIR_LR = 5e-5                # Adam
BATCH_SIZE = 8
GRAD_CLIP = 1.0
FLIP_THRESHOLD = 0.005          # 0.5 %, checked separately at each level
FLIP_CHECK_BATCHES = 50         # block length for the flip-rate check
REPAIR_CAP_PASSES = 1           # at most one pass over the client's training split

# Aggregation
AGGREGATION_GAMMA = config.AGGREGATION_GAMMA

# Step 0 (initial weights)
TRAINING_SEED = 42              # the seed main.py sets
STEP0_CONTROL_SEED = 12345      # control initialization for the verification
STEP0_MIN_SCORE = 10.0          # r * sqrt(N) of the regenerated weights against round 1 must exceed this
STEP0_CONTROL_MAX_SCORE = 5.0   # |r * sqrt(N)| of the control must stay below this ("near 0")

# Statistics pass
MARGIN_SAMPLE_PER_BATCH = 4096  # random positions per batch per level, fixed seed
DISTANCE_CHUNK_POSITIONS = 32768  # positions per distance block (memory only; results do not depend on it)

# Decoder fit
DECODER_FIT_PATCH_FRACTION = 0.25
DECODER_FIT_RIDGE_SCALE = 1e-3  # lambda = scale * trace(Sigma_nn) / patch_length

# Fine-tuning (Step 5)
FINETUNE_EPOCHS = 2
FINETUNE_LR = 5e-6

# Trial mode (real model, real data, short run)
TRIAL_SLICES_PER_CLIENT = 64
TRIAL_SHRINK_ROUNDS = 1
TRIAL_SHRINKS_PER_ROUND = 2

# -----------------
# Run mode and paths
# -----------------
FIXTURE = False
TRIAL = False
FIXTURE_RESULTS_DIR = "FedDSR_results_fixture"

RESULTS_DIR = None              # set by configure(); the trained model's results folder
DEVICE = config.DEVICE

# Folder names inside a results folder
UNLEARNING_DIRNAME = "Unlearning"
STAGE2_DIRNAME = "DecoderFineTuning_Local"
# Training output that must never be written to
PROTECTED_DIRNAMES = (config.AGGREGATOR, STAGE2_DIRNAME)

STEP_DIRNAMES = {
    1: "step1_stats",
    2: "step2_server",
    3: "step3_rounds",
    4: "step4_fit",
    5: "step5_finetune",
    6: "release",
}


def add_common_args(parser):
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Results folder of the trained model (holds FedMedSR/ and DecoderFineTuning_Local/)")
    parser.add_argument("--fixture", action="store_true",
                        help="Use the synthetic fixture and the shortened schedule")
    return parser


def configure(results_dir=None, fixture=False, trial=False):
    """Set the run mode and the results folder. Every unlearning script calls this first."""
    global RESULTS_DIR, FIXTURE, TRIAL, DEVICE, SHRINK_ROUNDS, SHRINKS_PER_ROUND
    FIXTURE, TRIAL = fixture, trial
    if fixture:
        from unlearning.fixture import apply_fixture_mode  # must run before any model is built
        RESULTS_DIR = apply_fixture_mode(results_dir or FIXTURE_RESULTS_DIR)
    else:
        if results_dir is None:
            raise ValueError("--results-dir is required unless --fixture is given")
        RESULTS_DIR = results_dir
    if trial:
        SHRINK_ROUNDS, SHRINKS_PER_ROUND = TRIAL_SHRINK_ROUNDS, TRIAL_SHRINKS_PER_ROUND
    # The training code reads its results folder from config
    config.RESULTS_DIR = RESULTS_DIR
    DEVICE = config.DEVICE
    return RESULTS_DIR


# -----------------
# Path helpers
# -----------------
def training_dir():
    """Folder with best_model.pth, best_decoder_client_k.pth and round_r/ (read only)."""
    return os.path.join(RESULTS_DIR, config.AGGREGATOR)


def stage2_dir():
    """Folder with the Stage-2 refined decoders (read only)."""
    return os.path.join(RESULTS_DIR, STAGE2_DIRNAME)


def unlearning_root():
    return os.path.join(RESULTS_DIR, UNLEARNING_DIRNAME)


def phi0_dir():
    """Made once per trained model and shared by all withdrawals."""
    return os.path.join(unlearning_root(), "phi0")


def client_name(k):
    return config.CLIENT_DATASETS[k]["name"]


def withdraw_dir(withdrawn):
    """One folder per withdrawal; a trial run gets its own folder so it never blocks the full run."""
    suffix = "_trial" if TRIAL else ""
    return os.path.join(unlearning_root(), f"withdraw_{withdrawn}_{client_name(withdrawn)}{suffix}")


def step_dir(withdrawn, step):
    return os.path.join(withdraw_dir(withdrawn), STEP_DIRNAMES[step])


def logs_dir(withdrawn=None):
    base = withdraw_dir(withdrawn) if withdrawn is not None else unlearning_root()
    return os.path.join(base, "logs")
