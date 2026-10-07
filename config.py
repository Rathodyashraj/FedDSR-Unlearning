# config.py
import torch
import numpy as np


# -----------------
# Project Config
# -----------------
PROJECT_NAME = "FedDSR_VQVAE2"
RESULTS_DIR = "FedDSR_results"

# -----------------
# Hardware Config
# -----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------
# Data Config
# -----------------
CLIENT_DATASETS = {
    0: {
        "name": "covid19_ct",
        "path": "../dataset/COVID-19 CT/curated_data/curated_data",
        "ext": "png"  
    },
    1: {
        "name": "pancreas",
        "path": "../dataset/Pancreas/Pancreas-CT/Pancreas-CT",
        "ext": "dcm"  
    },
    2: {
        "name": "kidney",
        "path": "../dataset/kidney/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone",
        "ext": "jpg"
    },
    3: {
        "name": "Brain_Stroke_CT",
        "path": "../dataset/Brain_Stroke_CT_Dataset/Brain_Stroke_CT_Dataset",
        "ext": ["png", "dcm"] 
    }
}
NUM_IMAGES_PER_CLIENT = None
DATA_SPLIT = {"train": 0.8, "val": 0.1, "test": 0.1}
DATA_USE_CLAHE = False          
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
AGGREGATION_GAMMA = 0.5

IMAGE_SIZE = 256
IMAGE_CHANNELS = 1  

NUM_CLIENTS = 4
GLOBAL_ROUNDS = 30
AGGREGATOR = "FedMedSR"
LOCAL_EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 5e-5  
FEDPROX_MU = 0.01
EARLY_STOPPING_PATIENCE = 15
NUM_WORKERS = 8


CODEBOOK_DEAD_THRESHOLD = 10
CODEBOOK_RESET_INTERVAL = 100
# ORTHO_WEIGHT = 0.01 


ENCODER_TOP_CONFIG = {
    "in_channels": IMAGE_CHANNELS,
    "hidden_channels": 384,     
    "num_residual_layers": 6,  # More layers for better features    
    "num_residual_hiddens": 96,  # Wider residual blocks
    "downsample_factor": 2
}

QUANTIZER_TOP_CONFIG = {
    "num_embeddings": 1024,        
    "embedding_dim": 128,         
    "commitment_cost": 0.25
}

ENCODER_BOTTOM_CONFIG = {
    "in_channels": IMAGE_CHANNELS,
    "hidden_channels": 384,  # Increased from 256    
    "num_residual_layers": 6,  # More layers for fine details
    "num_residual_hiddens": 96,
    "downsample_factor": 1
}

QUANTIZER_BOTTOM_CONFIG = {
    "num_embeddings": 1024,       
    "embedding_dim": 128,       
    "commitment_cost": 0.25
}

DECODER_CONFIG = {
    "top_embedding_dim": QUANTIZER_TOP_CONFIG["embedding_dim"],
    "bottom_embedding_dim": QUANTIZER_BOTTOM_CONFIG["embedding_dim"],
    "hidden_channels": 384,  # Increased from 256      
    "num_residual_layers": 6,  # Deeper decoder  
    "num_residual_hiddens": 96,
    "out_channels": IMAGE_CHANNELS
}


LOSS_CONFIG = {
    "use_perceptual_loss": True,
    "perceptual_loss_weight": 0.10,
    "structural_loss_weight": 0.20,
    "reconstruction_loss_weight": 2.0,
    "edge_loss_weight": 0.25,
    "vq_loss_weight": 0.06,
    "charbonnier_eps": 1e-3
}

