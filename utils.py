# utils.py
import logging, os, sys, torch, numpy as np, pandas as pd
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import config
from PIL import Image, ImageDraw, ImageFont
from torchvision.models import VGG19_Weights
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19


class UnbufferedStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

def setup_logger():
    handler = UnbufferedStreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger

def tensor_to_pil(tensor):
    """
    Converts a normalized tensor [-1,1] or [0,1] into a PIL image.
    Supports grayscale (1 channel) and RGB (3 channel).
    """
    tensor = tensor.cpu().detach()
    if tensor.ndim == 4:   # batch → take first element
        tensor = tensor.squeeze(0)

    # Normalize from [-1,1] → [0,1] if needed
    if tensor.min() < 0:
        tensor = (tensor * 0.5) + 0.5
    tensor = tensor.clamp(0, 1)

    if tensor.ndim == 2:   # H,W
        np_img = (tensor.numpy() * 255).astype(np.uint8)
        return Image.fromarray(np_img, mode="L")
    elif tensor.ndim == 3: # C,H,W
        if tensor.shape[0] == 1:   # grayscale with channel dim
            np_img = (tensor.squeeze(0).numpy() * 255).astype(np.uint8)
            return Image.fromarray(np_img, mode="L")
        elif tensor.shape[0] == 3: # RGB
            np_img = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            return Image.fromarray(np_img, mode="RGB")
    raise ValueError(f"Unsupported tensor shape for conversion: {tensor.shape}")



def calculate_metrics(hr_img, sr_img):
    """
    Compute PSNR and SSIM between two tensors using skimage.
    hr_img, sr_img: torch tensors in [-1,1] or [0,1], shape (C,H,W) or (H,W).
    Returns: (psnr_val, ssim_val)
    """
    hr_np = hr_img.cpu().detach()
    sr_np = sr_img.cpu().detach()

    # Normalize from [-1,1] to [0,1] if needed
    if hr_np.min() < 0:
        hr_np = (hr_np * 0.5) + 0.5
        sr_np = (sr_np * 0.5) + 0.5

    hr_np = hr_np.squeeze().numpy()
    sr_np = sr_np.squeeze().numpy()

    # Decide channel_axis
    if hr_np.ndim == 3 and hr_np.shape[0] in [1, 3]:  # C,H,W → H,W,C
        hr_np = np.transpose(hr_np, (1, 2, 0))
        sr_np = np.transpose(sr_np, (1, 2, 0))
        channel_axis = -1
    else:
        channel_axis = None

    psnr_val = psnr(hr_np, sr_np, data_range=1.0)
    ssim_val = ssim(hr_np, sr_np, data_range=1.0, channel_axis=channel_axis)

    return psnr_val, ssim_val


# -----------------------------
# Visual Results (Grid + Save)
# -----------------------------
def save_visual_results(lr_img, hr_img, sr_img, path, round_num, client_id, sample_idx):
    """Saves LR, HR, SR individually + 2x2 comparison grid with error map."""

    HEADER_HEIGHT = 40
    PADDING = 10
    FONT_SIZE = 20
    BG_COLOR = "black"
    TEXT_COLOR = "white"

    os.makedirs(path, exist_ok=True)

    # Convert to PIL
    lr_pil = tensor_to_pil(lr_img)
    hr_pil = tensor_to_pil(hr_img)
    sr_pil = tensor_to_pil(sr_img)

    # --- Save individual images ---
    lr_pil.save(os.path.join(path, f"round_{round_num}_client_{client_id}_sample_{sample_idx}_LR.png"))
    hr_pil.save(os.path.join(path, f"round_{round_num}_client_{client_id}_sample_{sample_idx}_HR.png"))
    sr_pil.save(os.path.join(path, f"round_{round_num}_client_{client_id}_sample_{sample_idx}_SR.png"))

    # --- Error map (grayscale abs diff) ---
    hr_tensor_norm = (hr_img.cpu() * 0.5) + 0.5
    sr_tensor_norm = (sr_img.cpu() * 0.5) + 0.5
    error_tensor = torch.abs(hr_tensor_norm - sr_tensor_norm)
    error_pil = tensor_to_pil(error_tensor)

    # Upscale LR for grid
    lr_pil_big = lr_pil.resize(hr_pil.size, Image.BICUBIC)

    # --- Make 2x2 grid ---
    images = [lr_pil_big, hr_pil, sr_pil, error_pil]
    labels = ["Input (LR)", "Ground Truth (HR)", "Model Output (SR)", "Error Map"]

    grid_width = hr_pil.width * 2
    grid_height = (hr_pil.height + HEADER_HEIGHT) * 2
    new_im = Image.new("RGB", (grid_width, grid_height), BG_COLOR)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", FONT_SIZE)
    except IOError:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(new_im)
    positions = [
        (0, HEADER_HEIGHT),
        (hr_pil.width, HEADER_HEIGHT),
        (0, hr_pil.height + HEADER_HEIGHT * 2),
        (hr_pil.width, hr_pil.height + HEADER_HEIGHT * 2),
    ]
    label_positions = [
        (PADDING, PADDING),
        (hr_pil.width + PADDING, PADDING),
        (PADDING, hr_pil.height + HEADER_HEIGHT + PADDING),
        (hr_pil.width + PADDING, hr_pil.height + HEADER_HEIGHT + PADDING),
    ]

    for i in range(4):
        new_im.paste(images[i], positions[i])
        draw.text(label_positions[i], labels[i], font=font, fill=TEXT_COLOR)

    # Save the comparison grid
    output_filename = os.path.join(path, f"round_{round_num}_client_{client_id}_sample_{sample_idx}.png")
    new_im.save(output_filename)

# -----------------------------
# Results Logger
# -----------------------------
class ResultsLogger:
    def __init__(self, results_dir):
        self.results_dir, self.metrics_data = results_dir, []

    def add_round_results(self, round_num, client_id, psnr_val, ssim_val, loss, lpips_val=None, mse_val=None, fid_val=None, is_global=False):
        self.metrics_data.append({
            "round": round_num,
            "type": "global" if is_global else f"client_{client_id}",
            "client_id": "N/A" if is_global else client_id,
            "psnr": psnr_val,
            "ssim": ssim_val,
            "lpips": lpips_val,
            "mse": mse_val,
            "fid": fid_val,
            "loss": loss,
        })

    def save(self):
        pd.DataFrame(self.metrics_data).to_csv(
            os.path.join(self.results_dir, "metrics.csv"), index=False
        )


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (robust L1)"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps ** 2))
        return loss


class EdgeAwareLoss(nn.Module):
    """Sobel edge-aware loss for better detail preservation."""
    def __init__(self):
        super(EdgeAwareLoss, self).__init__()
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def forward(self, pred, target):
        """Compute edge loss using Sobel gradients."""
        # Get gradients
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1)
        target_grad_x = F.conv2d(target, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1)
        
        # Compute edge magnitudes
        pred_edges = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-6)
        target_edges = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-6)
        
        # L1 loss on edges
        return F.l1_loss(pred_edges, target_edges)



class VGGLoss(nn.Module):
    """Perceptual VGG19 Loss"""
    def __init__(self, device):
        super(VGGLoss, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:35].eval().to(device)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.loss_fn = nn.L1Loss()

    def forward(self, x, y):
        # Rescale to [0,1]
        x_rescaled = (x + 1) / 2
        y_rescaled = (y + 1) / 2
        vgg_mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        vgg_std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)

        x_vgg = (x_rescaled - vgg_mean) / vgg_std
        y_vgg = (y_rescaled - vgg_mean) / vgg_std

        return self.loss_fn(self.vgg(x_vgg), self.vgg(y_vgg))