# server.py
import torch, os, logging, matplotlib.pyplot as plt
from collections import OrderedDict
import config
from models import VQVAE2 
import torch.nn.functional as F
import numpy as np

class Server:
    def __init__(self, clients, results_logger):
        self.clients, self.results_logger = clients, results_logger
        self.logger = logging.getLogger()
        # Place the server/global model on CPU to save GPU memory
        self.global_model = VQVAE2().to('cpu')
        self.global_shared_params = self.global_model.get_shared_state_dict()

        # Track best model by validation PSNR (not loss)
        self.best_val_loss = float("inf")
        self.best_val_psnr = 0.0
        self.best_val_ssim = 0.0
        self.patience_counter = 0
        
        # Track metrics for plotting
        self.round_metrics = {
            'psnr': [],
            'ssim': [],
            'loss': [],
            'lpips': [],   
            'mse': [],
            'fid': []
        }

    def compute_client_representation(self, codebooks):
        top_mean = torch.mean(codebooks['top_codebook'], dim=0)
        bottom_mean = torch.mean(codebooks['bottom_codebook'], dim=0)
        c_k = (top_mean + bottom_mean) / 2.0
        return c_k  # Return raw vector, normalize inside similarity computation

    def compute_semantic_centrality(self, reps):
        num_clients = len(reps)
        stacked = torch.stack(reps)
        normed = F.normalize(stacked, p=2, dim=1)  # Normalize HERE for cosine similarity
        sim_matrix = torch.matmul(normed, normed.t())

        semantic_scores = []
        for i in range(num_clients):
            mask = torch.ones(num_clients, dtype=torch.bool)
            mask[i] = False  # Exclude self-similarity
            centrality = sim_matrix[i][mask].mean()
            semantic_scores.append(centrality.item())
        return torch.tensor(semantic_scores)

    def distribution_aware_aggregation(self, client_updates):
        """
        Distribution-Aware Aggregation (FedMedSR):
        w_final = γ * w_sim + (1-γ) * w_data
        """
        num_clients = len(client_updates)

        # Step 1: Compute client latent representations
        client_reps = []
        for client in self.clients:
            codebooks = client.get_codebook_vectors()
            rep = self.compute_client_representation(codebooks)
            client_reps.append(rep)

        # Step 2: Semantic centrality weights
        semantic_scores = self.compute_semantic_centrality(client_reps)
        semantic_weights = semantic_scores / torch.sum(semantic_scores)

        # Step 3: Dataset size weights
        dataset_sizes = torch.tensor([len(client.train_loader.dataset) for client in self.clients], dtype=torch.float32)
        data_weights = dataset_sizes / torch.sum(dataset_sizes)

        # Step 4: Combine with gamma
        gamma = config.AGGREGATION_GAMMA if hasattr(config, "AGGREGATION_GAMMA") else 0.5
        final_weights = gamma * semantic_weights + (1 - gamma) * data_weights

        self.logger.info("Distribution-Aware Aggregation Weights:")
        for i, w in enumerate(final_weights.tolist()):
            self.logger.info(f"Client {self.clients[i].client_id}: weight={w:.4f}")

        # Step 5: Aggregate into global model
        aggregated_params = OrderedDict()
        for key in self.global_shared_params.keys():
            aggregated_params[key] = torch.zeros_like(self.global_shared_params[key])
            for j in range(num_clients):
                aggregated_params[key] += final_weights[j].item() * client_updates[j][key]

        self.global_shared_params = aggregated_params
    """
    def aggregate_updates(self, client_updates, round_num):
        if round_num == 1:
            self.logger.info("Using FedAvg aggregation for round 1")
            num_clients = len(client_updates)
            aggregated_params = OrderedDict([(key, torch.zeros_like(self.global_shared_params[key])) for key in self.global_shared_params.keys()])
            for update in client_updates:
                for key in aggregated_params.keys():
                    aggregated_params[key] += update[key] / num_clients
            self.global_shared_params = aggregated_params
        else:
            self.logger.info("Using Distribution-Aware Aggregation (FedMedSR)")
            self.distribution_aware_aggregation(client_updates)
    """

    def aggregate_updates(self, client_updates, round_num):
        self.logger.info("Using Distribution-Aware Aggregation (FedMSR)")
        self.distribution_aware_aggregation(client_updates)
        

    def save_metrics_graphs(self, round_results_dir):
        """Save PSNR, SSIM, Loss, LPIPS, MSE, and FID graphs for current round"""
        if not self.round_metrics['psnr']:
            return

        rounds = list(range(1, len(self.round_metrics['psnr']) + 1))
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        metrics = ['psnr', 'ssim', 'loss', 'lpips', 'mse', 'fid']
        colors  = ['blue', 'green', 'red', 'magenta', 'cyan', 'orange']
        titles  = ['PSNR (dB)', 'SSIM', 'Loss', 'LPIPS', 'MSE', 'FID']

        for i, (m, c, t) in enumerate(zip(metrics, colors, titles)):
            axes[i].plot(rounds, self.round_metrics[m], '-o', color=c, linewidth=2, markersize=6)
            axes[i].set_xlabel("Round")
            axes[i].set_ylabel(t)
            axes[i].set_title(f"{t} vs Rounds")
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xticks(rounds)

        plt.tight_layout()
        plot_path = os.path.join(round_results_dir, "metrics_plot.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        self.logger.info(f"Metrics plot saved to: {plot_path}")


    def run(self):
        # Save the initial shared model once, so unlearning can anchor to it later
        round0_path = os.path.join(config.RESULTS_DIR, config.AGGREGATOR, "round_0_global_model.pth")
        if not os.path.exists(round0_path):
            os.makedirs(os.path.dirname(round0_path), exist_ok=True)
            torch.save(self.global_shared_params, round0_path)
            self.logger.info(f"Saved initial shared model to {round0_path}")

        for round_num in range(1, config.GLOBAL_ROUNDS + 1):
            self.logger.info(f"========== Starting Round {round_num}/{config.GLOBAL_ROUNDS} ==========")
            round_results_dir = os.path.join(config.RESULTS_DIR, config.AGGREGATOR, f"round_{round_num}")
            os.makedirs(round_results_dir, exist_ok=True)
            client_updates = []

            for client in self.clients:
                self.logger.info(f"--- Training Client {client.client_id} ---")
                client.set_shared_params(self.global_shared_params)
                client.train(round_num, self.global_shared_params)
                client_updates.append(client.get_shared_params())

            self.logger.info("--- Server Aggregating Client Updates ---")
            self.aggregate_updates(client_updates, round_num)

            # ---- Validate ----
            self.logger.info("--- Validating Models Post-Aggregation ---")
            total_val_psnr, total_val_ssim, total_val_loss = 0.0, 0.0, 0.0
            total_val_lpips, total_val_mse, total_val_fid = 0.0, 0.0, 0.0

            for client in self.clients:
                client.set_shared_params(self.global_shared_params)
                val_metrics = client.validate()
                self.logger.info(
                    f"Client {client.client_id} Validation: "
                    f"PSNR={val_metrics['psnr']:.4f}, "
                    f"SSIM={val_metrics['ssim']:.4f}, "
                    f"LPIPS={val_metrics['lpips']:.4f}, "
                    f"MSE={val_metrics['mse']:.4f}, "
                    f"FID={val_metrics['fid']:.4f}, "
                    f"Loss={val_metrics['loss']:.4f}"
                )
                total_val_psnr += val_metrics['psnr']
                total_val_ssim += val_metrics['ssim']
                total_val_lpips += val_metrics['lpips']
                total_val_mse   += val_metrics['mse']
                total_val_fid   += val_metrics['fid']
                total_val_loss  += val_metrics['loss']

            avg_val_psnr  = total_val_psnr / len(self.clients)
            avg_val_ssim  = total_val_ssim / len(self.clients)
            avg_val_lpips = total_val_lpips / len(self.clients)
            avg_val_mse   = total_val_mse / len(self.clients)
            avg_val_fid   = total_val_fid / len(self.clients)
            avg_val_loss  = total_val_loss / len(self.clients)

            self.logger.info(
                f"Global Average Validation: "
                f"PSNR={avg_val_psnr:.4f}, SSIM={avg_val_ssim:.4f}, "
                f"LPIPS={avg_val_lpips:.4f}, MSE={avg_val_mse:.6f}, "
                f"FID={avg_val_fid:.4f}, "
                f"Loss={avg_val_loss:.4f}"
            )

            # ---- Early stopping based on validation PSNR ----
            if avg_val_psnr > self.best_val_psnr:  # PSNR: higher is better
                self.best_val_psnr = avg_val_psnr
                self.best_val_ssim = avg_val_ssim
                self.best_val_loss = avg_val_loss
                self.patience_counter = 0
                self.logger.info(
                    f"New best validation PSNR: {self.best_val_psnr:.4f} dB "
                    f"(SSIM={self.best_val_ssim:.4f}). Saving model state."
                )
                torch.save(
                    self.global_shared_params,
                    os.path.join(config.RESULTS_DIR, config.AGGREGATOR, "best_model.pth")
                )
                for client in self.clients:
                    client_decoder_path = os.path.join(
                        config.RESULTS_DIR,
                        config.AGGREGATOR,
                        f"best_decoder_client_{client.client_id}.pth"
                    )
                    torch.save(client.model.state_dict(), client_decoder_path)
                    self.logger.info(f"Saved best decoder for client {client.client_id}")
            else:
                self.patience_counter += 1
                self.logger.info(
                    f"Validation PSNR did not improve. Best: {self.best_val_psnr:.4f}. "
                    f"Patience: {self.patience_counter}/{config.EARLY_STOPPING_PATIENCE}"
                )

            if self.patience_counter >= config.EARLY_STOPPING_PATIENCE:
                self.logger.warning(f"Early stopping triggered! Best PSNR: {self.best_val_psnr:.4f} dB (stopped at Round {round_num})")
                break


            # ---- Codebook Vitality Tracking ----
            round_vitality = {"top_active": [], "bottom_active": [],
                  "top_dead": [], "bottom_dead": []}

            for client in self.clients:
                stats = client.get_codebook_vitality()
                round_vitality["top_active"].append(stats["top"]["active"])
                round_vitality["bottom_active"].append(stats["bottom"]["active"])
                round_vitality["top_dead"].append(stats["top"]["dead"])
                round_vitality["bottom_dead"].append(stats["bottom"]["dead"])
                self.logger.info(
                    f"Client {client.client_id} Codebook Vitality | "
                    f"Top: active={stats['top']['active']}, dead={stats['top']['dead']}, entropy={stats['top']['entropy']:.3f} | "
                    f"Bottom: active={stats['bottom']['active']}, dead={stats['bottom']['dead']}, entropy={stats['bottom']['entropy']:.3f}"
                )
            
            if not hasattr(self, "vitality_log"):
                self.vitality_log = {
                    "top_active": [], "bottom_active": [],
                    "top_dead": [], "bottom_dead": []
                }

            self.vitality_log["top_active"].append(np.mean(round_vitality["top_active"]))
            self.vitality_log["bottom_active"].append(np.mean(round_vitality["bottom_active"]))
            self.vitality_log["top_dead"].append(np.mean(round_vitality["top_dead"]))
            self.vitality_log["bottom_dead"].append(np.mean(round_vitality["bottom_dead"]))


            plt.figure(figsize=(8,5))
            plt.plot(self.vitality_log["top_active"], label="Top Active")
            plt.plot(self.vitality_log["bottom_active"], label="Bottom Active")
            plt.plot(self.vitality_log["top_dead"], label="Top Dead", linestyle="--")
            plt.plot(self.vitality_log["bottom_dead"], label="Bottom Dead", linestyle="--")

            plt.xlabel("Federated Round")
            plt.ylabel("Number of Codes")
            plt.title("Codebook Vitality Across Federated Training")
            plt.legend()
            plt.grid(True)

            plt.savefig(os.path.join(round_results_dir, "codebook_vitality.png"), dpi=150)
            plt.close()


            # ---- Evaluate ----
            self.logger.info("--- Evaluating Models Post-Aggregation ---")
            total_global_psnr, total_global_ssim, total_global_loss = 0, 0, 0
            total_global_lpips, total_global_mse, total_global_fid = 0, 0, 0

            for client in self.clients:
                client.set_shared_params(self.global_shared_params)
                metrics = client.evaluate(round_num, round_results_dir)
                self.logger.info(
                    f"Client {client.client_id} Metrics: "
                    f"PSNR={metrics['psnr']:.4f}, "
                    f"SSIM={metrics['ssim']:.4f}, "
                    f"LPIPS={metrics['lpips']:.4f}, "
                    f"MSE={metrics['mse']:.6f}, "
                    f"FID={metrics['fid']:.4f}, "
                    f"Loss={metrics['loss']:.4f}"
                )
                total_global_psnr += metrics['psnr']
                total_global_ssim += metrics['ssim']
                total_global_lpips += metrics['lpips']
                total_global_mse += metrics['mse']
                total_global_fid += metrics['fid']
                total_global_loss += metrics['loss']

            avg_global_psnr = total_global_psnr / len(self.clients)
            avg_global_ssim = total_global_ssim / len(self.clients)
            avg_global_lpips = total_global_lpips / len(self.clients)
            avg_global_mse = total_global_mse / len(self.clients)
            avg_global_fid = total_global_fid / len(self.clients)
            avg_global_loss = total_global_loss / len(self.clients)

            # Store metrics for plotting
            self.round_metrics['psnr'].append(avg_global_psnr)
            self.round_metrics['ssim'].append(avg_global_ssim)
            self.round_metrics['lpips'].append(avg_global_lpips)
            self.round_metrics['mse'].append(avg_global_mse)
            self.round_metrics['fid'].append(avg_global_fid)
            self.round_metrics['loss'].append(avg_global_loss)

            self.results_logger.add_round_results(
                round_num, 'N/A',
                avg_global_psnr,
                avg_global_ssim,
                avg_global_loss,
                lpips_val=avg_global_lpips,
                mse_val=avg_global_mse,
                fid_val=avg_global_fid,
                is_global=True
            )
            self.logger.info(
                f"Global Average Metrics: "
                f"PSNR={avg_global_psnr:.4f}, SSIM={avg_global_ssim:.4f}, "
                f"LPIPS={avg_global_lpips:.4f}, MSE={avg_global_mse:.6f}, "
                f"FID={avg_global_fid:.4f}, "
                f"Loss={avg_global_loss:.4f}"
            )

            # Save metrics graphs + model
            self.save_metrics_graphs(round_results_dir)
            torch.save(self.global_shared_params, os.path.join(round_results_dir, 'global_model.pth'))
            self.results_logger.save()

        self.logger.info("Federated Learning process completed.")
