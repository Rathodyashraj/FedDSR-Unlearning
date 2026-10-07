# main.py
import argparse, os, logging, torch, numpy as np
import config
from data import get_dataloaders
from client import Client
from server import Server
from utils import setup_logger, ResultsLogger

def main(args):
    torch.manual_seed(42); np.random.seed(42)
    config.GLOBAL_ROUNDS, config.NUM_CLIENTS, config.AGGREGATOR, config.EARLY_STOPPING_PATIENCE = args.rounds, args.clients, args.aggregator, args.patience
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu); config.DEVICE = torch.device(f"cuda:{args.gpu}")

    logger = setup_logger()
    logger.info(f"Using device: {config.DEVICE}")
    
    results_base_dir = os.path.join(config.RESULTS_DIR, config.AGGREGATOR)
    os.makedirs(results_base_dir, exist_ok=True)
    results_logger = ResultsLogger(results_base_dir)

    logger.info(f"Starting Federated VQ-VAE-2 with {config.AGGREGATOR} strategy.")
    logger.info(f"Running for {config.GLOBAL_ROUNDS} rounds with {config.NUM_CLIENTS} clients.")
    

    clients = []
    client_ids_to_run = list(config.CLIENT_DATASETS.keys())[:config.NUM_CLIENTS]
    # Detect available GPUs

    for i in client_ids_to_run:
        logger.info(f"Loading data for Client {i} ({config.CLIENT_DATASETS[i]['name']})...")
        train_loader, val_loader, test_loader = get_dataloaders(i)
        if train_loader is None or test_loader is None or val_loader is None:
            logger.error(f"Could not load data for client {i}. Exiting."); return
        # Assign all clients to the same device (config.DEVICE)
        device = config.DEVICE
        logger.info(f"Assigning Client {i} to device {device}")
        clients.append(Client(client_id=i, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, results_logger=results_logger, device=device))

    if not clients:
        logger.error("No clients were initialized.")
        return

    server = Server(clients, results_logger)
    server.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated VQ-VAE-2 for Super-Resolution")
    parser.add_argument("--aggregator", type=str, default="FedMedSR", choices=["fedprox", "fedavg", "FedMedSR"], help="Aggregation strategy")
    parser.add_argument("--patience", type=int, default=15, help="Patience for early stopping based on validation PSNR")
    parser.add_argument("--rounds", type=int, default=30, help="Number of global communication rounds")
    parser.add_argument("--clients", type=int, default=4, help="Number of clients to participate")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (e.g., 0)")
    main(parser.parse_args())