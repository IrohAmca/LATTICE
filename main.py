import json
import logging
import torch
import argparse

from data.data_loader import create_data_loaders
from models.transformer.base import Transformer
from train.trainer import TranslationTrainer
from utils.visualization.torch_viz import create_model_history_visualizations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_translation_model(src_vocab_size, tgt_vocab_size, config):
    """Create a transformer model for translation"""
    model = Transformer(
        src_vocab=src_vocab_size,
        tgt_vocab=tgt_vocab_size,
        e_N=config.get("encoder_layers", 6),
        d_N=config.get("decoder_layers", 6),
        d_model=config.get("d_model", 512),
        d_ff=config.get("d_ff", 2048),
        e_num_heads=config.get("encoder_heads", 8),
        d_num_heads=config.get("decoder_heads", 8),
        dropout=config.get("dropout", 0.1),
        max_len=config.get("max_len", 100),
    )

    # Store model parameters for saving
    model.model_params = {
        "src_vocab_size": src_vocab_size,
        "tgt_vocab_size": tgt_vocab_size,
        "encoder_layers": config.get("encoder_layers", 6),
        "decoder_layers": config.get("decoder_layers", 6),
        "d_model": config.get("d_model", 512),
        "d_ff": config.get("d_ff", 2048),
        "encoder_heads": config.get("encoder_heads", 8),
        "decoder_heads": config.get("decoder_heads", 8),
        "dropout": config.get("dropout", 0.1),
        "max_len": config.get("max_len", 100),
    }

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    model.apply(init_weights)
    return model


def prepare_translation_batch(batch, device):
    """Prepare batch for translation training"""
    src, tgt = batch

    # Create input and target sequences
    tgt_input = tgt[:, :-1]  # All tokens except last
    tgt_output = tgt[:, 1:]  # All tokens except first

    return {
        "input_ids": src.to(device),
        "decoder_input_ids": tgt_input.to(device),
        "labels": tgt_output.to(device),
    }


def get_config(
    model_size, default_path="config/train_maps", base_config="base_map.json"
):
    base_map = json.load(open(f"{default_path}/{base_config}"))
    model_map = json.load(open(f"{default_path}/{model_size}_map.json"))

    config = {**base_map, **model_map}
    config["model_size"] = model_size

    return config


def main(model_size, default_path="config/train_maps", base_config="base_map.json"):
    config = get_config(
        model_size=model_size, default_path=default_path, base_config=base_config
    )
    metrics_config = {"calculate": ["accuracy", "f1", "precision", "recall"]}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Loading data...")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = create_data_loaders(
        dataset_name=config["dataset_name"],
        batch_size=config["batch_size"],
        max_len=config["max_len"],
        min_freq=config["min_freq"],
        max_samples=config["max_samples"],
    )

    logger.info("Creating model...")
    model = create_translation_model(
        src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab), config=config
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    trainer = TranslationTrainer(
        model=model, config=config, metrics_config=metrics_config, device=device
    )

    logger.info("Starting training...")
    try:
        history = trainer.train(train_loader, val_loader)
        logger.info("Training completed successfully!")

        logger.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(test_loader)
        logger.info(f"Test metrics: {test_metrics}")

        return model, history, test_metrics, config

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train a Transformer model for translation."
    )   
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="Size of the model to train (default: small)",
    )
    args = parser.parse_args()

    model, history, test_metrics, config = main(model_size=args.model_size)

    logging.info("Creating visualizations...")
    create_model_history_visualizations(model, history, config, save_dir="reports")
