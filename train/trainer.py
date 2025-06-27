import json
import logging
import os
import platform
from datetime import datetime

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


class Trainer:
    def __init__(self, model, config, metrics_config, device="cuda"):
        self.model = model.to(device)
        self.config = config
        self.metrics_config = metrics_config
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.tokenizer = None
        self.class_weights = None
        self.training_history = {
            "epochs": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "f1": [],
            "precision": [],
            "recall": [],
        }
        self.best_val_loss = float("inf")
        self.model_name = config.get("model_name", "model")
        self.save_dir = config.get("save_dir", "models/saved")

        os.makedirs(os.path.join(self.save_dir, self.model_name), exist_ok=True)
        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=0, label_smoothing=config.get("label_smoothing", 0.1)
        )

    def setup_scheduler(self, optimizer, train_loader):
        if self.config.get("use_cosine_schedule", False):
            from torch.optim.lr_scheduler import CosineAnnealingLR

            return CosineAnnealingLR(
                optimizer,
                T_max=self.config.get("epochs", 30),
                eta_min=self.config.get("min_learning_rate", 1e-6),
            )
        else:
            epochs = self.config.get("epochs", 10)
            warmup_ratio = self.config.get("warmup_ratio", 0.1)

            return get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(len(train_loader) * warmup_ratio),
                num_training_steps=len(train_loader) * epochs,
            )

    def setup_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        return self

    def setup_optimizer(self):
        lr = self.config.get("learning_rate", 1e-4)
        weight_decay = self.config.get("weight_decay", 0.01)

        return AdamW(
            self.model.parameters(),
            lr=float(lr),
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def compute_class_weights(self, train_loader):
        all_labels = []
        for batch in train_loader:
            all_labels.append(batch["label"].cpu().numpy())
        all_labels = np.concatenate(all_labels)

        class_weights_np = compute_class_weight(
            class_weight="balanced", classes=np.unique(all_labels), y=all_labels
        )
        self.class_weights = torch.tensor(
            class_weights_np, dtype=torch.float32, device=self.device
        )
        return class_weights_np

    def save_model(self, epoch=None, metrics=None, **kwargs):
        if isinstance(epoch, dict):
            config = epoch
            val_loss = kwargs.get("val_loss", 0.0)
            val_perplexity = kwargs.get("val_perplexity", 0.0)
            epoch = kwargs.get("epoch", 0)

            metrics = {
                "loss": val_loss,
                "perplexity": val_perplexity,
                "token_accuracy": getattr(self, "last_token_accuracy", 0.0),
                "bleu_score": getattr(self, "last_bleu_score", 0.0),
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }
        else:
            config = self.config
            if metrics is None:
                metrics = {}

        model_dir = os.path.join(self.save_dir, self.model_name)
        os.makedirs(model_dir, exist_ok=True)

        torch.save(self.model.state_dict(), f"{model_dir}/best_model.pth")
        torch.save(self.model, f"{model_dir}/full_model.pt")

        try:
            max_len = self.config.get("max_len", 128)
            example_input_ids = torch.ones((1, max_len), dtype=torch.long).to(
                self.device
            )
            example_attention_mask = torch.ones((1, max_len), dtype=torch.long).to(
                self.device
            )

            traced_model = torch.jit.trace(
                self.model, (example_input_ids, example_attention_mask)
            )
            traced_model.save(f"{model_dir}/traced_model.pt")
            self.logger.info("TorchScript model successfully saved.")
        except Exception as e:
            self.logger.error(f"Failed to save TorchScript model: {e}")

        total_params = sum(p.numel() for p in self.model.parameters())
        param_count = {
            name: param.numel() for name, param in self.model.named_parameters()
        }
        model_params = getattr(self.model, "model_params", {})

        summary_data = {
            "model_name": self.model_name,
            "epoch": (epoch + 1) if epoch is not None else "N/A",
            "performance": {
                "best_val_loss": float(self.best_val_loss),
                **metrics,
            },
            "architecture": {
                "total_parameters": total_params,
                "parameter_count": param_count,
                "model_params": model_params,
                "structure": str(self.model),
            },
            "training_config": config,
            "history": self.training_history,
            "metadata": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_name": self.model_name,
                "tokenizer": self.tokenizer.name_or_path
                if hasattr(self.tokenizer, "name_or_path")
                else str(type(self.tokenizer)),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda
                if torch.cuda.is_available()
                else "N/A",
                "system_info": platform.platform(),
            },
        }

        def default_serializer(o):
            if isinstance(o, (np.generic, np.ndarray)):
                return o.tolist()
            if isinstance(o, torch.Tensor):
                return o.tolist()
            if isinstance(o, datetime):
                return o.isoformat()
            raise TypeError(
                f"Object of type {o.__class__.__name__} is not JSON serializable"
            )

        with open(f"{model_dir}/training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4, default=default_serializer)

        legacy_params = {
            "model_architecture": {
                "structure": str(self.model),
                "parameter_count": param_count,
                "total_parameters": total_params,
                "model_params": model_params,
            },
            "training_config": config,
            "performance": summary_data["performance"],
            "history": self.training_history,
            "metadata": summary_data["metadata"],
        }

        with open(f"{model_dir}/training_params.json", "w", encoding="utf-8") as f:
            json.dump(legacy_params, f, indent=4, default=default_serializer)

        token_accuracy = metrics.get(
            "token_accuracy", getattr(self, "last_token_accuracy", 0.0)
        )
        bleu_score = metrics.get("bleu_score", getattr(self, "last_bleu_score", 0.0))
        perplexity = metrics.get("perplexity", 0.0)

        summary_text = f"""===== TRANSLATION MODEL SUMMARY =====
                        Model name: {self.model_name}
                        Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

                        Performance Metrics:
                        - Best validation loss: {self.best_val_loss:.4f} (epoch {(epoch + 1) if epoch is not None else "N/A"})
                        - Final validation perplexity: {perplexity:.2f}
                        - Token accuracy: {token_accuracy:.1f}%
                        - BLEU score: {bleu_score:.2f}

                        Model Architecture:
                        - Parameters: {total_params:,}
                        - Encoder layers: {config.get("encoder_layers", "N/A")}
                        - Decoder layers: {config.get("decoder_layers", "N/A")}
                        - Model dimension: {config.get("d_model", "N/A")}

                        Translation Quality Assessment:
                        - Perplexity <15: {"✓" if perplexity < 15 else "✗"} (Good baseline)
                        - Token accuracy >60%: {"✓" if token_accuracy > 60 else "✗"}
                        - BLEU >15: {"✓" if bleu_score > 15 else "✗"}
                        """

        with open(f"{model_dir}/model_summary.txt", "w", encoding="utf-8") as f:
            f.write(summary_text)

        self.logger.info(f"Model and comprehensive summary saved to '{model_dir}'")

    def curriculum_learning(self, epoch):
        max_len = self.config.get("max_len", 80)

        if max_len > 16:
            return 16 * (epoch / 10)
        else:
            return max_len

    def train_epoch(self, train_loader, optimizer, criterion, scaler):
        self.model.train()
        total_loss = 0
        total_tokens = 0

        progress_bar = tqdm(train_loader, desc="Training Translation")
        accumulation_steps = self.config.get("gradient_accumulation_steps", 1)

        current_max_len = self.config.get(
            "current_max_len", self.config.get("max_len", 80)
        )

        for i, batch in enumerate(progress_bar):
            try:
                src, tgt = batch

                if self.config.get("use_curriculum_learning", False):
                    src, tgt = self.filter_batch_by_length((src, tgt), current_max_len)

                    if src.size(0) == 0:
                        continue

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:].contiguous()

                src = src.to(self.device)
                tgt_input = tgt_input.to(self.device)
                tgt_output = tgt_output.to(self.device)

                if i % accumulation_steps == 0:
                    optimizer.zero_grad()

                outputs = self.model(src, tgt_input)

                outputs = outputs.view(-1, outputs.size(-1))
                tgt_output = tgt_output.view(-1)

                loss = criterion(outputs, tgt_output) / accumulation_steps

                scaler.scale(loss).backward()

                if (i + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.get("max_grad_norm", 1.0),
                    )
                    scaler.step(optimizer)
                    scaler.update()

                total_loss += loss.item() * accumulation_steps
                total_tokens += (tgt_output != 0).sum().item()

                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item() * accumulation_steps:.4f}",
                        "perplexity": f"{torch.exp(torch.tensor(loss.item() * accumulation_steps)):.2f}",
                        "max_len": current_max_len,
                    }
                )

            except Exception as e:
                self.logger.error(f"Error in batch {i}: {e}")
                continue

        avg_loss = total_loss / len(train_loader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return avg_loss, perplexity

    def indices_to_text(self, indices, vocab):
        if hasattr(vocab, "itos"):
            return " ".join(
                [vocab.itos[idx] for idx in indices if idx not in [0, 1, 2, 3]]
            )
        elif isinstance(vocab, dict):
            idx_to_token = {idx: token for token, idx in vocab.items()}
            return " ".join(
                [
                    idx_to_token.get(idx, "<UNK>")
                    for idx in indices
                    if idx not in [0, 1, 2, 3]
                ]
            )
        else:
            return " ".join([str(idx) for idx in indices if idx not in [0, 1, 2, 3]])

    def calculate_bleu_score(self, val_loader, src_vocab=None, tgt_vocab=None):
        import sacrebleu

        self.model.eval()
        predictions = []
        references = []

        max_samples = 100
        sample_count = 0

        with torch.no_grad():
            for batch in val_loader:
                if sample_count >= max_samples:
                    break

                src, tgt = batch

                try:
                    translated = self.translate_batch(src, src_vocab, tgt_vocab)

                    for i in range(min(len(translated), len(tgt))):
                        pred_text = self.indices_to_text(
                            translated[i].cpu().numpy(), tgt_vocab
                        )
                        ref_text = self.indices_to_text(tgt[i].cpu().numpy(), tgt_vocab)

                        if pred_text.strip() and ref_text.strip():
                            predictions.append(pred_text)
                            references.append(ref_text)

                        sample_count += 1
                        if sample_count >= max_samples:
                            break

                except Exception as e:
                    self.logger.warning(f"BLEU calculation error: {e}")
                    continue

        if len(predictions) > 0 and len(references) > 0:
            try:
                references_sacre = [[r] for r in references]
                bleu = sacrebleu.corpus_bleu(
                    predictions, references_sacre, lowercase=True
                )
                self.logger.info(
                    f"SacreBLEU Score calculated on {len(predictions)} samples: {bleu.score}"
                )
                return bleu.score
            except Exception as e:
                self.logger.warning(f"BLEU calculation failed: {e}")
                return 0.0
        else:
            self.logger.warning("No valid predictions/references for BLEU calculation")
            return 0.0

    def translate_batch(self, src, src_vocab=None, tgt_vocab=None, max_len=50):
        if self.config.get("use_beam_search", False) and src.size(0) == 1:
            return self.beam_search_decode(src, max_len=max_len)

        src = src.to(self.device)
        batch_size = src.size(0)
        device = src.device

        src_len = src.size(1)
        safe_max_len = min(max_len, src_len + 20)

        sos_token = 1
        tgt = torch.full((batch_size, 1), sos_token, device=device, dtype=torch.long)

        for step in range(safe_max_len):
            try:
                if tgt.size(1) > src_len + 15:
                    self.logger.warning(f"Target length ({tgt.size(1)}) exceeding safe limit relative to source ({src_len})")
                    break

                outputs = self.model(src, tgt)
                
                if outputs.size(1) != tgt.size(1):
                    self.logger.warning(f"Output length mismatch: got {outputs.size(1)}, expected {tgt.size(1)}")
                    break
                
                next_token = outputs[:, -1:].argmax(dim=-1)
                tgt = torch.cat([tgt, next_token], dim=1)

                eos_token = 2
                if (next_token == eos_token).all():
                    break

            except RuntimeError as e:
                if "size" in str(e) and "must match" in str(e):
                    self.logger.warning(f"Tensor size mismatch at step {step}: src_len={src.size(1)}, tgt_len={tgt.size(1)}")
                    break
                else:
                    self.logger.warning(f"Translation error at step {step}: {e}")
                    break
            except Exception as e:
                self.logger.warning(f"Translation error at step {step}: {e}")
                break

        return tgt

    def beam_search_decode(self, src, max_len=50):
        self.model.eval()
        beam_width = self.config.get("beam_width", 5)
        sos_token = 1
        eos_token = 2

        with torch.no_grad():
            beams = [
                (torch.tensor([sos_token], device=self.device, dtype=torch.long), 0.0)
            ]
            completed_beams = []

            for _ in range(max_len):
                new_beams = []
                for seq, score in beams:
                    if seq[-1].item() == eos_token:
                        completed_beams.append((seq, score))
                        continue

                    outputs = self.model(src, seq.unsqueeze(0))
                    next_token_logits = outputs[:, -1, :]
                    log_probs = torch.nn.functional.log_softmax(
                        next_token_logits, dim=-1
                    )

                    top_log_probs, top_indices = log_probs.topk(beam_width)

                    for i in range(beam_width):
                        next_tok = top_indices[0, i]
                        log_p = top_log_probs[0, i].item()
                        new_seq = torch.cat([seq, next_tok.unsqueeze(0)])
                        new_beams.append((new_seq, score + log_p))
                all_candidates = new_beams + completed_beams
                beams = sorted(
                    all_candidates, key=lambda x: x[1] / len(x[0]), reverse=True
                )[:beam_width]

                if all(b[0][-1].item() == eos_token for b in beams):
                    break

            best_seq, _ = sorted(beams, key=lambda x: x[1] / len(x[0]), reverse=True)[0]
            return best_seq.unsqueeze(0)

    def calculate_token_accuracy(self, val_loader):
        self.model.eval()
        correct_tokens = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in val_loader:
                src, tgt = batch
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:].contiguous()

                src = src.to(self.device)
                tgt_input = tgt_input.to(self.device)
                tgt_output = tgt_output.to(self.device)

                outputs = self.model(src, tgt_input)

                predicted = torch.argmax(outputs, dim=-1)

                mask = tgt_output != 0
                correct_tokens += ((predicted == tgt_output) & mask).sum().item()
                total_tokens += mask.sum().item()

        return (correct_tokens / total_tokens) * 100 if total_tokens > 0 else 0.0

    def evaluate(self, val_loader, criterion=None, label_encoder=None):
        self.model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                src, tgt = batch
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:].contiguous()

                src = src.to(self.device)
                tgt_input = tgt_input.to(self.device)
                tgt_output = tgt_output.to(self.device)

                outputs = self.model(src, tgt_input)
                outputs = outputs.view(-1, outputs.size(-1))
                tgt_output = tgt_output.view(-1)

                loss = self.criterion(outputs, tgt_output)
                total_loss += loss.item()
                total_tokens += (tgt_output != 0).sum().item()

        avg_loss = total_loss / len(val_loader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        token_accuracy = self.calculate_token_accuracy(val_loader)

        bleu_score = self.calculate_bleu_score(val_loader)

        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "token_accuracy": token_accuracy,
            "bleu_score": bleu_score,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    def train(self, train_loader, val_loader, label_encoder=None):
        start_time = datetime.now()

        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

        epochs = self.config.get("epochs", 10)

        optimizer = self.setup_optimizer()
        scheduler = self.setup_scheduler(optimizer, train_loader)

        criterion = self.criterion

        early_stopping = self.config.get("early_stopping", {})
        early_stopping_enabled = early_stopping.get("enabled", False)
        patience = early_stopping.get("patience", 3)
        min_delta = early_stopping.get("min_delta", 0.01)
        patience_counter = 0

        use_mixed_precision = self.config.get("mixed_precision", False)
        scaler = torch.amp.GradScaler(
            self.device.type if self.device.type == "cuda" else "cpu"
        )

        epoch_progress = tqdm(range(epochs), desc="Epochs")

        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(
            f"Early stopping {'enabled' if early_stopping_enabled else 'disabled'}"
        )
        self.logger.info(
            f"Mixed precision {'enabled' if use_mixed_precision else 'disabled'}"
        )

        for epoch in epoch_progress:
            if self.config.get("use_curriculum_learning", False):
                current_max_len = self.curriculum_learning(epoch)
                self.logger.info(
                    f"Epoch {epoch + 1}: Using max_len = {current_max_len}"
                )
                self.config["current_max_len"] = current_max_len

            if use_mixed_precision:
                train_loss, train_perplexity = self._train_epoch_mixed_precision(
                    train_loader, optimizer, criterion, scaler
                )
            else:
                train_loss, train_perplexity = self.train_epoch(
                    train_loader, optimizer, criterion, scaler
                )

            scheduler.step()

            metrics = self.evaluate(val_loader, criterion=criterion)

            val_loss = metrics["loss"]
            val_perplexity = metrics["perplexity"]
            token_accuracy = metrics.get("token_accuracy", 0.0)
            bleu_score = metrics.get("bleu_score", 0.0)

            self.training_history["epochs"].append(epoch + 1)
            self.training_history["train_loss"].append(train_loss)
            self.training_history["train_acc"].append(token_accuracy)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["val_acc"].append(val_perplexity)
            self.training_history["f1"].append(0.0)
            self.training_history["precision"].append(0.0)
            self.training_history["recall"].append(0.0)

            epoch_progress.set_postfix(
                {
                    "train_loss": f"{train_loss:.4f}",
                    "train_ppl": f"{train_perplexity:.2f}",
                    "val_loss": f"{val_loss:.4f}",
                    "val_ppl": f"{val_perplexity:.2f}",
                    "token_acc": f"{token_accuracy:.1f}%",
                    "bleu": f"{bleu_score:.2f}",
                }
            )

            self.logger.info(f"Epoch {epoch + 1}/{epochs}:")
            self.logger.info(
                f"Train Loss: {train_loss:.4f}, Train Perplexity: {train_perplexity:.2f}"
            )
            self.logger.info(
                f"Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.2f}"
            )
            self.logger.info(f"Token Accuracy: {token_accuracy:.1f}%")

            if early_stopping_enabled:
                if val_loss < self.best_val_loss - min_delta:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                    self.logger.info(f"Validation loss improved to {val_loss:.4f}")
                    self.last_token_accuracy = token_accuracy
                    self.last_bleu_score = bleu_score
                    self.save_model(epoch=epoch, metrics=metrics)
                else:
                    patience_counter += 1
                    self.logger.info(
                        f"Validation loss did not improve. Patience: {patience_counter}/{patience}"
                    )

                    if patience_counter >= patience:
                        self.logger.info(
                            f"EARLY STOPPING TRIGGERED AFTER {epoch + 1} EPOCHS"
                        )
                        break

        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()

        self.logger.info("\nTraining completed!")
        self.logger.info(f"Total time: {training_duration / 60:.2f} minutes")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

        return self.training_history

    def filter_batch_by_length(self, batch, max_len):
        src, tgt = batch

        src_lens = (src != 0).sum(dim=1)
        tgt_lens = (tgt != 0).sum(dim=1)

        mask = (src_lens <= max_len) & (tgt_lens <= max_len)

        if mask.sum() == 0:
            return src, tgt

        return src[mask], tgt[mask]

    def _train_epoch_mixed_precision(self, train_loader, optimizer, criterion, scaler):
        self.model.train()
        total_loss = 0
        total_tokens = 0

        accumulation_steps = self.config.get("gradient_accumulation_steps", 1)
        progress_bar = tqdm(train_loader, desc="Training (mixed precision)")

        for i, batch in enumerate(progress_bar):
            src, tgt = batch

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:].contiguous()

            src = src.to(self.device)
            tgt_input = tgt_input.to(self.device)
            tgt_output = tgt_output.to(self.device)

            if i % accumulation_steps == 0:
                optimizer.zero_grad()

            device_type = "cuda" if self.device.type == "cuda" else "cpu"

            with torch.amp.autocast(device_type=device_type):
                outputs = self.model(src, tgt_input)
                outputs = outputs.view(-1, outputs.size(-1))
                tgt_output_flat = tgt_output.view(-1)
                loss = criterion(outputs, tgt_output_flat) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.get("max_grad_norm", 1.0),
                )
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * accumulation_steps
            total_tokens += (tgt_output_flat != 0).sum().item()

            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item() * accumulation_steps:.4f}",
                    "perplexity": f"{torch.exp(torch.tensor(loss.item() * accumulation_steps)):.2f}",
                }
            )

        avg_loss = total_loss / len(train_loader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return avg_loss, perplexity
