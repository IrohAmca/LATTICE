import json
import logging
import os
import platform
from datetime import datetime

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .metrics import compute_metrics


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

    def setup_scheduler(self, optimizer, train_loader):
        epochs = self.config.get("epochs", 10)
        warmup_ratio = self.config.get("warmup_ratio", 0.1)

        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(len(train_loader) * warmup_ratio),
            num_training_steps=len(train_loader) * epochs,
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

    def train_epoch(self, train_loader, optimizer, criterion, scaler):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc="Training")

        accumulation_steps = self.config.get("gradient_accumulation_steps", 1)

        for i, batch in enumerate(progress_bar):
            inputs = {
                "input_ids": batch["input_ids"].to(self.device),
                "attention_mask": batch["attention_mask"].to(self.device),
            }
            labels = batch["label"].to(self.device)

            if i % accumulation_steps == 0:
                optimizer.zero_grad()

            outputs = self.model(**inputs)

            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs

            loss = criterion(logits, labels) / accumulation_steps

            scaler.scale(loss).backward()

            optimizer.step()

            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.config.get("max_grad_norm")
                )
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * accumulation_steps

            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "accuracy": f"{100 * correct / total:.2f}%",
                }
            )

        return total_loss / len(train_loader), 100 * correct / total

    def evaluate(self, val_loader, criterion=None, label_encoder=None):
        self.model.eval()
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                inputs = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device),
                }
                labels = batch["label"].to(self.device)

                outputs = self.model(**inputs)

                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs

                loss = criterion(logits, labels)
                total_loss += loss.item()

                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)

        true_labels = label_encoder.inverse_transform(all_labels)
        pred_labels = label_encoder.inverse_transform(all_preds)
        metrics = compute_metrics(true_labels, pred_labels, self.metrics_config)
        metrics["loss"] = avg_loss
        return metrics

    def predict(self, data_loader, label_encoder=None):
        self.model.eval()
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                inputs = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device),
                }
                outputs = self.model(**inputs)

                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs

                probabilities = torch.softmax(logits, dim=1)
                _, predictions = torch.max(logits, 1)

                all_preds.extend(predictions.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())

        if label_encoder is not None:
            decoded_preds = label_encoder.inverse_transform(all_preds)
            return decoded_preds, all_probs

        return all_preds, all_probs

    def save_model(
        self, config, train_loss, train_acc, val_loss, val_acc, val_perplexity, epoch
    ):
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

        model_architecture = str(self.model)
        param_count = {}
        total_params = 0
        for name, param in self.model.named_parameters():
            param_count[name] = param.numel()
            total_params += param.numel()

        model_params = getattr(self.model, "model_params", {})

        training_params = {
            "model_architecture": {
                "structure": model_architecture,
                "parameter_count": param_count,
                "total_parameters": total_params,
                "model_params": model_params,
            },
            "training_config": config,
            "performance": {
                "best_val_loss": float(self.best_val_loss),
                "best_epoch": epoch + 1,
                "final_train_loss": float(train_loss),
                "final_train_acc": float(train_acc),
                "final_val_loss": float(val_loss),
                "final_val_acc": float(val_acc),
            },
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

        with open(f"{model_dir}/training_params.json", "w", encoding="utf-8") as f:
            json.dump(training_params, f, indent=4)

        summary = f"""
            ===== TRANSLATION MODEL SUMMARY =====
            Model name: {self.model_name}
            Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            
            Performance Metrics:
            - Best validation loss: {self.best_val_loss:.4f} (epoch {epoch + 1})
            - Final validation perplexity: {val_perplexity:.2f}
            - Token accuracy: {getattr(self, "last_token_accuracy", "N/A")}%
            - BLEU score: {getattr(self, "last_bleu_score", "N/A")}
            
            Model Architecture:
            - Parameters: {total_params:,}
            - Encoder layers: {config.get("encoder_layers", "N/A")}
            - Decoder layers: {config.get("decoder_layers", "N/A")}
            - Model dimension: {config.get("d_model", "N/A")}
            
            Translation Quality Assessment:
            - Perplexity <15: ✓ (Good baseline)
            - Token accuracy >60%: {"✓" if getattr(self, "last_token_accuracy", 0) > 60 else "✗"}
            - BLEU >15: {"✓" if getattr(self, "last_bleu_score", 0) > 15 else "✗"}
            """

        with open(f"{model_dir}/model_summary.txt", "w", encoding="utf-8") as f:
            f.write(summary)

        self.logger.info(f"Model and parameters saved to '{model_dir}' directory.")

    def train(self, train_loader, val_loader, label_encoder=None):
        start_time = datetime.now()

        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

        epochs = self.config.get("epochs", 10)
        lr = self.config.get("learning_rate", 1e-4)
        weight_decay = self.config.get("weight_decay", 0.01)
        warmup_ratio = self.config.get("warmup_ratio", 0.1)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)

        config = {
            "optimizer": "AdamW",
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "max_epochs": epochs,
            "batch_size": train_loader.batch_size
            if hasattr(train_loader, "batch_size")
            else None,
            "warmup_ratio": warmup_ratio,
            "max_grad_norm": max_grad_norm,
            "device": str(self.device),
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        optimizer = self.setup_optimizer()
        scheduler = self.setup_scheduler(optimizer, train_loader)

        class_weights_np = self.compute_class_weights(train_loader)
        config["class_weights"] = class_weights_np.tolist()

        use_weighted_loss = self.config.get("use_weighted_loss", False)
        if use_weighted_loss:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            self.logger.info("Using weighted cross-entropy loss")
        else:
            criterion = nn.CrossEntropyLoss()

        early_stopping = self.config.get("early_stopping", {})
        early_stopping_enabled = early_stopping.get("enabled", False)
        patience = early_stopping.get("patience", 3)
        min_delta = early_stopping.get("min_delta", 0.01)
        patience_counter = 0
        best_val_loss = float("inf")

        use_mixed_precision = self.config.get("mixed_precision", False)
        scaler = torch.amp.GradScaler(self.device.type)

        save_strategy = self.config.get("save_strategy", "epoch")
        save_steps = self.config.get("save_steps", 100)
        save_counter = 0
        save_total_limit = self.config.get("save_total_limit", 3)
        saved_checkpoints = []

        epoch_progress = tqdm(range(epochs), desc="Epochs")

        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(
            f"Early stopping {'enabled' if early_stopping_enabled else 'disabled'}"
        )
        self.logger.info(
            f"Mixed precision {'enabled' if use_mixed_precision else 'disabled'}"
        )

        for epoch in epoch_progress:
            if use_mixed_precision:
                train_loss, train_acc = self._train_epoch_mixed_precision(
                    train_loader, optimizer, criterion, scaler
                )
            else:
                train_loss, train_acc = self.train_epoch(
                    train_loader, optimizer, criterion, scaler
                )

            scheduler.step()

            metrics = self.evaluate(
                val_loader, criterion=criterion, label_encoder=label_encoder
            )

            f1 = metrics["f1"]
            precision = metrics["precision"]
            val_loss = metrics["loss"]
            val_acc = metrics["accuracy"]

            val_perplexity = torch.exp(torch.tensor(val_loss)).item()

            self.training_history["epochs"].append(epoch + 1)
            self.training_history["train_loss"].append(train_loss)
            self.training_history["train_acc"].append(train_acc)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["val_acc"].append(val_acc)
            self.training_history["f1"].append(f1)
            self.training_history["precision"].append(precision)

            epoch_progress.set_postfix(
                {
                    "train_loss": f"{train_loss:.4f}",
                    "train_acc": f"{train_acc:.2f}%",
                    "val_loss": f"{val_loss:.4f}",
                    "val_acc": f"{val_acc:.2f}%",
                    "f1": f"{f1:.2f}%",
                    "precision": f"{precision:.2f}%",
                }
            )

            self.logger.info(f"Epoch {epoch + 1}/{epochs}:")
            self.logger.info(
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%"
            )
            self.logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            if save_strategy == "epoch" or (
                save_strategy == "steps" and save_counter >= save_steps
            ):
                save_counter = 0
                self.save_model(
                    config,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                    val_perplexity,
                    epoch,
                )

                checkpoint_path = os.path.join(
                    self.save_dir, self.model_name, f"checkpoint-{epoch}"
                )
                saved_checkpoints.append(checkpoint_path)

                if save_total_limit > 0 and len(saved_checkpoints) > save_total_limit:
                    oldest_checkpoint = saved_checkpoints.pop(0)
                    if os.path.exists(oldest_checkpoint):
                        import shutil

                        shutil.rmtree(oldest_checkpoint)
                        self.logger.info(f"Removed old checkpoint: {oldest_checkpoint}")

            save_counter += 1

            if early_stopping_enabled:
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.logger.info(f"Validation loss improved to {val_loss:.4f}")

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_model(
                            config,
                            train_loss,
                            train_acc,
                            val_loss,
                            val_acc,
                            val_perplexity,
                            epoch,
                        )
                else:
                    patience_counter += 1
                    self.logger.info(
                        f"Validation loss did not improve. Patience: {patience_counter}/{patience}"
                    )

                    if patience_counter >= patience:
                        self.logger.info(
                            f"EARLY STOPPING TRIGGERED AFTER {epoch + 1} EPOCHS!!!"
                        )
                        break

        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        config["end_time"] = end_time.strftime("%Y-%m-%d %H:%M:%S")
        config["training_duration_seconds"] = training_duration

        self.logger.info("\nTraining completed!")
        self.logger.info(f"Total time: {training_duration / 60:.2f} minutes")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

        return self.training_history

    def _train_epoch_mixed_precision(self, train_loader, optimizer, criterion, scaler):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        accumulation_steps = self.config.get("gradient_accumulation_steps", 1)

        progress_bar = tqdm(train_loader, desc="Training (mixed precision)")

        for i, batch in enumerate(progress_bar):
            inputs = {
                "input_ids": batch["input_ids"].to(self.device),
                "attention_mask": batch["attention_mask"].to(self.device),
            }
            labels = batch["label"].to(self.device)

            if i % accumulation_steps == 0:
                optimizer.zero_grad()

            with torch.amp.autocast(device_type=self.device.type):
                outputs = self.model(**inputs)

                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs

                loss = criterion(logits, labels) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.config.get("max_grad_norm")
                )
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * accumulation_steps

            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item() * accumulation_steps:.4f}",
                    "accuracy": f"{100 * correct / total:.2f}%",
                }
            )

        return total_loss / len(train_loader), 100 * correct / total


class TranslationTrainer(Trainer):
    def __init__(self, model, config, metrics_config, device="cuda"):
        super().__init__(model, config, metrics_config, device)
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
            return super().setup_scheduler(optimizer, train_loader)

    def curriculum_learning(self, epoch):
        max_len = self.config.get("max_len", 80)

        if max_len > 16:
            return 16  * (epoch % 10)
        else:
            return max_len
    def compute_class_weights(self, train_loader):
        return np.array([1.0])

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
                bleu_score = self.compute_simple_bleu(predictions, references)
                self.logger.info(f"BLEU Score calculated on {len(predictions)} samples")
                return bleu_score
            except Exception as e:
                self.logger.warning(f"BLEU calculation failed: {e}")
                return 0.0
        else:
            self.logger.warning("No valid predictions/references for BLEU calculation")
            return 0.0

    def compute_simple_bleu(self, predictions, references):
        import math
        from collections import Counter

        def get_ngrams(tokens, n):
            return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

        def bleu_score(pred_tokens, ref_tokens, max_n=4):
            scores = []
            for n in range(1, max_n + 1):
                pred_ngrams = Counter(get_ngrams(pred_tokens, n))
                ref_ngrams = Counter(get_ngrams(ref_tokens, n))

                if len(pred_ngrams) == 0:
                    scores.append(0.0)
                    continue

                matches = sum((pred_ngrams & ref_ngrams).values())
                total = sum(pred_ngrams.values())
                scores.append(matches / total if total > 0 else 0.0)

            if all(score > 0 for score in scores):
                bleu = math.exp(sum(math.log(score) for score in scores) / len(scores))
            else:
                bleu = 0.0

            pred_len = len(pred_tokens)
            ref_len = len(ref_tokens)
            if pred_len > ref_len:
                bp = 1.0
            else:
                bp = math.exp(1 - ref_len / pred_len) if pred_len > 0 else 0.0

            return bleu * bp

        total_bleu = 0.0
        valid_pairs = 0

        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()

            if len(pred_tokens) > 0 and len(ref_tokens) > 0:
                bleu = bleu_score(pred_tokens, ref_tokens)
                total_bleu += bleu
                valid_pairs += 1

        return (total_bleu / valid_pairs * 100) if valid_pairs > 0 else 0.0

    def translate_batch(self, src, src_vocab=None, tgt_vocab=None, max_len=50):
        src = src.to(self.device)
        batch_size = src.size(0)
        device = src.device

        sos_token = 1
        tgt = torch.full((batch_size, 1), sos_token, device=device, dtype=torch.long)

        for step in range(max_len):
            try:
                outputs = self.model(src, tgt)
                next_token = outputs[:, -1:].argmax(dim=-1)

                tgt = torch.cat([tgt, next_token], dim=1)

                eos_token = 2
                if (next_token == eos_token).all():
                    break

            except Exception as e:
                self.logger.warning(f"Translation error at step {step}: {e}")
                break

        return tgt

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

        blue_score = self.calculate_bleu_score(val_loader)

        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "token_accuracy": token_accuracy,
            "bleu_score": blue_score,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    def train(self, train_loader, val_loader, label_encoder=None):
        start_time = datetime.now()

        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

        epochs = self.config.get("epochs", 10)
        lr = self.config.get("learning_rate", 1e-4)
        weight_decay = self.config.get("weight_decay", 0.01)
        warmup_ratio = self.config.get("warmup_ratio", 0.1)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)

        config = {
            "optimizer": "AdamW",
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "max_epochs": epochs,
            "batch_size": getattr(
                train_loader, "batch_size", self.config.get("batch_size")
            ),
            "warmup_ratio": warmup_ratio,
            "max_grad_norm": max_grad_norm,
            "device": str(self.device),
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        optimizer = self.setup_optimizer()
        scheduler = self.setup_scheduler(optimizer, train_loader)

        criterion = self.criterion

        early_stopping = self.config.get("early_stopping", {})
        early_stopping_enabled = early_stopping.get("enabled", False)
        patience = early_stopping.get("patience", 3)
        min_delta = early_stopping.get("min_delta", 0.01)
        patience_counter = 0
        best_val_loss = float("inf")

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
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.logger.info(f"Validation loss improved to {val_loss:.4f}")

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.last_token_accuracy = token_accuracy
                        self.last_bleu_score = bleu_score
                        self.save_model(
                            config,
                            train_loss,
                            train_perplexity,
                            val_loss,
                            val_perplexity,
                            epoch,
                            val_perplexity,
                        )
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
