import torch
import torch.nn.functional as F
import numpy as np
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from .early_stopping import EarlyStopping

class Trainer:
    def __init__(self, model, optimizer, loss_fn, device, config):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.config = config # Full config dictionary

        # ### START: 코드 수정 (AttributeError 해결) ###
        self.task_type = config['task_type']
        self.finetune_task_name = config.get('finetune_params', {}).get('task_name', None)

        training_cfg = config['training']
        run_output_dir = os.path.join(config['base_results_dir'], config['run_name'], training_cfg['output_dir'])
        os.makedirs(run_output_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(run_output_dir, "best_checkpoint.pt")
        self.early_stopper = EarlyStopping(
            patience=training_cfg['patience_early_stopping'],
            verbose=True,
            path=checkpoint_path,
            delta=training_cfg.get('early_stopping_delta', 0.0)
        )
        
        self.vel_size_for_interp = config['model'].get('velocity_output_dim')
        if self.finetune_task_name == "velocity_prediction" and not self.vel_size_for_interp:
            print("Warning: 'velocity_output_dim' not found in model config for velocity prediction interpolation.")
        # ### END: 코드 수정 ###

    def _process_batch(self, batch, is_training):
        batch_on_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # ### START: 코드 수정 ###
        # Use clean labels as input if configured (for some fine-tuning tasks)
        if self.task_type == "finetune" and self.config['data'].get('use_clean_labels_as_input', False):
             inputs_embeds = batch_on_device['labels'].float()
        else:
            inputs_embeds = batch_on_device['inputs_embeds'].float()

        # Forward pass
        if self.task_type == "pretrain" or self.finetune_task_name in ["denoising"]:
            outputs = self.model(inputs_embeds=inputs_embeds)
            predictions = outputs.logits
            attentions = outputs.attentions if self.config['model'].get('output_attentions', False) else None
        
        elif self.task_type == "finetune":
            bert_output = self.model.bert(inputs_embeds=inputs_embeds, output_attentions=self.config['model'].get('output_attentions', False))
            sequence_output = bert_output[0]
            attentions = bert_output.attentions if self.config['model'].get('output_attentions', False) else None
            predictions = self.model.cls(sequence_output)
        else:
            raise ValueError(f"Unknown task_type for model forward pass: {self.task_type}")

        # Loss Calculation
        loss = None
        if self.task_type == "pretrain":
            mask_label = batch_on_device['mask_label'].float()
            target_labels = batch_on_device['labels'].float()
            if mask_label.ndim == 2 and predictions.ndim == 3:
                mask_label = mask_label.unsqueeze(-1).expand_as(predictions)
            loss = self.loss_fn(predictions * mask_label, target_labels * mask_label)

        elif self.finetune_task_name == "denoising":
            target_labels = batch_on_device['labels'].float()
            loss = self.loss_fn(predictions, target_labels)
        
        elif self.finetune_task_name == "velocity_prediction":
            raw_vel_labels = batch_on_device['velocity_label']
            if raw_vel_labels.shape[-1] != self.vel_size_for_interp:
                 target_labels = F.interpolate(raw_vel_labels.float().unsqueeze(1), size=self.vel_size_for_interp, mode='nearest').squeeze(1)
            else:
                target_labels = raw_vel_labels.float()
            loss = self.loss_fn(predictions, target_labels)

        elif self.finetune_task_name == "fault_detection":
            target_labels = batch_on_device['fault_label'].view(-1, 1).float()
            loss = self.loss_fn(predictions, target_labels)
        # (Add other fine-tuning task loss calculations here if needed)
        
        if loss is None:
            raise ValueError("Loss was not calculated. Check task configuration.")
        # ### END: 코드 수정 ###
            
        return loss, predictions, attentions


    def _train_epoch(self, train_dataloader):
        self.model.train()
        total_loss = 0
        loop_train = tqdm(train_dataloader, desc="Training Epoch", leave=False)
        for batch in loop_train:
            self.optimizer.zero_grad()
            loss, _, _ = self._process_batch(batch, is_training=True)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            loop_train.set_postfix(loss=loss.item())
        return total_loss / len(train_dataloader)

    def _validate_epoch(self, val_dataloader):
        self.model.eval()
        total_loss = 0
        last_batch_attentions = None
        loop_val = tqdm(val_dataloader, desc="Validation Epoch", leave=False)
        with torch.no_grad():
            for batch_idx, batch in enumerate(loop_val):
                loss, _, attentions = self._process_batch(batch, is_training=False)
                total_loss += loss.item()
                loop_val.set_postfix(loss=loss.item())
                if batch_idx == len(val_dataloader) - 1 and attentions:
                    last_batch_attentions = attentions
        return total_loss / len(val_dataloader), last_batch_attentions

    def train(self, train_dataloader, val_dataloader):
        print(f"--- Starting Training for: {self.config['run_name']} ---")
        training_cfg = self.config['training']
        print(f"Device: {self.device}, Epochs: {training_cfg['epochs']}, Patience: {training_cfg['patience_early_stopping']}")

        train_losses, val_losses, all_attentions_over_epochs = [], [], []
        start_time_total = time.time()

        for epoch in range(training_cfg['epochs']):
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch+1}/{training_cfg['epochs']}")
            avg_train_loss = self._train_epoch(train_dataloader)
            avg_val_loss, last_val_attentions = self._validate_epoch(val_dataloader)
            train_losses.append(avg_train_loss); val_losses.append(avg_val_loss)
            if last_val_attentions: all_attentions_over_epochs.append(last_val_attentions)
            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Duration: {epoch_duration:.2f}s")
            self.early_stopper(avg_val_loss, self.model)
            if self.early_stopper.early_stop:
                print("Early stopping triggered.")
                break
        
        print(f"--- Training Finished. Total time: {time.time() - start_time_total:.2f}s ---")
        if os.path.exists(self.early_stopper.path):
            print(f"Loading best model from checkpoint: {self.early_stopper.path}")
            self.model.load_state_dict(torch.load(self.early_stopper.path))
        
        if training_cfg.get("plot_loss_curve", True):
            self._plot_loss_curve(train_losses, val_losses)

        final_attentions = all_attentions_over_epochs[-1] if all_attentions_over_epochs else None
        return self.model, train_losses, val_losses, final_attentions

    def _plot_loss_curve(self, train_losses, val_losses):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label=f'Training Loss (Final: {train_losses[-1]:.4f})')
        plt.plot(val_losses, label=f'Validation Loss (Min: {min(val_losses):.4f})')
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.title(f'Loss Curve for {self.config["run_name"]}')
        plt.legend(); plt.grid(True)
        plot_path = os.path.join(os.path.dirname(self.early_stopper.path), "loss_curve.png")
        plt.savefig(plot_path)
        print(f"Loss curve saved to {plot_path}")
        plt.close()