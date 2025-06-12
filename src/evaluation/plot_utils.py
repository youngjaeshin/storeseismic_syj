# src/evaluation/plot_utils.py
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import seaborn as sns # For confusion matrix

def plot_loss_curve(train_losses, val_losses, output_path, run_name=""):
    """Plots and saves the training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label=f'Training Loss (Final: {train_losses[-1]:.4f})' if train_losses else 'Training Loss')
    plt.plot(val_losses, label=f'Validation Loss (Final: {val_losses[-1]:.4f}, Min: {min(val_losses):.4f})' if val_losses else 'Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve for {run_name}')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Loss curve saved to {output_path}")
    plt.close()

def plot_attention_maps(attentions, output_dir, run_name="", num_layers_to_plot=4, num_heads_to_plot=4, epoch=None):
    """
    Plots and saves attention maps for specified layers and heads.
    attentions: A list of attention tensors, one for each layer.
                Each tensor shape: (batch_size, num_heads, seq_len_q, seq_len_k)
    """
    if not attentions:
        print("No attention weights provided for plotting.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Use the first sample in the batch for plotting
    # Determine actual number of layers and heads available
    actual_num_layers = len(attentions)
    if actual_num_layers == 0: return
    actual_num_heads = attentions[0].shape[1]

    layers_to_plot = min(num_layers_to_plot, actual_num_layers)
    heads_to_plot = min(num_heads_to_plot, actual_num_heads)
    
    fig, axs = plt.subplots(layers_to_plot, heads_to_plot, figsize=(heads_to_plot * 3, layers_to_plot * 3))
    if layers_to_plot == 1 and heads_to_plot == 1: # Handle single subplot case
        axs = np.array([[axs]])
    elif layers_to_plot == 1:
        axs = axs.reshape(1, -1)
    elif heads_to_plot == 1:
        axs = axs.reshape(-1, 1)


    for layer_idx in range(layers_to_plot):
        for head_idx in range(heads_to_plot):
            ax = axs[layer_idx, head_idx]
            # Detach, move to CPU, convert to numpy. Use first batch sample.
            attention_map = attentions[layer_idx][0, head_idx].cpu().detach().numpy()
            im = ax.imshow(attention_map, cmap='coolwarm', aspect='auto')
            ax.set_title(f'Layer {layer_idx+1}, Head {head_idx+1}')
            ax.set_xlabel("Key Sequence Position")
            ax.set_ylabel("Query Sequence Position")
            # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


    plt.tight_layout(rect=[0, 0, 0.95, 0.95]) # Adjust layout to make space for a global colorbar if needed
    # Optional: Add a single colorbar for the entire figure
    # cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    # fig.colorbar(im, cax=cbar_ax)

    epoch_str = f"_epoch{epoch}" if epoch is not None else ""
    plot_path = os.path.join(output_dir, f"attention_maps_{run_name}{epoch_str}.png")
    plt.suptitle(f"Attention Maps for {run_name}{epoch_str}", fontsize=16, y=0.99)
    plt.savefig(plot_path)
    print(f"Attention maps saved to {plot_path}")
    plt.close()


def plot_pretrain_reconstruction_results(inputs_list, reconstructed_list, labels_list, output_dir, run_name="", num_examples=4):
    """
    Plots input, reconstructed, label, and difference for pre-training results.
    Assumes inputs are (Batch, SeqLen, Features). For plotting, typically permute to (Features, SeqLen) for display.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_samples_to_plot = min(num_examples, len(inputs_list))

    for i in range(num_samples_to_plot):
        inp = inputs_list[i].cpu().permute(1,0).numpy() # Display as (Features, SeqLen)
        rec = reconstructed_list[i].cpu().permute(1,0).numpy()
        lab = labels_list[i].cpu().permute(1,0).numpy()
        diff = 10 * (lab - rec)

        perc = np.percentile(np.abs(lab), 99.5)
        if perc == 0: perc = 1.0 # Avoid vmin=vmax=0

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        titles = ["Input", "Reconstructed", "Label", "10x (Label - Recon.)"]
        data_to_plot = [inp, rec, lab, diff]
        vmins = [-perc, -perc, -perc, -1.0] # Diff often has smaller range
        vmaxs = [perc, perc, perc, 1.0]

        for ax, data, title, vmin, vmax in zip(axes, data_to_plot, titles, vmins, vmaxs):
            im = ax.imshow(data, aspect='auto', vmin=vmin, vmax=vmax, cmap='seismic', interpolation='none')
            ax.set_title(title)
            ax.set_xlabel("Sequence Length (Tokens)")
            ax.set_ylabel("Token Feature Dimension")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"pretrain_recon_{run_name}_example{i}.png")
        plt.savefig(plot_path)
        print(f"Pre-training reconstruction plot saved to {plot_path}")
        plt.close()

def plot_denoising_results(inputs_list, denoised_list, labels_list, output_dir, run_name="", num_examples=4, aspect_ratio=1.0):
    """Plots results for denoising task."""
    os.makedirs(output_dir, exist_ok=True)
    num_samples_to_plot = min(num_examples, len(inputs_list))

    for i in range(num_samples_to_plot):
        inp = inputs_list[i].cpu().permute(1,0).numpy() # Display as (Features, SeqLen)
        den = denoised_list[i].cpu().permute(1,0).numpy()
        lab = labels_list[i].cpu().permute(1,0).numpy()

        perc = np.percentile(np.abs(lab), 99.5)
        if perc == 0: perc = 1.0

        fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=False, sharey=False) # Match original 2x3
        
        # Row 1
        axes[0, 0].imshow(inp, aspect=aspect_ratio, vmin=-perc, vmax=perc, cmap='seismic', interpolation='none')
        axes[0, 0].set_title("Input")
        axes[0, 1].imshow(den, aspect=aspect_ratio, vmin=-perc, vmax=perc, cmap='seismic', interpolation='none')
        axes[0, 1].set_title("Denoised")
        axes[0, 2].imshow(inp - den, aspect=aspect_ratio, vmin=-perc, vmax=perc, cmap='seismic', interpolation='none')
        axes[0, 2].set_title("(Input - Denoised)")

        # Row 2
        axes[1, 0].imshow(lab, aspect=aspect_ratio, vmin=-perc, vmax=perc, cmap='seismic', interpolation='none')
        axes[1, 0].set_title("Label (Clean)")
        axes[1, 1].imshow(lab - den, aspect=aspect_ratio, vmin=-perc, vmax=perc, cmap='seismic', interpolation='none')
        axes[1, 1].set_title("(Label - Denoised)")
        axes[1, 2].imshow(inp - lab, aspect=aspect_ratio, vmin=-perc, vmax=perc, cmap='seismic', interpolation='none')
        axes[1, 2].set_title("(Input - Label) [Noise]")

        for ax_row in axes:
            for ax in ax_row:
                ax.set_xlabel("Sequence Length (e.g., Time Samples or Traces)")
                ax.set_ylabel("Feature Dimension (e.g., Receivers or Time Samples)")
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"denoising_{run_name}_example{i}.png")
        plt.suptitle(f"Denoising Result Example {i} for {run_name}", fontsize=16)
        fig.subplots_adjust(top=0.92)
        plt.savefig(plot_path)
        print(f"Denoising plot saved to {plot_path}")
        plt.close()


def plot_velocity_prediction_results(
    seismic_inputs_list, predicted_velocities_list, true_velocities_list,
    output_dir, run_name="", num_examples=4,
    depth_scale_factor=0.01, # To convert depth samples to km for y-axis
    seismic_aspect=0.5, vel_xlims=(1300, 4000), vel_ylims_km=(2.0, 0) # ylims in km, inverted
    ):
    """Plots input seismic, predicted velocity profile, and true velocity profile."""
    os.makedirs(output_dir, exist_ok=True)
    num_samples_to_plot = min(num_examples, len(seismic_inputs_list))

    for i in range(num_samples_to_plot):
        seismic = seismic_inputs_list[i].cpu().permute(1,0).numpy() # (Features, SeqLen)
        pred_vel = predicted_velocities_list[i].cpu().numpy() # (VelDepthSamples)
        true_vel = true_velocities_list[i].cpu().numpy()   # (VelDepthSamples)

        perc_seismic = np.percentile(np.abs(seismic), 99.5)
        if perc_seismic == 0: perc_seismic = 1.0

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot Seismic Input
        im_seis = axes[0].imshow(seismic, aspect=seismic_aspect, vmin=-perc_seismic, vmax=perc_seismic, cmap='seismic', interpolation='none')
        axes[0].set_title("Input Seismic Section")
        axes[0].set_xlabel("Sequence Length (e.g., Traces)")
        axes[0].set_ylabel("Feature Dimension (e.g., Time Samples)")
        fig.colorbar(im_seis, ax=axes[0], label="Amplitude", fraction=0.046, pad=0.04)

        # Plot Velocity Profiles
        depth_samples_pred = np.arange(len(pred_vel) + 1) * depth_scale_factor
        depth_samples_true = np.arange(len(true_vel) + 1) * depth_scale_factor

        axes[1].step(np.pad(true_vel, (0, 1), 'edge'), depth_samples_true, color='black', label='True Velocity')
        axes[1].step(np.pad(pred_vel, (0, 1), 'edge'), depth_samples_pred, color='red', label='Predicted Velocity', linestyle='--')
        
        axes[1].set_xlabel("Velocity (m/s)")
        axes[1].set_ylabel("Depth (km)")
        axes[1].set_title("Velocity Profile Prediction")
        if vel_xlims: axes[1].set_xlim(vel_xlims)
        if vel_ylims_km: axes[1].set_ylim(vel_ylims_km) # Inverted y-axis (depth increases downwards)
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.suptitle(f"Velocity Prediction Example {i} for {run_name}", fontsize=16)
        fig.subplots_adjust(top=0.90)
        plot_path = os.path.join(output_dir, f"velocity_pred_{run_name}_example{i}.png")
        plt.savefig(plot_path)
        print(f"Velocity prediction plot saved to {plot_path}")
        plt.close()


def plot_confusion_matrix_custom(true_labels, pred_labels, class_names, output_path, run_name=""):
    """Plots and saves a confusion matrix using Seaborn."""
    from sklearn.metrics import confusion_matrix # Local import as sklearn might not be always present
    
    cm = confusion_matrix(true_labels, pred_labels, labels=np.arange(len(class_names)))
    
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid") # Using seaborn's style
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14}) # Font size for annotations
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title(f'Confusion Matrix for {run_name}', fontsize=16)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Confusion matrix saved to {output_path}")
    plt.close()