# src/evaluation/evaluator_metrics.py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

def calculate_mse(tensor1, tensor2):
    """Calculates Mean Squared Error between two tensors."""
    if not isinstance(tensor1, torch.Tensor): tensor1 = torch.tensor(tensor1)
    if not isinstance(tensor2, torch.Tensor): tensor2 = torch.tensor(tensor2)
    return F.mse_loss(tensor1.float(), tensor2.float()).item()

def calculate_denoising_metrics(inputs_list, denoised_list, labels_list):
    """
    Calculates MSEs and reduction rates for denoising results.
    Assumes inputs are lists of individual sample tensors.
    """
    metrics_all_samples = []
    if not inputs_list: return metrics_all_samples

    for inp, den, lab in zip(inputs_list, denoised_list, labels_list):
        inp_np = inp.cpu().numpy()
        den_np = den.cpu().numpy()
        lab_np = lab.cpu().numpy()

        mse_input_denoised = calculate_mse(inp_np, den_np)
        mse_label_denoised = calculate_mse(lab_np, den_np)
        mse_input_label = calculate_mse(inp_np, lab_np) # Original noise power

        reduction_rate = float('nan')
        if mse_input_label > 1e-9: # Avoid division by zero or near-zero
            reduction_rate = (mse_input_label - mse_label_denoised) / mse_input_label * 100
        elif mse_label_denoised < 1e-9: # If both are near zero, perfect denoising
            reduction_rate = 100.0
        
        metrics_all_samples.append({
            "mse_input_vs_denoised": mse_input_denoised,
            "mse_label_vs_denoised": mse_label_denoised,
            "mse_input_vs_label (noise_power)": mse_input_label,
            "reduction_rate_percent": reduction_rate
        })
    
    # Calculate Averages
    avg_metrics = {}
    if metrics_all_samples:
        for key in metrics_all_samples[0].keys():
            valid_values = [m[key] for m in metrics_all_samples if not np.isnan(m[key])]
            if valid_values:
                avg_metrics[f"avg_{key}"] = np.mean(valid_values)
            else:
                avg_metrics[f"avg_{key}"] = float('nan')
                
    return metrics_all_samples, avg_metrics


def calculate_classification_metrics(true_labels_list, pred_labels_list, class_names=None, pred_probs_list=None):
    """
    Calculates accuracy, precision, recall, F1-score, and classification report.
    true_labels_list, pred_labels_list: lists or 1D numpy arrays of integer labels.
    class_names: list of string names for classes.
    pred_probs_list: optional, for metrics like ROC AUC if needed later.
    """
    if not isinstance(true_labels_list, np.ndarray): true_labels_list = np.array(true_labels_list)
    if not isinstance(pred_labels_list, np.ndarray): pred_labels_list = np.array(pred_labels_list)

    accuracy = accuracy_score(true_labels_list, pred_labels_list)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels_list, pred_labels_list, average='weighted', zero_division=0
    )
    
    report_str = classification_report(
        true_labels_list, pred_labels_list, target_names=class_names, zero_division=0
    )
    report_dict = classification_report(
        true_labels_list, pred_labels_list, target_names=class_names, output_dict=True, zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_score_weighted": f1,
        "classification_report_str": report_str,
        "classification_report_dict": report_dict
    }
    return metrics

def calculate_velocity_prediction_metrics(predicted_velocities_list, true_velocities_list):
    """
    Calculates MSE and relative MSE for velocity prediction.
    Assumes inputs are lists of individual 1D velocity profile tensors/arrays.
    """
    metrics_all_samples = []
    if not predicted_velocities_list: return metrics_all_samples, {}

    for pred_vel, true_vel in zip(predicted_velocities_list, true_velocities_list):
        pred_np = pred_vel.cpu().numpy() if isinstance(pred_vel, torch.Tensor) else np.array(pred_vel)
        true_np = true_vel.cpu().numpy() if isinstance(true_vel, torch.Tensor) else np.array(true_vel)

        mse = calculate_mse(pred_np, true_np)
        
        norm_true_label = np.sum(true_np**2) / true_np.size # (L2 norm squared) / num_elements
        relative_mse = float('nan')
        if norm_true_label > 1e-9:
            relative_mse = mse / norm_true_label
        
        metrics_all_samples.append({
            "mse": mse,
            "relative_mse": relative_mse
        })

    avg_metrics = {}
    if metrics_all_samples:
        for key in metrics_all_samples[0].keys():
            valid_values = [m[key] for m in metrics_all_samples if not np.isnan(m[key])]
            if valid_values:
                avg_metrics[f"avg_{key}"] = np.mean(valid_values)
            else:
                avg_metrics[f"avg_{key}"] = float('nan')
                
    return metrics_all_samples, avg_metrics