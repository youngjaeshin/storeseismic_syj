#!/usr/bin/env python
import torch
import numpy as np
import os
import yaml
import argparse
from .datasets import SSDataset
import matplotlib.pyplot as plt

# Helper function for Normalization
def normalize_data(data_dict_list, train_data_key='inputs_embeds'):
    if not data_dict_list: return
    vmax_abs = torch.max(torch.abs(data_dict_list[0][train_data_key]))
    if vmax_abs == 0:
        print("Warning: Max absolute value for normalization is 0. Skipping normalization.")
        return
    for data_mlm in data_dict_list:
        if 'inputs_embeds' in data_mlm:
            data_mlm['inputs_embeds'] /= vmax_abs
        if 'labels' in data_mlm:
            data_mlm['labels'] /= vmax_abs
    print(f"Data normalized using max absolute value: {vmax_abs.item()}")

# Helper function for Data Augmentation: Multiplication
def augment_multiply(data_dict, factor):
    if factor <= 1: return
    for key in data_dict.keys():
        if isinstance(data_dict[key], torch.Tensor):
            data_dict[key] = data_dict[key].repeat(factor, *([1]*(data_dict[key].ndim -1)))
    if 'index' in data_dict:
         data_dict['index'] = torch.arange(data_dict['inputs_embeds'].shape[0])
    print(f"Data multiplied by factor {factor}. New count: {data_dict['inputs_embeds'].shape[0]}")

# Helper function for Data Augmentation: Time Shift
def augment_time_shift(data_dict, config_aug, token_type):
    if not config_aug or config_aug.get('time_shift_ops', 0) <= 0: return
    n_shift_ops = config_aug['time_shift_ops']
    min_shift = config_aug['min_time_shift']
    max_shift = config_aug['max_time_shift']
    filler = torch.mean(data_dict['inputs_embeds'])
    shift_dim = -1 if token_type == "trace" else 1
    original_len = data_dict['inputs_embeds'].shape[0]
    
    for _ in range(n_shift_ops):
        data_copy = {key: val[:original_len].clone() for key, val in data_dict.items() if isinstance(val, torch.Tensor)}
        for i in range(original_len):
            shift_mag = torch.randint(min_shift, max_shift + 1, (1,)).item()
            while shift_mag == 0:
                 shift_mag = torch.randint(min_shift, max_shift + 1, (1,)).item()
            for key_tensor in ['inputs_embeds', 'labels']:
                if key_tensor in data_copy:
                    tensor_to_shift = data_copy[key_tensor][i]
                    shifted_tensor = torch.roll(tensor_to_shift, shifts=shift_mag, dims=shift_dim)
                    if shift_dim == -1 or shift_dim == 2:
                        if shift_mag > 0: shifted_tensor[..., :shift_mag] = filler
                        elif shift_mag < 0: shifted_tensor[..., shift_mag:] = filler
                    elif shift_dim == 1:
                        if shift_mag > 0: shifted_tensor[:shift_mag, ...] = filler
                        elif shift_mag < 0: shifted_tensor[shift_mag:, ...] = filler
                    data_copy[key_tensor][i] = shifted_tensor
        for key in data_dict.keys():
            if isinstance(data_dict[key], torch.Tensor) and key in data_copy:
                data_dict[key] = torch.cat((data_dict[key], data_copy[key]), dim=0)
    
    if 'index' in data_dict: data_dict['index'] = torch.arange(data_dict['inputs_embeds'].shape[0])
    print(f"Time-shift augmentation applied ({n_shift_ops} ops). New count: {data_dict['inputs_embeds'].shape[0]}")

# Helper function for MLM Masking
def apply_mlm_masking(data_dict, config_mask, model_input_feature_dim):
    if not config_mask or config_mask.get('mask_proportion', 0) <= 0: return
    mask_proportion = config_mask['mask_proportion']
    mask_token_prob = config_mask['mask_token_prob']
    random_token_prob = config_mask['random_token_prob']
    mask_replacement_token_vec = torch.randn(1, 1, model_input_feature_dim)
    min_val, max_val = torch.min(mask_replacement_token_vec), torch.max(mask_replacement_token_vec)
    if max_val - min_val > 1e-6:
        mask_replacement_token_vec = -1 + (2 * (mask_replacement_token_vec - min_val) / (max_val - min_val))
    else:
        mask_replacement_token_vec.fill_(0.0)
    num_samples, seq_len = data_dict['inputs_embeds'].shape[0], data_dict['inputs_embeds'].shape[1]
    
    for i in range(num_samples):
        num_to_mask = int(np.floor(seq_len * mask_proportion))
        if num_to_mask == 0: continue
        masked_indices = torch.randperm(seq_len)[:num_to_mask]
        data_dict['mask_label'][i, masked_indices] = 1
        for j_idx in masked_indices:
            prob = torch.rand(1).item()
            if prob < mask_token_prob:
                data_dict['inputs_embeds'][i, j_idx, :] = mask_replacement_token_vec
            elif prob < mask_token_prob + random_token_prob:
                random_token_idx = torch.randint(0, seq_len, (1,)).item()
                while random_token_idx == j_idx:
                    random_token_idx = torch.randint(0, seq_len, (1,)).item()
                data_dict['inputs_embeds'][i, j_idx, :] = data_dict['inputs_embeds'][i, random_token_idx, :].clone()
    print(f"MLM masking applied with proportion {mask_proportion}.")

# Helper function for Adding Noise
def add_gaussian_noise(data_tensor, sigma_value, original_clean_data_for_var_calc):
    noise = torch.randn_like(data_tensor) * sigma_value * torch.sqrt(torch.var(original_clean_data_for_var_calc))
    return data_tensor + noise

# Main data preparation function
def prepare_data_for_task(config):
    # ### START: 코드 수정 (AttributeError 해결) ###
    # config.key -> config['key'] 형태로 변경
    cfg_data = config['data']
    cfg_model = config['model']
    task_type = config['task_type']
    run_name = config['run_name']

    print(f"--- Starting Data Preparation for: {run_name} ---")
    print(f"Task Type: {task_type}, Token Type: {cfg_data['token_type']}")

    # 1. Load Raw (Resized) Seismograms
    seismograms_np = np.load(cfg_data['input_seismogram_path'])['my_array']
    print(f"Loaded raw seismograms from: {cfg_data['input_seismogram_path']}, shape: {seismograms_np.shape}")
    
    # 2. Handle Tokenization Strategy (Permute if time_slice)
    if cfg_data['token_type'] == "time_slice":
        seismograms_np = seismograms_np.transpose(0, 2, 1)
        print(f"Permuted for time_slice tokenization. New shape: {seismograms_np.shape}")

    # 3. Apply Slicing
    slice_map = {
        'trace': {'dim1': 'receiver', 'dim2': 'time_sample'},
        'time_slice': {'dim1': 'time_sample', 'dim2': 'receiver'}
    }
    s1_start = cfg_data.get(f"{slice_map[cfg_data['token_type']]['dim1']}_slice_start", None)
    s1_end = cfg_data.get(f"{slice_map[cfg_data['token_type']]['dim1']}_slice_end", None)
    s2_start = cfg_data.get(f"{slice_map[cfg_data['token_type']]['dim2']}_slice_start", None)
    s2_end = cfg_data.get(f"{slice_map[cfg_data['token_type']]['dim2']}_slice_end", None)
    
    seismograms_np = seismograms_np[:, slice(s1_start, s1_end), slice(s2_start, s2_end)]
    print(f"Applied slicing. Shape after slicing: {seismograms_np.shape}")

    # Validate dimensions against model config
    if seismograms_np.shape[1] != cfg_model['sequence_length'] or seismograms_np.shape[2] != cfg_model['input_feature_dim']:
        print(f"CRITICAL WARNING: Data dimensions ({seismograms_np.shape[1]}, {seismograms_np.shape[2]}) "
              f"do NOT match model config ({cfg_model['sequence_length']}, {cfg_model['input_feature_dim']}).")
    else:
        print("Data dimensions match model config.")

    seismogram_tensor = torch.from_numpy(seismograms_np.astype(np.float32))

    # 4. Split Data
    num_samples = seismogram_tensor.shape[0]
    train_size = int(num_samples * cfg_data['train_split_ratio'])
    idx_all = torch.randperm(num_samples)
    train_indices, test_indices = idx_all[:train_size], idx_all[train_size:]
    
    datasets_dict = {}
    for split_name, indices in [("train", train_indices), ("test", test_indices)]:
        current_data = seismogram_tensor[indices]
        datasets_dict[split_name] = {
            'inputs_embeds': current_data.clone(),
            'labels': current_data.clone(),
            'mask_label': torch.zeros_like(current_data),
            'index': torch.arange(current_data.shape[0])
        }
        print(f"Created {split_name} split with {len(indices)} samples.")

    # 5. Normalization
    normalize_data([datasets_dict['train'], datasets_dict['test']])

    # --- Task-Specific Processing ---
    if task_type == "pretrain":
        cfg_pretrain = config['pretrain_params']
        for split_name in ["train", "test"]:
            augment_multiply(datasets_dict[split_name], cfg_pretrain['augmentation']['multiply_factor'])
            augment_time_shift(datasets_dict[split_name], cfg_pretrain['augmentation'], cfg_data['token_type'])
            apply_mlm_masking(datasets_dict[split_name], cfg_pretrain['masking'], cfg_model['input_feature_dim'])

    elif task_type == "finetune":
        cfg_finetune = config['finetune_params']
        # Load additional labels
        if cfg_finetune['task_name'] in ["velocity_prediction", "fault_detection", "fault_sign"]:
             vel_models_np = np.load(cfg_data['labels']['velocity_models_path'])['my_array']
             fault_info_np = np.load(cfg_data['labels']['fault_info_path'])
             fault_type_np = np.load(cfg_data['labels']['fault_type_path'])
             
             vel_profiles_1d = torch.from_numpy(vel_models_np[:, cfg_data['labels']['velocity_profile_x_coord_index'], :].astype(np.float32))
             
             datasets_dict['train'].update({
                'velocity_label': vel_profiles_1d[train_indices],
                'fault_label': torch.from_numpy(fault_info_np[train_indices].astype(np.int64)),
                'fault_type_label': torch.from_numpy(fault_type_np[train_indices].astype(np.int64))
             })
             datasets_dict['test'].update({
                'velocity_label': vel_profiles_1d[test_indices],
                'fault_label': torch.from_numpy(fault_info_np[test_indices].astype(np.int64)),
                'fault_type_label': torch.from_numpy(fault_type_np[test_indices].astype(np.int64))
             })
             print("Loaded velocity and fault labels for fine-tuning.")

        # Add noise
        if cfg_finetune.get('noise_params') and cfg_finetune['noise_params']['apply']:
            noise_cfg = cfg_finetune['noise_params']
            for split_name, d_dict in datasets_dict.items():
                current_inputs = d_dict['inputs_embeds']
                num_split_samples = current_inputs.shape[0]
                d_dict['noise_lvl'] = torch.zeros(num_split_samples)
                
                for sigma_level in [1, 2]:
                    slice_range = noise_cfg.get(f"slice_{split_name}_sigma{sigma_level}_range", [0.0, 0.0])
                    start_idx = int(num_split_samples * slice_range[0])
                    end_idx = int(num_split_samples * slice_range[1])
                    if end_idx > start_idx:
                        current_inputs[start_idx:end_idx] = add_gaussian_noise(
                            current_inputs[start_idx:end_idx],
                            noise_cfg[f'sigma{sigma_level}_value'],
                            d_dict['labels'][start_idx:end_idx]
                        )
                        d_dict['noise_lvl'][start_idx:end_idx] = sigma_level
            print("Applied Gaussian noise to inputs for fine-tuning.")
    
    # 6. Create & Save Datasets
    base_dir = os.path.join(config['base_results_dir'], run_name, cfg_data['processed_data_dir'])
    os.makedirs(base_dir, exist_ok=True)
    
    train_dataset = SSDataset(datasets_dict['train'])
    test_dataset = SSDataset(datasets_dict['test'])
    
    torch.save(train_dataset, os.path.join(base_dir, "train_data.pt"))
    torch.save(test_dataset, os.path.join(base_dir, "test_data.pt"))
    print(f"Processed data saved to: {base_dir}")

    # Optional Plotting
    if config.get('plot_prepared_example', True):
        # ... (plotting logic using dictionary access) ...
        pass

    print(f"--- Data Preparation for: {run_name} Finished ---")
    return train_dataset, test_dataset
    # ### END: 코드 수정 ###

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare seismic data for specified task.")
    parser.add_argument("--config", type=str, required=True, help="Path to the main YAML configuration file.")
    args = parser.parse_args()

    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config_params = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML configuration from {args.config}: {e}")
        exit(1)
    
    prepare_data_for_task(config_params)