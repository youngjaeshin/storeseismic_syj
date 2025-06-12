#!/usr/bin/env python
import numpy as np
import os
import yaml
import argparse
import matplotlib.pyplot as plt

def apply_time_amplification(data, time_interval, total_time, power):
    if not power or power <= 0:
        return data
    n_models, n_receivers, n_timeseries = data.shape
    time_values = np.linspace(0, total_time, n_timeseries)**power
    amplified_data = data * time_values[np.newaxis, np.newaxis, :]
    return amplified_data

def resize_by_slicing(data, receiver_step, time_step):
    """
    Resizes the data using direct step slicing.
    """
    n_models, n_receivers_original, n_time_original = data.shape
    print(f"Original shape per model: ({n_receivers_original}, {n_time_original})")
    print(f"Downsampling with receiver step: {receiver_step}, time step: {time_step}")

    resized_data = data[:, ::receiver_step, ::time_step]
    
    print(f"Shape after slicing: ({resized_data.shape[1]}, {resized_data.shape[2]})")
    return resized_data

def resize_seismic_data(config_resize):
    print("Starting seismogram resizing...")
    cfg = config_resize

    input_path = os.path.join(cfg['input_directory'], cfg['input_seismograms_filename'])
    output_path = os.path.join(cfg['output_directory'], cfg['output_seismograms_filename'])
    os.makedirs(cfg['output_directory'], exist_ok=True)
    
    try:
        data_load = np.load(input_path)
        seismograms_whole = data_load['my_array']
        print(f"Loaded seismograms from {input_path}, shape: {seismograms_whole.shape}")
    except Exception as e:
        print(f"Error loading seismograms from {input_path}: {e}")
        return

    # Time amplification (if enabled in config)
    if cfg.get('apply_time_amplification', False):
        power = cfg['time_amplification_power']
        dt_param = cfg.get('dt_for_amplification_s', 0.001)
        total_time_param = cfg.get('total_time_for_amplification_s', 2.0)
        print(f"Applying time amplification with power {power}...")
        seismograms_whole = apply_time_amplification(
            seismograms_whole,
            time_interval=dt_param,
            total_time=total_time_param,
            power=power
        )
    
    # ### START: 코드 수정 (스텝 크기 직접 사용) ###
    receiver_step = cfg.get('receiver_downsample_step', 1)
    time_step = cfg.get('time_downsample_step', 1)
    
    resized_array = resize_by_slicing(
        seismograms_whole,
        receiver_step,
        time_step
    )
    # ### END: 코드 수정 ###

    print(f"Final resized shape: {resized_array.shape}")

    try:
        np.savez_compressed(output_path, my_array=resized_array)
        print(f"Resized seismograms saved to {output_path}")
    except Exception as e:
        print(f"Error saving resized seismograms to {output_path}: {e}")
        return

    # Optional: Plot an example of a resized seismogram
    if cfg.get('plot_example', True) and resized_array.shape[0] > 0:
        try:
            print("Plotting an example of a resized seismogram...")
            example_seismogram = resized_array[0]
            perc = np.percentile(np.abs(example_seismogram), 99.5)
            if perc == 0: perc = 1.0

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(example_seismogram.T, cmap='seismic', aspect='auto', vmin=-perc, vmax=perc)
            ax.set_ylabel('Time Samples (Resized)')
            ax.set_xlabel('Receiver Index')
            ax.set_title(f'Example Resized Seismogram (0) - Shape: {example_seismogram.shape}')
            plt.colorbar(ax.images[0], ax=ax, label='Amplitude')
            
            plot_output_dir = os.path.join(cfg['output_directory'], "plots")
            os.makedirs(plot_output_dir, exist_ok=True)
            example_plot_path = os.path.join(plot_output_dir, "resized_seismogram_example.png")
            fig.savefig(example_plot_path)
            print(f"Saved example plot to {example_plot_path}")
            plt.close(fig)
        except Exception as e:
            print(f"Could not plot example: {e}")

    print("Seismogram resizing finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize seismic data.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file for resizing (e.g., configs/resize.yaml).")
    args = parser.parse_args()

    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config_params = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML configuration from {args.config}: {e}")
        exit(1)
    
    resize_seismic_data(config_params)