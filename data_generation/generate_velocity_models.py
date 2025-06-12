#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import yaml
import argparse

def create_velocity_model(
    n_layers,
    fault_probability,
    distance_samples,
    max_model_depth_samples,
    water_depth_samples,
    cfg_vel_gen_dict
):
    """
    Creates a 2D velocity model with optional faulting, directly in km/s.
    """
    min_thickness_samples = int(max_model_depth_samples * cfg_vel_gen_dict['min_thickness_ratio'])
    max_thickness_samples = int(max_model_depth_samples * cfg_vel_gen_dict['max_thickness_ratio'])
    
    min_velocity_first_layer_val = cfg_vel_gen_dict['min_velocity_first_layer']
    max_velocity_first_layer_val = cfg_vel_gen_dict['max_velocity_first_layer']
    max_velocity_overall_val = cfg_vel_gen_dict['max_velocity_overall']
    
    model_layers = np.zeros((distance_samples, max_model_depth_samples))
    model_faulted = np.zeros((distance_samples, max_model_depth_samples))
    model_final_with_water = np.zeros((distance_samples, max_model_depth_samples + water_depth_samples))

    if max_model_depth_samples <= 0 :
        model_final_with_water[:, :water_depth_samples] = 1.5 # ## 단위 변경 ## Water velocity in km/s
        return model_final_with_water, 0, 2

    min_thickness_samples = max(1, min_thickness_samples)
    max_thickness_samples = max(min_thickness_samples, max_thickness_samples)
    
    # Layer thickness calculation (same as before)
    if n_layers == 1:
        thicknesses = np.array([max_model_depth_samples])
    elif n_layers > 1:
        raw_thicknesses = np.random.randint(min_thickness_samples, max_thickness_samples + 1, n_layers - 1)
        current_sum_raw = np.sum(raw_thicknesses)
        if current_sum_raw == 0: current_sum_raw = 1 # Avoid division by zero
        thicknesses_scaled = np.round(raw_thicknesses * (max_model_depth_samples / current_sum_raw)).astype(int)
        thicknesses_scaled = np.maximum(thicknesses_scaled, 1)
        thicknesses = np.zeros(n_layers, dtype=int)
        thicknesses[:-1] = thicknesses_scaled
        thicknesses[-1] = max_model_depth_samples - np.sum(thicknesses_scaled)
        if thicknesses[-1] <= 0: # Robustness fix
            avg_thick = max(1, max_model_depth_samples // n_layers)
            thicknesses = np.full(n_layers, avg_thick, dtype=int)
            thicknesses[-1] = max_model_depth_samples - np.sum(thicknesses[:-1])
    else: # n_layers == 0
        thicknesses = []

    # ## 단위 변경 ## Use np.random.uniform for float velocities in km/s
    first_layer_velocity = np.random.uniform(
        min_velocity_first_layer_val, max_velocity_first_layer_val
    )
    velocities = [first_layer_velocity]

    if n_layers > 1:
        st = (max_velocity_overall_val - velocities[-1]) / float(n_layers - 1)
        for _ in range(n_layers - 1):
            increment = np.random.uniform(st * 0.8, st * 1.2)
            next_velocity = velocities[-1] + increment
            next_velocity = min(next_velocity, max_velocity_overall_val)
            velocities.append(next_velocity)
            
    current_depth = 0
    for i_layer in range(n_layers):
        thickness = thicknesses[i_layer]
        velocity = velocities[i_layer]
        if thickness > 0:
            model_layers[:, current_depth : current_depth + thickness] = velocity
            current_depth += thickness
    if current_depth < max_model_depth_samples:
        model_layers[:, current_depth:] = velocities[-1] if velocities else 1.5

    # Faulting logic (same as before)
    fault_exist = 0; fault_type = 2
    if np.random.rand() < fault_probability:
        # ... (faulting logic remains the same)
        fault_exist = 1; fault_type = np.random.randint(0, 2)
        fault_point_x = np.random.randint(int(distance_samples * 0.3), int(distance_samples * 0.7) + 1)
        fault_dip_degrees = np.random.uniform(cfg_vel_gen_dict['fault_dip_min_degrees'], cfg_vel_gen_dict['fault_dip_max_degrees'])
        min_throw_s = int(max_model_depth_samples * cfg_vel_gen_dict['fault_throw_min_ratio'])
        max_throw_s = int(max_model_depth_samples * cfg_vel_gen_dict['fault_throw_max_ratio'])
        fault_throw_samples = np.random.randint(max(1, min_throw_s), max(min_throw_s, max_throw_s) + 1)
        fault_dip_direction = np.random.choice([-1.0, 1.0])
        model_faulted[:, :] = velocities[0]
        for iz in range(max_model_depth_samples):
            for ix in range(distance_samples):
                z_on_fault_plane = fault_dip_direction * math.tan(math.radians(fault_dip_degrees)) * (ix - fault_point_x)
                if iz >= z_on_fault_plane:
                    if fault_type == 0: src_z = iz - fault_throw_samples
                    else: src_z = iz + fault_throw_samples
                    if 0 <= src_z < max_model_depth_samples: model_faulted[ix, iz] = model_layers[ix, src_z]
                    elif src_z < 0: model_faulted[ix, iz] = velocities[0]
                    else: model_faulted[ix, iz] = velocities[-1]
                else: model_faulted[ix, iz] = model_layers[ix, iz]
    else:
        model_faulted = model_layers

    ## --- 단위 변경: 물 속도를 1.5 km/s로 설정 --- ##
    model_final_with_water[:, :water_depth_samples] = 1.5
    if max_model_depth_samples > 0:
        model_final_with_water[:, water_depth_samples:] = model_faulted

    return model_final_with_water, fault_exist, fault_type

def main_worker(config):
    cfg_vel_main = config['velocity_model_generation']
    cfg_seis_main = config['seismogram_generation']
    print("Starting velocity model generation (in km/s)...")

    output_dir = cfg_vel_main['output_directory']
    os.makedirs(output_dir, exist_ok=True)

    velocity_models_array = np.zeros(
        (cfg_vel_main['n_models'], cfg_vel_main['distance_samples'], cfg_vel_main['depth_samples'])
    )
    fault_info_list = []; fault_type_list = []

    for i in range(cfg_vel_main['n_models']):
        if (i + 1) % 50 == 0: print(f"Generating model {i+1}/{cfg_vel_main['n_models']}")
        
        n_layers_rand = np.random.randint(cfg_vel_main['min_layers'], cfg_vel_main['max_layers'])
        water_depth_rand = np.random.randint(cfg_vel_main['water_depth_min_samples'], cfg_vel_main['water_depth_max_samples'])
        max_model_depth_for_layers = cfg_vel_main['depth_samples'] - water_depth_rand
        
        if max_model_depth_for_layers < n_layers_rand and n_layers_rand > 0:
            n_layers_rand = max(0, max_model_depth_for_layers)
        
        velocity_model, fault_exists, fault_type_val = create_velocity_model(
            n_layers_rand, cfg_vel_main['fault_probability'],
            cfg_vel_main['distance_samples'], max_model_depth_for_layers,
            water_depth_rand, cfg_vel_main
        )
        velocity_models_array[i] = velocity_model
        fault_info_list.append(fault_exists); fault_type_list.append(fault_type_val)

        if i < config.get('num_examples_to_plot', 3):
            fig, ax = plt.subplots(figsize=(8, 6))
            extent = [
                0, cfg_vel_main['distance_samples'] * cfg_seis_main['dx_km'],
                cfg_vel_main['depth_samples'] * cfg_seis_main['dz_km'], 0
            ]
            ## --- 단위 변경: 플롯 라벨 및 범위 수정 --- ##
            im = ax.imshow(
                velocity_model.T, aspect='auto', cmap='jet', origin='upper',
                vmin=1.4, vmax=cfg_vel_main['max_velocity_overall'], extent=extent
            )
            plt.colorbar(im, ax=ax, label='Velocity (km/s)') # 라벨 변경
            ax.set_xlabel('Distance (km)'); ax.set_ylabel('Depth (km)')
            ax.xaxis.set_label_position('top'); ax.xaxis.tick_top()
            plt.title(f"Generated Velocity Model {i+1} (Fault: {fault_exists}, Type: {fault_type_val})")
            plt.tight_layout()
            plot_filename = os.path.join(output_dir, f"velocity_model_example_{i+1}.png")
            fig.savefig(plot_filename)
            print(f"Saved example plot: {plot_filename}")
            plt.close(fig)

    path_models = os.path.join(output_dir, cfg_vel_main['velocity_models_filename'])
    path_fault_info = os.path.join(output_dir, cfg_vel_main['fault_info_filename'])
    path_fault_type = os.path.join(output_dir, cfg_vel_main['fault_type_filename'])

    np.savez_compressed(path_models, my_array=velocity_models_array)
    np.save(path_fault_info, np.array(fault_info_list))
    np.save(path_fault_type, np.array(fault_type_list))

    print(f"Velocity models (in km/s) saved to {path_models}")
    print("Velocity model generation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 2D Velocity Models for Seismic Simulation.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config_yaml = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML configuration from {args.config}: {e}")
        exit(1)
        
    main_worker(config_yaml)