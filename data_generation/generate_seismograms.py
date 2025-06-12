#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import yaml
import argparse

def ricker_wavelet(f, length=2.0, dt=0.001, peak_time=0.1):
    total_samples = int(length / dt)
    delay_samples = int(peak_time / dt)
    t_formula = (np.arange(total_samples) - delay_samples) * dt
    pi2 = (np.pi * f * t_formula)**2
    y = (1.0 - 2.0 * pi2) * np.exp(-pi2)
    return y

def main_worker(config):
    cfg_sg = config['seismogram_generation']
    cfg_vel = config['velocity_model_generation']
    print("Starting seismogram generation...")

    os.makedirs(cfg_sg['output_directory'], exist_ok=True)

    vel_models_path = os.path.join(cfg_sg['input_velocity_model_directory'], cfg_sg['input_velocity_models_filename'])
    velocity_models = np.load(vel_models_path)['my_array']
    print(f"Loaded {velocity_models.shape[0]} velocity models (in km/s).")

    t_time = cfg_sg['t_time_seconds']; dt = cfg_sg['dt_seconds']; nt = int(round(t_time / dt))
    dx = cfg_sg['dx_km']; dz = cfg_sg['dz_km']
    f0 = cfg_sg['f0_ricker_hz']; source_peak_time = cfg_sg['peak_loc_ricker_seconds']
    source_wavelet = ricker_wavelet(f0, length=t_time, dt=dt, peak_time=source_peak_time)

    nx_vel, nz_vel = velocity_models[0].shape
    pml = int(nx_vel / 10)
    a1 = (9.0/8.0)/dx; a2 = (-1.0/24.0)/dx
    R = 0.001
    n_receivers = nx_vel - 2
    seismograms_all_models = np.zeros((velocity_models.shape[0], n_receivers, nt), dtype=np.float32)

    for imodel, vp_physical in enumerate(velocity_models):
        if (imodel + 1) % cfg_sg.get('print_model_interval', 50) == 0:
            print(f"Processing model {imodel+1}/{velocity_models.shape[0]}")

        npx, npz = nx_vel + 2*pml + 4, nz_vel + pml + 4
        
        p1 = np.zeros((npx, npz), dtype=np.float32); p2 = np.zeros_like(p1)
        ax = np.zeros_like(p1); az = np.zeros_like(p1)
        ppsix1 = np.zeros_like(p1); ppsix2 = np.zeros_like(p1); ppsiz1 = np.zeros_like(p1); ppsiz2 = np.zeros_like(p1)
        apsix1 = np.zeros_like(p1); apsix2 = np.zeros_like(p1); apsiz1 = np.zeros_like(p1); apsiz2 = np.zeros_like(p1)
        
        vp_padded = np.pad(vp_physical, ((pml+2, pml+2), (2, pml+2)), 'edge')

        min_ratio = cfg_sg.get('source_x_position_min_ratio', 0.0)
        max_ratio = cfg_sg.get('source_x_position_max_ratio', 1.0)
        min_shot_x_physical = int(nx_vel * min_ratio)
        max_shot_x_physical = int(nx_vel * max_ratio)
        random_shot_x_physical = np.random.randint(min_shot_x_physical, max_shot_x_physical) if max_shot_x_physical > min_shot_x_physical else min_shot_x_physical
        shot_x = random_shot_x_physical + pml + 2
        shot_z = 3 
        srcf = np.zeros((npx, npz))
        srcf[shot_x, shot_z] = 1.0

        receivers_z_pos = 3
        receivers_x_pos = np.linspace(pml + 3, nx_vel + pml, n_receivers, dtype=int)
        seismogram_single_model = np.zeros((n_receivers, nt))

        for it in range(nt):
            p3 = np.zeros_like(p1)
            pdx2 = np.zeros((npx, npz), dtype=np.float32); pdz2 = np.zeros((npx, npz), dtype=np.float32)
            pdx2[2:-2, 2:-2] = a1 * (p2[3:-1, 2:-2] - p2[2:-2, 2:-2]) + a2 * (p2[4:, 2:-2] - p2[1:-3, 2:-2])
            pdz2[2:-2, 2:-2] = a1 * (p2[2:-2, 3:-1] - p2[2:-2, 2:-2]) + a2 * (p2[2:-2, 4:] - p2[2:-2, 1:-3])
            dpml0 = math.log(1. / R) * 3. * vp_padded[2:-2, 2:-2] / (2. * dx * pml)
            profile_lin_left = (np.arange(pml, 0, -1).reshape((pml, 1)) / float(pml))**2
            dpml = dpml0[2:pml+2, :] * profile_lin_left; damp_left = np.exp(-dpml * dt)
            ppsix2[2:pml+2, 2:-2] = damp_left * ppsix1[2:pml+2, 2:-2] + (damp_left - 1) * pdx2[2:pml+2, 2:-2]
            profile_lin_right = (np.arange(1, pml + 1).reshape((pml, 1)) / float(pml))**2
            dpml = dpml0[-pml-2:-2, :] * profile_lin_right; damp_right = np.exp(-dpml * dt)
            ppsix2[-pml-2:-2, 2:-2] = damp_right * ppsix1[-pml-2:-2, 2:-2] + (damp_right - 1) * pdx2[-pml-2:-2, 2:-2]
            profile_lin_bottom = (np.arange(1, pml + 1).reshape((1, pml)) / float(pml))**2
            dpml = dpml0[:, -pml-2:-2] * profile_lin_bottom; damp_bottom = np.exp(-dpml * dt)
            ppsiz2[2:-2, -pml-2:-2] = damp_bottom * ppsiz1[2:-2, -pml-2:-2] + (damp_bottom - 1) * pdz2[2:-2, -pml-2:-2]
            ax_field = pdx2 + ppsix2; az_field = pdz2 + ppsiz2
            az_field[:, 1] = az_field[:, 2]; az_field[:, 0] = az_field[:, 3]
            adx = np.zeros((npx, npz), dtype=np.float32); adz = np.zeros((npx, npz), dtype=np.float32)
            adx[2:-2, 2:-2] = a1 * (ax_field[2:-2, 2:-2] - ax_field[1:-3, 2:-2]) + a2 * (ax_field[3:-1, 2:-2] - ax_field[:-4, 2:-2])
            adz[2:-2, 2:-2] = a1 * (az_field[2:-2, 2:-2] - az_field[2:-2, 1:-3]) + a2 * (az_field[2:-2, 3:-1] - az_field[2:-2, :-4])
            apsix2[2:pml+2, 2:-2] = damp_left * apsix1[2:pml+2, 2:-2] + (damp_left - 1) * adx[2:pml+2, 2:-2]
            apsix2[-pml-2:-2, 2:-2] = damp_right * apsix1[-pml-2:-2, 2:-2] + (damp_right - 1) * adx[-pml-2:-2, 2:-2]
            apsiz2[2:-2, -pml-2:-2] = damp_bottom * apsiz1[2:-2, -pml-2:-2] + (damp_bottom - 1) * adz[2:-2, -pml-2:-2]
            px2 = adx + apsix2; pz2 = adz + apsiz2
            p3 = 2 * p2 - p1 + (vp_padded * dt)**2 * (px2 + pz2 + srcf * source_wavelet[it] * 1e6)
            p3[:, 0:2] = 0.
            p1, p2 = p2, p3
            ppsix1, ppsiz1 = ppsix2, ppsiz2; apsix1, apsiz1 = apsix2, apsiz2
            for i_rec, rec_x in enumerate(receivers_x_pos):
                seismogram_single_model[i_rec, it] = p3[rec_x, receivers_z_pos]
        
        seismograms_all_models[imodel, :, :] = seismogram_single_model
        
        ### --- START: 그림 저장하는 코드 (pass 제거) --- ###
        if imodel < cfg_sg.get('num_seismogram_examples_to_plot', 0):
            fig_seis, ax_seis = plt.subplots(figsize=(7, 8))
            
            # seismogram_single_model을 사용해야 현재 모델의 결과가 그려집니다.
            current_seismogram = seismogram_single_model
            
            perc = np.percentile(np.abs(current_seismogram), 99.8)
            if perc == 0: perc = np.max(np.abs(current_seismogram))
            if perc == 0: perc = 1.0 # 최댓값도 0일 경우 대비

            dist_extent_km = nx_vel * cfg_sg['dx_km']
            time_extent_s = nt * dt

            ax_seis.imshow(current_seismogram.T, cmap='seismic', aspect='auto',
                           vmin=-perc, vmax=perc, extent=[0, dist_extent_km, time_extent_s, 0])
            ax_seis.set_ylabel('Time (s)')
            ax_seis.set_xlabel('Distance (km)')
            ax_seis.xaxis.set_label_position('top')
            ax_seis.xaxis.tick_top()
            plt.title(f"Seismogram for Model {imodel+1}")
            plt.tight_layout()
            
            seis_plot_path = os.path.join(cfg_sg['output_directory'], f"seismogram_example_{imodel+1}.png")
            fig_seis.savefig(seis_plot_path)
            print(f"Saved example seismogram plot to {seis_plot_path}")
            plt.close(fig_seis)
        ### --- END: 그림 저장하는 코드 --- ###

    seis_output_path = os.path.join(cfg_sg['output_directory'], cfg_sg['seismograms_filename'])
    np.savez_compressed(seis_output_path, my_array=seismograms_all_models)
    print(f"All seismograms saved to {seis_output_path}")
    print("Seismogram generation finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config_yaml = yaml.safe_load(f)
    main_worker(config_yaml)