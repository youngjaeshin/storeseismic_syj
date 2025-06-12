import numpy as np
import matplotlib.pyplot as plt
import math

# --- 1. 시뮬레이션 파라미터 ---
nx, nz = 200, 150
dx, dz = 0.01, 0.01    # km
dt = 0.001             # s
nt = 4000
f0 = 10.0
peak_time = 0.2

# --- 2. 간단한 속도 모델 생성 (km/s 단위) ---
vp = np.full((nx, nz), 1.5, dtype=np.float32)
vp[:, int(nz/2):] = 2.5
print(f"간단한 속도 모델 생성 완료 (km/s). Shape: {vp.shape}")

# --- 3. 소스 웨이브렛 생성 ---
t = np.arange(nt) * dt
t_formula = t - peak_time
pi2 = (np.pi * f0 * t_formula)**2
source_wavelet = (1.0 - 2.0 * pi2) * np.exp(-pi2)

# --- 4. FDM 격자 및 배열 초기화 ---
pml = int(nx / 10)
npx, npz = nx + 2*pml + 4, nz + pml + 4 # 원래 코드의 격자 크기 계산 방식

# 원래 코드의 미분 계수 계산 방식 (dx는 km 단위)
a1 = (9.0/8.0)/dx
a2 = (-1.0/24.0)/dx
R = 0.001

# 배열 초기화
p1 = np.zeros((npx, npz), dtype=np.float32)
p2 = np.zeros((npx, npz), dtype=np.float32)
ax = np.zeros((npx, npz), dtype=np.float32)
az = np.zeros((npx, npz), dtype=np.float32)
ppsix1 = np.zeros((npx, npz), dtype=np.float32)
ppsix2 = np.zeros((npx, npz), dtype=np.float32)
ppsiz1 = np.zeros((npx, npz), dtype=np.float32)
ppsiz2 = np.zeros((npx, npz), dtype=np.float32)
apsix1 = np.zeros((npx, npz), dtype=np.float32)
apsix2 = np.zeros((npx, npz), dtype=np.float32)
apsiz1 = np.zeros((npx, npz), dtype=np.float32)
apsiz2 = np.zeros((npx, npz), dtype=np.float32)

# 속도 모델 패딩
vp_padded = np.pad(vp, ((pml+2, pml+2), (2, pml+2)), 'edge')

# 소스 위치
srcf = np.zeros((npx, npz))
shot_x, shot_z = pml + 2, 3
srcf[shot_x, shot_z] = 1.0

# 수신기 위치
n_receivers = nx
receivers_z = 3
receivers_x = np.arange(pml + 2, pml + 2 + n_receivers) # 간결하게 수정
seismogram = np.zeros((n_receivers, nt))

print("FDM 시뮬레이션을 시작합니다 (원래 코드 로직 기반)...")

# --- 5. 시간 반복문 (원래 코드의 로직 사용) ---
for it in range(nt):
    p3 = np.zeros_like(p1)

    # 임시 배열 초기화
    pdx2_slice = np.zeros((npx-4, npz-4), dtype=np.float32)
    pdz2_slice = np.zeros((npx-4, npz-4), dtype=np.float32)

    # 1차 미분 계산 (원래 코드 방식)
    pdx2_slice = a1 * (p2[3:-1, 2:-2] - p2[2:-2, 2:-2]) + a2 * (p2[4:, 2:-2] - p2[1:-3, 2:-2])
    pdz2_slice = a1 * (p2[2:-2, 3:-1] - p2[2:-2, 2:-2]) + a2 * (p2[2:-2, 4:] - p2[2:-2, 1:-3])

    dpml0 = math.log(1. / R) * 3. * vp_padded[2:-2, 2:-2] / (2. * dx * pml)
    
    # PML 업데이트 (원래 코드 방식)
    profile_lin_left = (np.arange(pml, 0, -1).reshape((pml, 1)) / float(pml))**2
    dpml = dpml0[2:pml+2, :] * profile_lin_left
    damp_left = np.exp(-dpml * dt)
    ppsix2[2:pml+2, 2:-2] = damp_left * ppsix1[2:pml+2, 2:-2] + (damp_left - 1) * pdx2_slice[0:pml, :]

    profile_lin_right = (np.arange(1, pml + 1).reshape((pml, 1)) / float(pml))**2
    dpml = dpml0[-pml-2:-2, :] * profile_lin_right
    damp_right = np.exp(-dpml * dt)
    ppsix2[-pml-2:-2, 2:-2] = damp_right * ppsix1[-pml-2:-2, 2:-2] + (damp_right - 1) * pdx2_slice[-pml:, :]

    profile_lin_bottom = (np.arange(1, pml + 1).reshape((1, pml)) / float(pml))**2
    dpml = dpml0[:, -pml-2:-2] * profile_lin_bottom
    damp_bottom = np.exp(-dpml * dt)
    ppsiz2[2:-2, -pml-2:-2] = damp_bottom * ppsiz1[2:-2, -pml-2:-2] + (damp_bottom - 1) * pdz2_slice[:, -pml:]
    
    ax[2:-2, 2:-2] = pdx2_slice + ppsix2[2:-2, 2:-2]
    az[2:-2, 2:-2] = pdz2_slice + ppsiz2[2:-2, 2:-2]

    # 자유 표면 (원래 코드 방식)
    az[:, 1] = az[:, 2]
    az[:, 0] = az[:, 3]
    
    # 2차 미분항 계산 (원래 코드 방식)
    adx_slice = a1 * (ax[2:-2, 2:-2] - ax[1:-3, 2:-2]) + a2 * (ax[3:-1, 2:-2] - ax[:-4, 2:-2])
    adz_slice = a1 * (az[2:-2, 2:-2] - az[2:-2, 1:-3]) + a2 * (az[2:-2, 3:-1] - az[2:-2, :-4])
    
    # PML 업데이트 (원래 코드 방식)
    apsix2[2:pml+2, 2:-2] = damp_left * apsix1[2:pml+2, 2:-2] + (damp_left - 1) * adx_slice[0:pml, :]
    apsix2[-pml-2:-2, 2:-2] = damp_right * apsix1[-pml-2:-2, 2:-2] + (damp_right - 1) * adx_slice[-pml:, :]
    apsiz2[2:-2, -pml-2:-2] = damp_bottom * apsiz1[2:-2, -pml-2:-2] + (damp_bottom - 1) * adz_slice[:, -pml:]
    
    px2_slice = adx_slice + apsix2[2:-2, 2:-2]
    pz2_slice = adz_slice + apsiz2[2:-2, 2:-2]

    # ### 최종 핵심 수정: 소스 스케일 보정 적용 ###
    p3[2:-2, 2:-2] = 2 * p2[2:-2, 2:-2] - p1[2:-2, 2:-2] + \
        (vp_padded[2:-2, 2:-2] * dt)**2 * \
        (px2_slice + pz2_slice + srcf[2:-2, 2:-2] * source_wavelet[it] * 1e6) # 1e6 스케일링
    
    # p3 자유 표면 조건
    p3[:, 0:2] = 0.
    
    # 다음 스텝을 위해 배열 업데이트
    p1, p2 = p2, p3
    ppsix1, ppsiz1 = ppsix2, ppsiz2
    apsix1, apsiz1 = apsix2, apsiz2

    # 수신기 기록
    seismogram[:, it] = p3[receivers_x, receivers_z]
    
    if it % 500 == 0:
        print(f"Time step {it}: max pressure = {np.max(np.abs(p3)):.2e}")

print("FDM 시뮬레이션 완료.")

# --- 6. 결과 시각화 ---
plt.figure(figsize=(10, 7))
v_clip = np.max(np.abs(p3)) * 0.1
plt.imshow(p3.T, cmap='seismic', aspect='auto', vmin=-v_clip, vmax=v_clip)
plt.title(f'Wavefield Snapshot at Final Time Step')
plt.xlabel('X grid points'); plt.ylabel('Z grid points')
plt.colorbar(label='Amplitude')
plt.show()

plt.figure(figsize=(10, 7))
perc = np.percentile(np.abs(seismogram), 99.8)
if perc == 0: perc = 1.0
plt.imshow(seismogram, cmap='seismic', aspect='auto', vmin=-perc, vmax=perc)
plt.title('Generated Seismogram')
plt.xlabel('Receiver Index'); plt.ylabel('Time Steps')
plt.colorbar(label='Amplitude')
plt.show()