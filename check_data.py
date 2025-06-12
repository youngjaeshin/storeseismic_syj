import numpy as np
import matplotlib.pyplot as plt

# 생성된 데이터 파일의 경로
file_path = './synthetic_data/raw_generated/seismograms_whole.npz'

try:
    # .npz 파일 로드
    data = np.load(file_path)
    seismograms = data['my_array']
    
    print(f"파일 로드 성공: {file_path}")
    print(f"전체 데이터 형태: {seismograms.shape}")
    
    # 첫 번째 모델의 탄성파동 기록을 가져옵니다.
    first_seismogram = seismograms[0]
    
    # --- 데이터 값 확인을 위한 통계 정보 출력 (가장 중요한 부분) ---
    max_val = np.max(first_seismogram)
    min_val = np.min(first_seismogram)
    mean_val = np.mean(first_seismogram)
    std_val = np.std(first_seismogram)
    
    print("\n--- 첫 번째 모델 데이터 통계 ---")
    print(f"최댓값 (Max): {max_val}")
    print(f"최솟값 (Min): {min_val}")
    print(f"평균 (Mean): {mean_val}")
    print(f"표준편차 (Std Dev): {std_val}")
    print("---------------------------------")
    
    if max_val == 0.0 and min_val == 0.0:
        print("\n진단: 데이터의 모든 값이 0입니다. FDM 시뮬레이션에서 파동이 생성/기록되지 않았습니다.")
    else:
        print("\n진단: 데이터에 값이 존재합니다. 플롯을 다시 확인합니다.")
        
        # 데이터 시각화
        plt.figure(figsize=(8, 6))
        perc = np.percentile(np.abs(first_seismogram), 99.5)
        if perc < 1e-9: perc = np.max(np.abs(first_seismogram)) # 값이 너무 작을 경우 최댓값으로 대체
        if perc == 0: perc = 1.0 # 최댓값도 0일 경우
        
        plt.imshow(first_seismogram.T, aspect='auto', cmap='seismic', vmin=-perc, vmax=perc)
        plt.title(f'Example Seismogram (Model 0) - Max Abs Value: {np.max(np.abs(first_seismogram)):.2e}')
        plt.xlabel('Receiver Index')
        plt.ylabel('Time Samples')
        plt.colorbar(label='Amplitude')
        plt.show()

except FileNotFoundError:
    print(f"오류: 파일을 찾을 수 없습니다 - {file_path}")
except KeyError:
    print(f"오류: 파일 내에서 'my_array' 키를 찾을 수 없습니다.")
except Exception as e:
    print(f"오류 발생: {e}")