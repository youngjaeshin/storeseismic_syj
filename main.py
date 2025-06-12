import torch
import torch.optim as optim
import yaml
import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

# 우리가 만든 모듈들을 임포트합니다.
from src.data_utils import prepare_data_for_task, SSDataset
from src.models import create_bert_config, get_bert_model, BertOnlyMLMHead, DenoisingHead, FaultpredHead, VelpredHead
from src.training import Trainer, get_loss_function
from src.evaluation import (
    plot_loss_curve, 
    plot_attention_maps,
    plot_pretrain_reconstruction_results, # 사전 학습 결과 플롯 함수
    plot_denoising_results, 
    plot_velocity_prediction_results,
    plot_confusion_matrix,
    calculate_classification_metrics,
    calculate_denoising_metrics,
    calculate_velocity_prediction_metrics
)

def run_experiment(config):
    """하나의 실험(설정 파일 기준) 전체를 실행하는 메인 함수"""
    
    run_name = config['run_name']
    base_results_dir = config['base_results_dir']
    run_output_dir = os.path.join(base_results_dir, run_name)
    training_output_dir = os.path.join(run_output_dir, config['training']['output_dir'])
    plots_output_dir = os.path.join(run_output_dir, "plots")
    os.makedirs(training_output_dir, exist_ok=True)
    os.makedirs(plots_output_dir, exist_ok=True)
    
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    processed_data_dir = os.path.join(run_output_dir, config['data']['processed_data_dir'])
    train_data_path = os.path.join(processed_data_dir, "train_data.pt")
    test_data_path = os.path.join(processed_data_dir, "test_data.pt")

    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        print("Processed data not found. Running data preparation...")
        prepare_data_for_task(config)
    
    print("Loading processed data...")
    train_dataset = torch.load(train_data_path, weights_only=False)
    test_dataset = torch.load(test_data_path, weights_only=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    print("DataLoaders created.")

    print("Setting up model...")
    model_config = create_bert_config(config['model'], config['data'])
    
    if config['task_type'] == 'pretrain':
        model = get_bert_model(model_config)
        model.cls = BertOnlyMLMHead(model_config)
    elif config['task_type'] == 'finetune':
        # ... (파인튜닝 모델 설정 로직)
        pass # 현재는 사전 학습 테스트에 집중
    
    model.to(device)
    print("Model setup complete.")

    # --- START: 학습된 모델이 있으면 학습을 건너뛰는 로직 추가 ---
    checkpoint_path = os.path.join(training_output_dir, "best_checkpoint.pt")
    if os.path.exists(checkpoint_path) and not config.get('force_retrain', False):
        print(f"Found existing trained model at {checkpoint_path}. Loading weights and skipping training.")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        trained_model = model
    else:
        # 모델 학습
        optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
        loss_fn = get_loss_function(config['training']['loss_function'])
        trainer = Trainer(model, optimizer, loss_fn, device, config)
        trained_model, train_losses, val_losses, last_attentions = trainer.train(train_dataloader, test_dataloader)
        plot_loss_curve(train_losses, val_losses, os.path.join(plots_output_dir, "loss_curve.png"), run_name)
    # --- END: 학습 건너뛰기 로직 ---

    # 최종 평가
    print("Starting final evaluation on test set...")
    evaluate_model(trained_model, test_dataloader, device, config, plots_output_dir)
    
    print("Experiment finished successfully!")


def evaluate_model(model, test_dataloader, device, config, output_dir):
    """학습된 모델을 테스트 데이터셋으로 평가하고 결과를 저장하는 함수"""
    model.eval()
    all_inputs, all_labels, all_predictions = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            # ... (데이터를 device로 옮기는 로직은 동일)
            batch_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            inputs_embeds = batch_on_device.get('inputs_embeds').float()
            
            # 사전학습/파인튜닝에 따라 예측 및 라벨 설정
            task_name = config.get('finetune_params', {}).get('task_name', 'pretrain')
            
            if task_name == 'pretrain':
                predictions = model(inputs_embeds=inputs_embeds).logits
                labels = batch_on_device['labels']
            # ... (이하 파인튜닝 로직) ...
            
            all_inputs.append(inputs_embeds.cpu()); all_labels.append(labels.cpu()); all_predictions.append(predictions.cpu())

    all_inputs = torch.cat(all_inputs, dim=0); all_labels = torch.cat(all_labels, dim=0); all_predictions = torch.cat(all_predictions, dim=0)

    task_name = config.get('finetune_params', {}).get('task_name', 'pretrain')
    
    # --- START: 코드 수정 (사전 학습 평가 로직 추가) ---
    if task_name == 'pretrain':
        print("Evaluating pre-training reconstruction performance...")
        # 'inputs'는 마스킹된 데이터, 'predictions'는 복원된 결과, 'labels'는 원본 데이터입니다.
        plot_pretrain_reconstruction_results(
            inputs_list=all_inputs,
            reconstructed_list=all_predictions,
            labels_list=all_labels,
            output_dir=output_dir,
            run_name=config['run_name'],
            num_examples=4 # 4개의 예시를 플롯합니다.
        )
    # --- END: 코드 수정 ---
    elif task_name == 'denoising':
        # ... (Denoising 평가 로직)
        pass
    # ... (다른 파인튜닝 평가 로직)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main experiment runner for Seismic BERT project.")
    parser.add_argument("--config", type=str, required=True, help="Path to the main YAML configuration file.")
    args = parser.parse_args()
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config_params = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML configuration from {args.config}: {e}")
        exit(1)
    run_experiment(config_params)