import torch
import torch.optim as optim
import yaml
import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 우리가 만든 모듈들을 임포트합니다.
# BertOnlyMLMHead를 임포트 목록에 추가합니다.
from src.models import create_bert_config, get_bert_model, BertOnlyMLMHead, DenoisingHead, FaultpredHead, VelpredHead
from src.training import Trainer, get_loss_function
from src.evaluation import (
    plot_loss_curve, plot_attention_maps, plot_denoising_results, 
    plot_velocity_prediction_results, plot_confusion_matrix,
    calculate_classification_metrics, calculate_denoising_metrics,
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
    
    # --- START: 코드 수정 (사전 학습 헤드 교체) ---
    if config['task_type'] == 'pretrain':
        # BertForMaskedLM 모델을 생성합니다.
        model = get_bert_model(model_config)
        # 기본 헤드(self.cls)를 우리가 만든 단순한 BertOnlyMLMHead로 교체합니다.
        print("Replacing default prediction head with custom BertOnlyMLMHead for pre-training.")
        model.cls = BertOnlyMLMHead(model_config)
        
    elif config['task_type'] == 'finetune':
        finetune_params = config['finetune_params']
        # 사전 학습된 BERT 모델 불러오기
        model = get_bert_model(model_config, checkpoint_path=finetune_params['pretrained_model_path'])
        
        # 파인튜닝 작업에 맞는 헤드(Head)로 교체
        task_name = finetune_params['task_name']
        print(f"Attaching head for fine-tuning task: {task_name}")
        if task_name == 'denoising': model.cls = DenoisingHead(model_config)
        elif task_name == 'fault_detection': model.cls = FaultpredHead(model_config)
        elif task_name == 'velocity_prediction':
             # VelpredHead가 필요로 하는 추가 파라미터를 model_config에 설정
             model_config.vel_size = config['model']['velocity_output_dim']
             model_config.vel_min = config['model'].get('vel_min', 1.5)
             model_config.vel_max = config['model'].get('vel_max', 4.0)
             model.cls = VelpredHead(model_config)
        else: raise ValueError(f"Unknown fine-tuning task name: {task_name}")
    else:
        raise ValueError(f"Unknown task type: {config['task_type']}")
    # --- END: 코드 수정 ---
        
    model.to(device)
    print("Model setup complete.")

    if config['training']['optimizer'].lower() == 'radam':
        from radam import RAdam
        optimizer = RAdam(model.parameters(), lr=config['training']['learning_rate'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
        
    loss_fn = get_loss_function(config['training']['loss_function'])
    print(f"Optimizer: {config['training']['optimizer']}, Loss Function: {config['training']['loss_function']}")

    trainer = Trainer(model, optimizer, loss_fn, device, config)
    trained_model, train_losses, val_losses, last_attentions = trainer.train(train_dataloader, test_dataloader)
    
    plot_loss_curve(train_losses, val_losses, os.path.join(plots_output_dir, "loss_curve.png"), run_name)

    print("Starting final evaluation on test set...")
    evaluate_model(trained_model, test_dataloader, device, config, plots_output_dir)
    
    print("Experiment finished successfully!")


def evaluate_model(model, test_dataloader, device, config, output_dir):
    # (이하 평가 함수는 이전과 동일)
    # ...
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main experiment runner for Seismic BERT project.")
    parser.add_argument("--config", type=str, required=True, help="Path to the main YAML configuration file for the desired experiment.")
    args = parser.parse_args()
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config_params = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML configuration from {args.config}: {e}")
        exit(1)
    run_experiment(config_params)