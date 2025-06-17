# main.py
import torch
import torch.optim as optim
import yaml
import argparse
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

# ❗️[변경] 임포트 구문을 더 깔끔하게 수정
from src.data_utils.prepare_data import prepare_data_for_task
from src.models import create_bert_config, get_bert_model, BertOnlyMLMHead, DenoisingHead, FaultpredHead, VelpredHead
from src.training.trainer import Trainer
from src.utils import get_loss_function
from src.evaluation import (
    plot_loss_curve,
    plot_pretrain_reconstruction_results,
    # ... (다른 평가/플롯 함수들)
)

def setup_paths(config):
    """실험 결과가 저장될 경로들을 설정하고 생성합니다."""
    run_name = config['run_name']
    base_results_dir = config['base_results_dir']
    
    run_output_dir = os.path.join(base_results_dir, run_name)
    training_output_dir = os.path.join(run_output_dir, config['training']['output_dir'])
    plots_output_dir = os.path.join(run_output_dir, "plots")
    
    os.makedirs(training_output_dir, exist_ok=True)
    os.makedirs(plots_output_dir, exist_ok=True)
    
    return training_output_dir, plots_output_dir

def setup_dataloaders(config, run_output_dir):
    """데이터를 준비하고 데이터로더를 생성합니다."""
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
    return train_dataloader, test_dataloader

def setup_model(config):
    """작업에 맞는 모델을 설정하고 생성합니다."""
    print("Setting up model...")
    model_config = create_bert_config(config['model'], config['data'])
    
    if config['task_type'] == 'pretrain':
        model = get_bert_model(model_config)
        model.cls = BertOnlyMLMHead(model_config)
    elif config['task_type'] == 'finetune':
        finetune_params = config['finetune_params']
        model = get_bert_model(model_config, checkpoint_path=finetune_params['pretrained_model_path'])
        task_name = finetune_params['task_name']
        print(f"Attaching head for fine-tuning task: {task_name}")
        if task_name == 'denoising': model.cls = DenoisingHead(model_config)
        elif task_name == 'fault_detection': model.cls = FaultpredHead(model_config)
        elif task_name == 'velocity_prediction':
            model_config.vel_size = config['model']['vel_size']
            model.cls = VelpredHead(model_config)
        else: raise ValueError(f"Unknown fine-tuning task name: {task_name}")
    else:
        raise ValueError(f"Unknown task type: {config['task_type']}")
    
    print("Model setup complete.")
    return model

def run_training_or_load(model, config, train_dl, test_dl, device, training_dir, plots_dir):
    """학습을 실행하거나 이미 학습된 모델을 로드합니다."""
    checkpoint_path = os.path.join(training_dir, "best_checkpoint.pt")
    
    if os.path.exists(checkpoint_path) and not config.get('force_retrain', False):
        print(f"Found existing trained model at {checkpoint_path}. Loading weights...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        return model
    
    print("Starting model training...")
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    loss_fn = get_loss_function(config['training']['loss_function'])
    trainer = Trainer(model, optimizer, loss_fn, device, config)
    
    trained_model, train_losses, val_losses, _ = trainer.train(train_dl, test_dl)
    
    plot_loss_curve(train_losses, val_losses, os.path.join(plots_dir, "loss_curve.png"), config['run_name'])
    return trained_model

def run_evaluation(model, test_dataloader, device, config, plots_dir):
    """최종 평가를 실행하고 결과를 저장합니다."""
    print("Starting final evaluation on test set...")
    model.eval()
    all_inputs, all_labels, all_predictions = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            batch_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            inputs_embeds = batch_on_device.get('inputs_embeds').float()
            
            task_name = config.get('finetune_params', {}).get('task_name', 'pretrain')
            
            if task_name == 'pretrain':
                predictions = model(inputs_embeds=inputs_embeds).logits
                labels = batch_on_device['labels']
            else:
                raise NotImplementedError("Fine-tuning evaluation logic is not implemented yet.")
            
            all_inputs.append(inputs_embeds.cpu())
            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())

    all_inputs = torch.cat(all_inputs)
    all_labels = torch.cat(all_labels)
    all_predictions = torch.cat(all_predictions)

    task_name = config.get('finetune_params', {}).get('task_name', 'pretrain')
    if task_name == 'pretrain':
        plot_pretrain_reconstruction_results(
            inputs_list=all_inputs,
            reconstructed_list=all_predictions,
            labels_list=all_labels,
            output_dir=plots_dir,
            run_name=config['run_name'],
            num_examples=4
        )

def main(config):
    """하나의 실험 전체를 실행하는 메인 함수"""
    
    training_output_dir, plots_output_dir = setup_paths(config)
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_dataloader, test_dataloader = setup_dataloaders(config, os.path.dirname(training_output_dir))

    model = setup_model(config)
    model.to(device)

    trained_model = run_training_or_load(model, config, train_dataloader, test_dataloader, device, training_output_dir, plots_output_dir)

    run_evaluation(trained_model, test_dataloader, device, config, plots_output_dir)
    
    print("Experiment finished successfully!")

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
        
    main(config_params)