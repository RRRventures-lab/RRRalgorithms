from datetime import datetime
from optuna import Trial
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler, CmaEsSampler
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional, Callable, Any
import json
import numpy as np
import optuna
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Hyperparameter Tuning Framework for Neural Network Models

This module implements automated hyperparameter optimization using Optuna,
with support for distributed tuning, early stopping, and multi-objective optimization.

Features:
- Bayesian optimization with Tree-structured Parzen Estimator (TPE)
- Pruning of unpromising trials
- Multi-objective optimization (accuracy vs latency)
- Distributed tuning support
- Automatic model selection

Author: AI System
Date: 2025-10-12
"""



class HyperparameterOptimizer:
    """
    Main class for hyperparameter optimization of neural network models.
    """
    
    def __init__(
        self,
        model_class: type,
        train_dataset: Dataset,
        val_dataset: Dataset,
        objective_metric: str = 'accuracy',
        direction: str = 'maximize',
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        study_name: Optional[str] = None,
        storage: Optional[str] = None
    ):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            model_class: Class of the model to optimize
            train_dataset: Training dataset
            val_dataset: Validation dataset
            objective_metric: Metric to optimize ('accuracy', 'loss', 'sharpe', etc.)
            direction: 'maximize' or 'minimize'
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            n_jobs: Number of parallel jobs
            study_name: Name for the study
            storage: Database URL for distributed optimization
        """
        self.model_class = model_class
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.objective_metric = objective_metric
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        
        # Create study name if not provided
        if study_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            study_name = f"{model_class.__name__}_{timestamp}"
        self.study_name = study_name
        self.storage = storage
        
        # Best model tracking
        self.best_model = None
        self.best_params = None
        self.best_value = None
    
    def create_search_space(self, trial: Trial) -> Dict[str, Any]:
        """
        Define the hyperparameter search space.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            Dictionary of hyperparameters
        """
        # Model architecture parameters
        params = {
            'd_model': trial.suggest_categorical('d_model', [128, 256, 512]),
            'n_heads': trial.suggest_categorical('n_heads', [4, 8, 16]),
            'n_layers': trial.suggest_int('n_layers', 2, 12),
            'dim_feedforward': trial.suggest_int('dim_feedforward', 256, 2048, step=256),
            
            # Regularization
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'drop_path': trial.suggest_float('drop_path', 0.0, 0.3),
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-5, 1e-1),
            
            # Training parameters
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'warmup_ratio': trial.suggest_float('warmup_ratio', 0.0, 0.2),
            
            # Optimizer
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'lamb']),
            'scheduler': trial.suggest_categorical('scheduler', ['cosine', 'linear', 'exponential']),
            
            # Loss function
            'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.2),
            'focal_loss': trial.suggest_categorical('focal_loss', [True, False]),
        }
        
        # Conditional parameters
        if params['focal_loss']:
            params['focal_gamma'] = trial.suggest_float('focal_gamma', 1.0, 3.0)
        
        return params
    
    def create_model(self, params: Dict[str, Any]) -> nn.Module:
        """
        Create model with given hyperparameters.
        
        Args:
            params: Hyperparameter dictionary
        
        Returns:
            Initialized model
        """
        # Extract model-specific parameters
        model_params = {
            'd_model': params['d_model'],
            'n_heads': params['n_heads'],
            'n_layers': params['n_layers'],
            'dim_feedforward': params['dim_feedforward'],
            'dropout': params['dropout']
        }
        
        # Add any additional model-specific parameters
        if hasattr(self.model_class, '__init__'):
            import inspect
            sig = inspect.signature(self.model_class.__init__)
            for param_name in sig.parameters:
                if param_name in params and param_name not in model_params:
                    model_params[param_name] = params[param_name]
        
        model = self.model_class(**model_params)
        return model
    
    def create_optimizer(
        self,
        model: nn.Module,
        params: Dict[str, Any]
    ) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
        """
        Create optimizer and scheduler.
        
        Args:
            model: Model to optimize
            params: Hyperparameter dictionary
        
        Returns:
            Optimizer and scheduler
        """
        # Select optimizer
        if params['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
        elif params['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
        elif params['optimizer'] == 'lamb':
            # Use LAMB optimizer if available
            try:
                from apex.optimizers import FusedLAMB
                optimizer = FusedLAMB(
                    model.parameters(),
                    lr=params['learning_rate'],
                    weight_decay=params['weight_decay']
                )
            except ImportError:
                # Fallback to AdamW
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=params['learning_rate'],
                    weight_decay=params['weight_decay']
                )
        
        # Select scheduler
        total_steps = len(self.train_dataset) // params['batch_size'] * 10  # 10 epochs
        warmup_steps = int(total_steps * params['warmup_ratio'])
        
        if params['scheduler'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps
            )
        elif params['scheduler'] == 'linear':
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=warmup_steps
            )
        elif params['scheduler'] == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=0.95
            )
        else:
            scheduler = None
        
        return optimizer, scheduler
    
    def train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        params: Dict[str, Any],
        device: str = 'cpu'
    ) -> float:
        """
        Train model for one epoch.
        
        Args:
            model: Model to train
            dataloader: Training dataloader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            params: Hyperparameters
            device: Device to train on
        
        Returns:
            Average training loss
        """
        model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            
            # Calculate loss (assuming multi-task for price prediction)
            loss = 0.0
            for horizon in ['5min', '15min', '1hr']:
                if horizon in outputs and horizon in target:
                    if params.get('focal_loss', False):
                        # Focal loss
                        ce_loss = F.cross_entropy(
                            outputs[horizon]['logits'],
                            target[horizon],
                            reduction='none',
                            label_smoothing=params.get('label_smoothing', 0.0)
                        )
                        pt = torch.exp(-ce_loss)
                        focal_loss = ((1 - pt) ** params.get('focal_gamma', 2.0)) * ce_loss
                        loss += focal_loss.mean()
                    else:
                        # Standard cross-entropy
                        loss += F.cross_entropy(
                            outputs[horizon]['logits'],
                            target[horizon],
                            label_smoothing=params.get('label_smoothing', 0.0)
                        )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str = 'cpu'
    ) -> Dict[str, float]:
        """
        Evaluate model on validation data.
        
        Args:
            model: Model to evaluate
            dataloader: Validation dataloader
            device: Device to evaluate on
        
        Returns:
            Dictionary of metrics
        """
        model.eval()
        
        total_loss = 0.0
        correct_predictions = {horizon: 0 for horizon in ['5min', '15min', '1hr']}
        total_predictions = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                
                outputs = model(data)
                
                # Calculate loss and accuracy
                for horizon in ['5min', '15min', '1hr']:
                    if horizon in outputs and horizon in target:
                        loss = F.cross_entropy(outputs[horizon]['logits'], target[horizon])
                        total_loss += loss.item()
                        
                        predictions = torch.argmax(outputs[horizon]['probs'], dim=-1)
                        correct_predictions[horizon] += (predictions == target[horizon]).sum().item()
                
                total_predictions += data.size(0)
        
        # Calculate metrics
        metrics = {
            'loss': total_loss / (len(dataloader) * 3),  # Average across horizons
            'accuracy': sum(correct_predictions.values()) / (total_predictions * 3)
        }
        
        # Add per-horizon accuracy
        for horizon in ['5min', '15min', '1hr']:
            metrics[f'accuracy_{horizon}'] = correct_predictions[horizon] / total_predictions
        
        return metrics
    
    def objective(self, trial: Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial
        
        Returns:
            Objective value
        """
        # Sample hyperparameters
        params = self.create_search_space(trial)
        
        # Create model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = self.create_model(params)
        model = model.to(device)
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=params['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=params['batch_size'],
            shuffle=False
        )
        
        # Create optimizer and scheduler
        optimizer, scheduler = self.create_optimizer(model, params)
        
        # Training loop with early stopping
        best_val_metric = float('-inf') if self.direction == 'maximize' else float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(10):  # Train for up to 10 epochs
            # Train
            train_loss = self.train_epoch(
                model, train_loader, optimizer, scheduler, params, device
            )
            
            # Evaluate
            val_metrics = self.evaluate(model, val_loader, device)
            
            # Get objective metric
            if self.objective_metric in val_metrics:
                val_metric = val_metrics[self.objective_metric]
            else:
                val_metric = val_metrics['accuracy']
            
            # Report intermediate value for pruning
            trial.report(val_metric, epoch)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Early stopping
            if self.direction == 'maximize':
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                if val_metric < best_val_metric:
                    best_val_metric = val_metric
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        # Save best model if this trial is the best so far
        if self.best_value is None or \
           (self.direction == 'maximize' and best_val_metric > self.best_value) or \
           (self.direction == 'minimize' and best_val_metric < self.best_value):
            self.best_value = best_val_metric
            self.best_params = params
            self.best_model = model.state_dict()
        
        return best_val_metric
    
    def optimize(self) -> optuna.Study:
        """
        Run hyperparameter optimization.
        
        Returns:
            Optuna study object
        """
        # Create sampler
        sampler = TPESampler(seed=42)
        
        # Create pruner
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        
        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
            storage=self.storage,
            load_if_exists=True
        )
        
        # Optimize
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )
        
        return study
    
    def save_results(self, study: optuna.Study, save_dir: str):
        """
        Save optimization results.
        
        Args:
            study: Completed study
            save_dir: Directory to save results
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save best parameters
        with open(save_path / 'best_params.json', 'w') as f:
            json.dump(study.best_params, f, indent=2)
        
        # Save best model
        if self.best_model is not None:
            torch.save(self.best_model, save_path / 'best_model.pt')
        
        # Save study
        with open(save_path / 'study.pkl', 'wb') as f:
            pickle.dump(study, f)
        
        # Save optimization history
        df = study.trials_dataframe()
        df.to_csv(save_path / 'optimization_history.csv', index=False)
        
        # Generate report
        self.generate_report(study, save_path)
        
        print(f"Results saved to {save_path}")
    
    def generate_report(self, study: optuna.Study, save_path: Path):
        """
        Generate optimization report.
        
        Args:
            study: Completed study
            save_path: Path to save report
        """
        report = []
        report.append("=" * 50)
        report.append("HYPERPARAMETER OPTIMIZATION REPORT")
        report.append("=" * 50)
        report.append(f"Study Name: {self.study_name}")
        report.append(f"Optimization Direction: {self.direction}")
        report.append(f"Objective Metric: {self.objective_metric}")
        report.append(f"Number of Trials: {len(study.trials)}")
        report.append(f"Number of Completed Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        report.append("")
        
        report.append("BEST TRIAL:")
        report.append("-" * 30)
        report.append(f"Value: {study.best_value:.4f}")
        report.append(f"Trial Number: {study.best_trial.number}")
        report.append("")
        
        report.append("BEST PARAMETERS:")
        report.append("-" * 30)
        for key, value in study.best_params.items():
            report.append(f"  {key}: {value}")
        report.append("")
        
        report.append("PARAMETER IMPORTANCE:")
        report.append("-" * 30)
        try:
            importance = optuna.importance.get_param_importances(study)
            for key, value in importance.items():
                report.append(f"  {key}: {value:.4f}")
        except:
            report.append("  Could not calculate parameter importance")
        report.append("")
        
        report.append("TOP 5 TRIALS:")
        report.append("-" * 30)
        df = study.trials_dataframe()
        if not df.empty:
            df_sorted = df.sort_values('value', ascending=(self.direction == 'minimize'))
            top_5 = df_sorted.head(5)[['number', 'value', 'state']]
            report.append(top_5.to_string())
        
        # Save report
        with open(save_path / 'report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print('\n'.join(report))


class MultiObjectiveOptimizer(HyperparameterOptimizer):
    """
    Multi-objective hyperparameter optimization (e.g., accuracy vs latency).
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.objectives = ['accuracy', 'latency']  # Multiple objectives
    
    def measure_latency(self, model: nn.Module, input_shape: Tuple[int, ...], device: str = 'cpu') -> float:
        """
        Measure model inference latency.
        
        Args:
            model: Model to measure
            input_shape: Input tensor shape
            device: Device to measure on
        
        Returns:
            Average latency in milliseconds
        """
        model.eval()
        model = model.to(device)
        
        # Warm up
        dummy_input = torch.randn(input_shape).to(device)
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Measure
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(dummy_input)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        return np.mean(latencies)
    
    def multi_objective(self, trial: Trial) -> Tuple[float, float]:
        """
        Multi-objective function.
        
        Args:
            trial: Optuna trial
        
        Returns:
            Tuple of objective values (accuracy, latency)
        """
        # Get hyperparameters
        params = self.create_search_space(trial)
        
        # Create and train model (simplified)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = self.create_model(params)
        model = model.to(device)
        
        # Train briefly
        train_loader = DataLoader(self.train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=params['batch_size'])
        
        optimizer, scheduler = self.create_optimizer(model, params)
        
        # Quick training
        for epoch in range(3):
            self.train_epoch(model, train_loader, optimizer, scheduler, params, device)
        
        # Evaluate accuracy
        val_metrics = self.evaluate(model, val_loader, device)
        accuracy = val_metrics['accuracy']
        
        # Measure latency
        input_shape = (1, 100, 6)  # Example shape
        latency = self.measure_latency(model, input_shape, device)
        
        return accuracy, latency
    
    def optimize_multi_objective(self) -> optuna.Study:
        """
        Run multi-objective optimization.
        
        Returns:
            Multi-objective study
        """
        # Create multi-objective study
        study = optuna.create_study(
            study_name=f"{self.study_name}_multi",
            directions=['maximize', 'minimize'],  # Max accuracy, min latency
            sampler=TPESampler(seed=42),
            storage=self.storage,
            load_if_exists=True
        )
        
        # Optimize
        study.optimize(
            self.multi_objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        return study


if __name__ == "__main__":
    # Example usage
    print("Testing Hyperparameter Optimization Framework...")
    
    # Create dummy dataset
    class DummyDataset(Dataset):
        def __init__(self, size: int = 1000):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # Random data
            x = torch.randn(100, 6)  # [seq_len, features]
            y = {
                '5min': torch.randint(0, 3, (1,)).item(),
                '15min': torch.randint(0, 3, (1,)).item(),
                '1hr': torch.randint(0, 3, (1,)).item()
            }
            return x, y
    
    # Create dummy model class
    class DummyModel(nn.Module):
        def __init__(self, d_model=128, n_heads=4, n_layers=3, dim_feedforward=512, dropout=0.1, **kwargs):
            super().__init__()
            self.fc = nn.Linear(6, d_model)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout),
                n_layers
            )
            self.heads = nn.ModuleDict({
                '5min': nn.Linear(d_model, 3),
                '15min': nn.Linear(d_model, 3),
                '1hr': nn.Linear(d_model, 3)
            })
        
        def forward(self, x):
            x = self.fc(x)
            x = self.transformer(x)
            x = x.mean(dim=1)  # Global pooling
            
            outputs = {}
            for horizon in ['5min', '15min', '1hr']:
                logits = self.heads[horizon](x)
                outputs[horizon] = {
                    'logits': logits,
                    'probs': F.softmax(logits, dim=-1)
                }
            return outputs
    
    # Create datasets
    train_dataset = DummyDataset(1000)
    val_dataset = DummyDataset(200)
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(
        model_class=DummyModel,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        objective_metric='accuracy',
        direction='maximize',
        n_trials=5,  # Few trials for testing
        study_name='test_optimization'
    )
    
    # Run optimization
    print("\nRunning hyperparameter optimization...")
    study = optimizer.optimize()
    
    # Print results
    print(f"\nBest trial:")
    print(f"  Value: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")
    
    # Save results
    optimizer.save_results(study, 'optimization_results')
    
    print("\nHyperparameter optimization test completed!")