from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import time
import torch
import torch.nn as nn
import tracemalloc
import warnings

"""
Comprehensive Benchmarking Suite for Neural Network Models

This module provides tools for benchmarking model performance across multiple dimensions:
- Inference latency
- Memory usage
- Accuracy metrics
- Training speed
- Model size
- Power efficiency (when available)

Author: AI System
Date: 2025-10-12
"""



@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_name: str
    timestamp: str
    
    # Performance metrics
    inference_latency_ms: float
    inference_latency_std: float
    throughput_samples_per_sec: float
    
    # Memory metrics
    model_size_mb: float
    peak_memory_mb: float
    memory_allocated_mb: float
    
    # Accuracy metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Training metrics
    training_time_per_epoch: float
    convergence_epochs: int
    
    # Hardware info
    device: str
    device_name: str
    cpu_count: int
    
    # Additional metrics
    flops: Optional[int] = None
    parameters: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.__dict__
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class ModelBenchmark:
    """
    Comprehensive model benchmarking suite.
    """
    
    def __init__(
        self,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        warmup_iterations: int = 10,
        benchmark_iterations: int = 100,
        batch_size: int = 1
    ):
        """
        Initialize benchmark suite.
        
        Args:
            device: Device to run benchmarks on
            warmup_iterations: Number of warmup iterations
            benchmark_iterations: Number of benchmark iterations
            batch_size: Batch size for benchmarking
        """
        self.device = device
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.batch_size = batch_size
        
        # Get device information
        if device == 'cuda':
            self.device_name = torch.cuda.get_device_name(0)
        else:
            self.device_name = "CPU"
        
        self.cpu_count = psutil.cpu_count()
    
    def measure_inference_latency(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        return_all: bool = False
    ) -> Tuple[float, float, Optional[List[float]]]:
        """
        Measure inference latency.
        
        Args:
            model: Model to benchmark
            input_shape: Shape of input tensor
            return_all: Whether to return all measurements
        
        Returns:
            Mean latency, standard deviation, and optionally all measurements
        """
        model.eval()
        model = model.to(self.device)
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape).to(self.device)
        
        # Warmup
        for _ in range(self.warmup_iterations):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Synchronize if using CUDA
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        for _ in range(self.benchmark_iterations):
            start = time.perf_counter()
            
            with torch.no_grad():
                _ = model(dummy_input)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to milliseconds
        
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        
        if return_all:
            return mean_latency, std_latency, latencies
        return mean_latency, std_latency, None
    
    def measure_throughput(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        duration_seconds: int = 10
    ) -> float:
        """
        Measure model throughput.
        
        Args:
            model: Model to benchmark
            input_shape: Shape of input tensor
            duration_seconds: Duration of throughput test
        
        Returns:
            Throughput in samples per second
        """
        model.eval()
        model = model.to(self.device)
        
        # Create dummy input batch
        batch_input_shape = (self.batch_size,) + input_shape[1:]
        dummy_input = torch.randn(*batch_input_shape).to(self.device)
        
        # Warmup
        for _ in range(self.warmup_iterations):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Benchmark
        start_time = time.time()
        total_samples = 0
        
        while time.time() - start_time < duration_seconds:
            with torch.no_grad():
                _ = model(dummy_input)
            total_samples += self.batch_size
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
        
        elapsed_time = time.time() - start_time
        throughput = total_samples / elapsed_time
        
        return throughput
    
    def measure_memory_usage(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...]
    ) -> Dict[str, float]:
        """
        Measure memory usage.
        
        Args:
            model: Model to benchmark
            input_shape: Shape of input tensor
        
        Returns:
            Dictionary with memory metrics
        """
        model = model.to(self.device)
        
        # Model size
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / 1024 / 1024
        
        # Memory usage during inference
        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            # Initial memory
            initial_memory = torch.cuda.memory_allocated() / 1024 / 1024
            
            # Run inference
            dummy_input = torch.randn(*input_shape).to(self.device)
            with torch.no_grad():
                _ = model(dummy_input)
            
            # Peak memory
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            current_memory = torch.cuda.memory_allocated() / 1024 / 1024
            
        else:
            # CPU memory tracking
            tracemalloc.start()
            
            dummy_input = torch.randn(*input_shape)
            with torch.no_grad():
                _ = model(dummy_input)
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            peak_memory = peak / 1024 / 1024
            current_memory = current / 1024 / 1024
            initial_memory = 0
        
        return {
            'model_size_mb': model_size_mb,
            'peak_memory_mb': peak_memory,
            'memory_allocated_mb': current_memory - initial_memory
        }
    
    def measure_flops(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...]
    ) -> Optional[int]:
        """
        Measure FLOPs (Floating Point Operations).
        
        Args:
            model: Model to benchmark
            input_shape: Shape of input tensor
        
        Returns:
            Number of FLOPs or None if unable to calculate
        """
        try:
            from thop import profile
            
            model = model.to(self.device)
            dummy_input = torch.randn(*input_shape).to(self.device)
            
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            return int(flops)
        except ImportError:
            warnings.warn("thop not installed. Cannot measure FLOPs.")
            return None
        except Exception as e:
            warnings.warn(f"Error measuring FLOPs: {e}")
            return None
    
    def measure_accuracy(
        self,
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        horizons: List[str] = ['5min', '15min', '1hr']
    ) -> Dict[str, float]:
        """
        Measure model accuracy metrics.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            horizons: Prediction horizons
        
        Returns:
            Dictionary of accuracy metrics
        """
        model.eval()
        model = model.to(self.device)
        
        all_predictions = {h: [] for h in horizons}
        all_targets = {h: [] for h in horizons}
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                
                outputs = model(data)
                
                for horizon in horizons:
                    if horizon in outputs:
                        preds = torch.argmax(outputs[horizon]['probs'], dim=-1)
                        all_predictions[horizon].append(preds.cpu())
                        all_targets[horizon].append(target[horizon].cpu())
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {}
        for horizon in horizons:
            if all_predictions[horizon]:
                preds = torch.cat(all_predictions[horizon]).numpy()
                targets = torch.cat(all_targets[horizon]).numpy()
                
                metrics[f'accuracy_{horizon}'] = accuracy_score(targets, preds)
                metrics[f'precision_{horizon}'] = precision_score(targets, preds, average='macro', zero_division=0)
                metrics[f'recall_{horizon}'] = recall_score(targets, preds, average='macro', zero_division=0)
                metrics[f'f1_{horizon}'] = f1_score(targets, preds, average='macro', zero_division=0)
        
        # Average metrics
        metrics['accuracy'] = np.mean([v for k, v in metrics.items() if 'accuracy' in k])
        metrics['precision'] = np.mean([v for k, v in metrics.items() if 'precision' in k])
        metrics['recall'] = np.mean([v for k, v in metrics.items() if 'recall' in k])
        metrics['f1_score'] = np.mean([v for k, v in metrics.items() if 'f1' in k])
        
        return metrics
    
    def benchmark_model(
        self,
        model: nn.Module,
        model_name: str,
        input_shape: Tuple[int, ...],
        test_loader: Optional[torch.utils.data.DataLoader] = None
    ) -> BenchmarkResult:
        """
        Run complete benchmark suite on a model.
        
        Args:
            model: Model to benchmark
            model_name: Name of the model
            input_shape: Input tensor shape
            test_loader: Optional test data loader for accuracy metrics
        
        Returns:
            BenchmarkResult object
        """
        print(f"Benchmarking {model_name}...")
        
        # Inference latency
        print("  Measuring inference latency...")
        mean_latency, std_latency, _ = self.measure_inference_latency(model, input_shape)
        
        # Throughput
        print("  Measuring throughput...")
        throughput = self.measure_throughput(model, input_shape, duration_seconds=5)
        
        # Memory usage
        print("  Measuring memory usage...")
        memory_metrics = self.measure_memory_usage(model, input_shape)
        
        # FLOPs
        print("  Calculating FLOPs...")
        flops = self.measure_flops(model, input_shape)
        
        # Parameter count
        parameters = sum(p.numel() for p in model.parameters())
        
        # Accuracy metrics (if test loader provided)
        if test_loader:
            print("  Measuring accuracy...")
            accuracy_metrics = self.measure_accuracy(model, test_loader)
        else:
            accuracy_metrics = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
        
        # Create result
        result = BenchmarkResult(
            model_name=model_name,
            timestamp=datetime.now().isoformat(),
            inference_latency_ms=mean_latency,
            inference_latency_std=std_latency,
            throughput_samples_per_sec=throughput,
            model_size_mb=memory_metrics['model_size_mb'],
            peak_memory_mb=memory_metrics['peak_memory_mb'],
            memory_allocated_mb=memory_metrics['memory_allocated_mb'],
            accuracy=accuracy_metrics['accuracy'],
            precision=accuracy_metrics['precision'],
            recall=accuracy_metrics['recall'],
            f1_score=accuracy_metrics['f1_score'],
            training_time_per_epoch=0.0,  # Would need training benchmark
            convergence_epochs=0,  # Would need training benchmark
            device=self.device,
            device_name=self.device_name,
            cpu_count=self.cpu_count,
            flops=flops,
            parameters=parameters
        )
        
        return result


class BenchmarkComparison:
    """
    Compare and visualize benchmark results across multiple models.
    """
    
    def __init__(self, results: List[BenchmarkResult]):
        """
        Initialize comparison.
        
        Args:
            results: List of benchmark results
        """
        self.results = results
        self.df = pd.DataFrame([r.to_dict() for r in results])
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate comparison report.
        
        Args:
            save_path: Optional path to save report
        
        Returns:
            Report string
        """
        report = []
        report.append("=" * 80)
        report.append("MODEL BENCHMARK COMPARISON REPORT")
        report.append("=" * 80)
        report.append(f"Number of models: {len(self.results)}")
        report.append(f"Benchmark date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Device: {self.results[0].device} ({self.results[0].device_name})")
        report.append("")
        
        # Performance summary
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 40)
        
        for result in self.results:
            report.append(f"\n{result.model_name}:")
            report.append(f"  Latency: {result.inference_latency_ms:.2f} ± {result.inference_latency_std:.2f} ms")
            report.append(f"  Throughput: {result.throughput_samples_per_sec:.1f} samples/sec")
            report.append(f"  Memory: {result.peak_memory_mb:.1f} MB peak, {result.model_size_mb:.1f} MB model")
            report.append(f"  Accuracy: {result.accuracy:.3f}")
            report.append(f"  Parameters: {result.parameters:,}")
            if result.flops:
                report.append(f"  FLOPs: {result.flops:,}")
        
        # Best models
        report.append("\n" + "=" * 40)
        report.append("BEST MODELS BY METRIC")
        report.append("-" * 40)
        
        metrics_to_optimize = {
            'inference_latency_ms': ('minimize', 'Lowest Latency'),
            'throughput_samples_per_sec': ('maximize', 'Highest Throughput'),
            'accuracy': ('maximize', 'Highest Accuracy'),
            'model_size_mb': ('minimize', 'Smallest Model'),
            'peak_memory_mb': ('minimize', 'Lowest Memory Usage')
        }
        
        for metric, (direction, label) in metrics_to_optimize.items():
            if direction == 'minimize':
                best_idx = self.df[metric].idxmin()
            else:
                best_idx = self.df[metric].idxmax()
            
            best_model = self.df.loc[best_idx, 'model_name']
            best_value = self.df.loc[best_idx, metric]
            
            if metric.endswith('_ms'):
                report.append(f"{label}: {best_model} ({best_value:.2f} ms)")
            elif metric.endswith('_mb'):
                report.append(f"{label}: {best_model} ({best_value:.1f} MB)")
            elif metric == 'throughput_samples_per_sec':
                report.append(f"{label}: {best_model} ({best_value:.1f} samples/sec)")
            else:
                report.append(f"{label}: {best_model} ({best_value:.3f})")
        
        # Efficiency metrics
        report.append("\n" + "=" * 40)
        report.append("EFFICIENCY METRICS")
        report.append("-" * 40)
        
        self.df['accuracy_per_param'] = self.df['accuracy'] / self.df['parameters'] * 1e6
        self.df['accuracy_per_mb'] = self.df['accuracy'] / self.df['model_size_mb']
        
        for _, row in self.df.iterrows():
            report.append(f"\n{row['model_name']}:")
            report.append(f"  Accuracy per million params: {row['accuracy_per_param']:.3f}")
            report.append(f"  Accuracy per MB: {row['accuracy_per_mb']:.3f}")
        
        report_str = '\n'.join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_str)
        
        return report_str
    
    def plot_comparison(self, save_path: Optional[str] = None):
        """
        Create comparison plots.
        
        Args:
            save_path: Optional path to save plots
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model Benchmark Comparison', fontsize=16)
        
        # Latency comparison
        ax = axes[0, 0]
        ax.bar(self.df['model_name'], self.df['inference_latency_ms'])
        ax.errorbar(range(len(self.df)), self.df['inference_latency_ms'], 
                   yerr=self.df['inference_latency_std'], fmt='none', color='black')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Inference Latency')
        ax.tick_params(axis='x', rotation=45)
        
        # Throughput comparison
        ax = axes[0, 1]
        ax.bar(self.df['model_name'], self.df['throughput_samples_per_sec'])
        ax.set_ylabel('Throughput (samples/sec)')
        ax.set_title('Throughput')
        ax.tick_params(axis='x', rotation=45)
        
        # Memory comparison
        ax = axes[0, 2]
        x = range(len(self.df))
        width = 0.35
        ax.bar([i - width/2 for i in x], self.df['model_size_mb'], width, label='Model Size')
        ax.bar([i + width/2 for i in x], self.df['peak_memory_mb'], width, label='Peak Memory')
        ax.set_ylabel('Memory (MB)')
        ax.set_title('Memory Usage')
        ax.set_xticks(x)
        ax.set_xticklabels(self.df['model_name'], rotation=45)
        ax.legend()
        
        # Accuracy comparison
        ax = axes[1, 0]
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        x = np.arange(len(self.df))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i * width, self.df[metric], width, label=metric.capitalize())
        
        ax.set_ylabel('Score')
        ax.set_title('Accuracy Metrics')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(self.df['model_name'], rotation=45)
        ax.legend()
        
        # Latency vs Accuracy scatter
        ax = axes[1, 1]
        ax.scatter(self.df['inference_latency_ms'], self.df['accuracy'], s=100)
        for i, txt in enumerate(self.df['model_name']):
            ax.annotate(txt, (self.df['inference_latency_ms'].iloc[i], 
                             self.df['accuracy'].iloc[i]))
        ax.set_xlabel('Latency (ms)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Latency vs Accuracy Trade-off')
        
        # Model size vs Accuracy
        ax = axes[1, 2]
        ax.scatter(self.df['model_size_mb'], self.df['accuracy'], s=100)
        for i, txt in enumerate(self.df['model_name']):
            ax.annotate(txt, (self.df['model_size_mb'].iloc[i], 
                             self.df['accuracy'].iloc[i]))
        ax.set_xlabel('Model Size (MB)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Size vs Accuracy Trade-off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        
        plt.show()
    
    def save_results(self, save_dir: str):
        """
        Save all benchmark results.
        
        Args:
            save_dir: Directory to save results
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        self.df.to_csv(save_path / 'benchmark_results.csv', index=False)
        
        # Save individual results as JSON
        for result in self.results:
            with open(save_path / f'{result.model_name}_benchmark.json', 'w') as f:
                f.write(result.to_json())
        
        # Generate and save report
        report = self.generate_report()
        with open(save_path / 'benchmark_report.txt', 'w') as f:
            f.write(report)
        
        # Save comparison plots
        self.plot_comparison(str(save_path / 'benchmark_comparison.png'))
        
        print(f"Results saved to {save_path}")


if __name__ == "__main__":
    # Example usage
    print("Testing Model Benchmark Suite...")
    
    # Create dummy models for testing
    class SmallModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(600, 128)
            self.fc2 = nn.Linear(128, 64)
            self.heads = nn.ModuleDict({
                '5min': nn.Linear(64, 3),
                '15min': nn.Linear(64, 3),
                '1hr': nn.Linear(64, 3)
            })
        
        def forward(self, x):
            x = x.reshape(x.size(0), -1)  # Flatten
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            
            outputs = {}
            for horizon in ['5min', '15min', '1hr']:
                logits = self.heads[horizon](x)
                outputs[horizon] = {
                    'logits': logits,
                    'probs': F.softmax(logits, dim=-1)
                }
            return outputs
    
    class LargeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(600, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.heads = nn.ModuleDict({
                '5min': nn.Linear(128, 3),
                '15min': nn.Linear(128, 3),
                '1hr': nn.Linear(128, 3)
            })
        
        def forward(self, x):
            x = x.reshape(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            
            outputs = {}
            for horizon in ['5min', '15min', '1hr']:
                logits = self.heads[horizon](x)
                outputs[horizon] = {
                    'logits': logits,
                    'probs': F.softmax(logits, dim=-1)
                }
            return outputs
    
    # Create benchmark suite
    benchmark = ModelBenchmark(
        device='cpu',  # Use CPU for testing
        warmup_iterations=5,
        benchmark_iterations=10,
        batch_size=4
    )
    
    # Benchmark models
    input_shape = (1, 100, 6)  # [batch, seq_len, features]
    
    small_model = SmallModel()
    large_model = LargeModel()
    
    results = []
    
    print("\nBenchmarking Small Model...")
    small_result = benchmark.benchmark_model(
        small_model,
        "SmallModel",
        input_shape
    )
    results.append(small_result)
    
    print("\nBenchmarking Large Model...")
    large_result = benchmark.benchmark_model(
        large_model,
        "LargeModel",
        input_shape
    )
    results.append(large_result)
    
    # Compare results
    print("\nGenerating comparison report...")
    comparison = BenchmarkComparison(results)
    report = comparison.generate_report()
    print(report)
    
    # Save results
    comparison.save_results('benchmark_results')
    
    print("\nBenchmark suite test completed!")