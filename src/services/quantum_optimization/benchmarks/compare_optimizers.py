from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Callable
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import time

"""
Optimization Benchmarks and Comparisons

This module provides comprehensive benchmarking tools to compare
quantum-inspired optimizers against classical methods across multiple
problem types and metrics.

Metrics tracked:
- Solution quality (objective value)
- Execution time
- Convergence rate
- Scalability

Author: Quantum Optimization Team
Date: 2025-10-11
"""


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    optimizer_name: str
    problem_name: str
    problem_size: int
    best_score: float
    execution_time: float
    n_evaluations: int
    converged: bool
    convergence_iteration: int = -1


@dataclass
class BenchmarkSummary:
    """Summary of all benchmark results"""
    results: List[BenchmarkResult]
    winner_by_quality: Dict[str, str]
    winner_by_speed: Dict[str, str]
    average_speedup: float
    average_quality_improvement: float


class OptimizerBenchmark:
    """
    Comprehensive benchmark suite for optimization algorithms
    """

    def __init__(self, output_dir: str = "benchmarks/results"):
        """
        Initialize benchmark suite

        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def benchmark_portfolio_optimization(self,
                                        problem_sizes: List[int] = [5, 10, 20]) -> List[BenchmarkResult]:
        """
        Benchmark portfolio optimization algorithms

        Args:
            problem_sizes: List of portfolio sizes (number of assets)

        Returns:
            List of benchmark results
        """
        from ..portfolio import QAOAPortfolioOptimizer, ClassicalMarkowitzOptimizer, PortfolioConstraints

        logger.info("\n" + "="*60)
        logger.info("BENCHMARKING PORTFOLIO OPTIMIZATION")
        logger.info("="*60)

        results = []

        for n_assets in problem_sizes:
            logger.info(f"\nProblem size: {n_assets} assets")

            # Generate random problem instance
            np.random.seed(42)
            expected_returns = np.random.uniform(0.05, 0.15, n_assets)
            random_matrix = np.random.randn(n_assets, n_assets)
            covariance = np.dot(random_matrix, random_matrix.T) * 0.01

            constraints = PortfolioConstraints(
                min_weight=0.0,
                max_weight=1.0,
                long_only=True,
                risk_tolerance=1.0
            )

            # Benchmark QAOA
            logger.info("  Running QAOA optimizer...")
            qaoa = QAOAPortfolioOptimizer(n_layers=3, n_iterations=50)
            start = time.time()
            qaoa_result = qaoa.optimize(expected_returns, covariance, constraints)
            qaoa_time = time.time() - start

            results.append(BenchmarkResult(
                optimizer_name="QAOA",
                problem_name="portfolio_optimization",
                problem_size=n_assets,
                best_score=qaoa_result.sharpe_ratio,
                execution_time=qaoa_time,
                n_evaluations=qaoa_result.iterations,
                converged=qaoa_result.converged
            ))

            # Benchmark Classical
            logger.info("  Running Classical optimizer...")
            classical = ClassicalMarkowitzOptimizer()
            start = time.time()
            classical_result = classical.optimize(expected_returns, covariance, constraints)
            classical_time = time.time() - start

            results.append(BenchmarkResult(
                optimizer_name="Classical_Markowitz",
                problem_name="portfolio_optimization",
                problem_size=n_assets,
                best_score=classical_result.sharpe_ratio,
                execution_time=classical_time,
                n_evaluations=classical_result.iterations,
                converged=classical_result.converged
            ))

            # Compare
            speedup = classical_time / qaoa_time if qaoa_time > 0 else 0
            quality_ratio = qaoa_result.sharpe_ratio / classical_result.sharpe_ratio if classical_result.sharpe_ratio != 0 else 1

            logger.info(f"  QAOA Sharpe: {qaoa_result.sharpe_ratio:.4f}, Time: {qaoa_time:.2f}s")
            logger.info(f"  Classical Sharpe: {classical_result.sharpe_ratio:.4f}, Time: {classical_time:.2f}s")
            logger.info(f"  Speedup: {speedup:.2f}x, Quality ratio: {quality_ratio:.2f}")

        self.results.extend(results)
        return results

    def benchmark_hyperparameter_tuning(self,
                                       problem_sizes: List[int] = [3, 5, 8]) -> List[BenchmarkResult]:
        """
        Benchmark hyperparameter tuning algorithms

        Args:
            problem_sizes: List of hyperparameter space sizes

        Returns:
            List of benchmark results
        """
        from ..hyperparameter import QuantumAnnealingTuner, GridSearchTuner, HyperparameterSpace

        logger.info("\n" + "="*60)
        logger.info("BENCHMARKING HYPERPARAMETER TUNING")
        logger.info("="*60)

        results = []

        # Define test function (Rastrigin - challenging optimization landscape)
        def rastrigin(params):
            """Rastrigin function (many local minima)"""
            A = 10
            n = len(params)
            values = [params[f'x{i}'] for i in range(n)]
            return -(A * n + sum(x**2 - A * np.cos(2 * np.pi * x) for x in values))

        for n_params in problem_sizes:
            logger.info(f"\nProblem size: {n_params} parameters")

            # Define search space
            param_spaces = [
                HyperparameterSpace(f'x{i}', 'continuous', bounds=(-5.12, 5.12))
                for i in range(n_params)
            ]

            # Benchmark Quantum Annealing
            logger.info("  Running Quantum Annealing tuner...")
            qa_tuner = QuantumAnnealingTuner(
                param_spaces=param_spaces,
                n_qubits=10,
                n_iterations=30,
                n_parallel=2
            )
            start = time.time()
            qa_result = qa_tuner.tune(rastrigin, maximize=False)
            qa_time = time.time() - start

            results.append(BenchmarkResult(
                optimizer_name="Quantum_Annealing",
                problem_name="hyperparameter_tuning",
                problem_size=n_params,
                best_score=-qa_result.best_score,  # Convert back to original scale
                execution_time=qa_time,
                n_evaluations=qa_result.n_evaluations,
                converged=True
            ))

            # Benchmark Grid Search
            logger.info("  Running Grid Search...")
            grid_tuner = GridSearchTuner(param_spaces, n_points=5)
            start = time.time()
            grid_result = grid_tuner.tune(rastrigin, maximize=False)
            grid_time = time.time() - start

            results.append(BenchmarkResult(
                optimizer_name="Grid_Search",
                problem_name="hyperparameter_tuning",
                problem_size=n_params,
                best_score=-grid_result.best_score,
                execution_time=grid_time,
                n_evaluations=grid_result.n_evaluations,
                converged=True
            ))

            # Compare
            speedup = grid_time / qa_time if qa_time > 0 else 0
            quality_improvement = (grid_result.best_score - qa_result.best_score) / abs(grid_result.best_score) * 100

            logger.info(f"  QA score: {-qa_result.best_score:.4f}, Time: {qa_time:.2f}s, Evals: {qa_result.n_evaluations}")
            logger.info(f"  Grid score: {-grid_result.best_score:.4f}, Time: {grid_time:.2f}s, Evals: {grid_result.n_evaluations}")
            logger.info(f"  Speedup: {speedup:.2f}x, Quality improvement: {quality_improvement:.2f}%")

        self.results.extend(results)
        return results

    def benchmark_feature_selection(self,
                                    problem_sizes: List[Tuple[int, int]] = [(100, 20), (200, 50)]) -> List[BenchmarkResult]:
        """
        Benchmark feature selection algorithms

        Args:
            problem_sizes: List of (n_samples, n_features) tuples

        Returns:
            List of benchmark results
        """
        from ..features import QuantumFeatureSelector, ClassicalFeatureSelector
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        logger.info("\n" + "="*60)
        logger.info("BENCHMARKING FEATURE SELECTION")
        logger.info("="*60)

        results = []

        for n_samples, n_features in problem_sizes:
            logger.info(f"\nProblem size: {n_samples} samples, {n_features} features")

            # Generate synthetic classification problem
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_features // 2,
                n_redundant=n_features // 4,
                random_state=42
            )

            n_features_to_select = n_features // 2

            # Benchmark Quantum Feature Selector
            logger.info("  Running Quantum Feature Selector...")
            quantum_selector = QuantumFeatureSelector(
                n_features_to_select=n_features_to_select,
                n_iterations=20,
                population_size=15
            )
            start = time.time()
            quantum_selector.fit(X, y, RandomForestClassifier())
            quantum_time = time.time() - start

            # Evaluate quality
            X_quantum = quantum_selector.transform(X)
            quantum_score = np.mean(cross_val_score(
                RandomForestClassifier(random_state=42),
                X_quantum, y, cv=3
            ))

            results.append(BenchmarkResult(
                optimizer_name="Quantum_Feature_Selection",
                problem_name="feature_selection",
                problem_size=n_features,
                best_score=quantum_score,
                execution_time=quantum_time,
                n_evaluations=quantum_selector.n_evaluations_,
                converged=True
            ))

            # Benchmark Classical Feature Selector
            logger.info("  Running Classical Feature Selector...")
            classical_selector = ClassicalFeatureSelector(n_features_to_select=n_features_to_select)
            start = time.time()
            classical_selector.fit(X, y)
            classical_time = time.time() - start

            X_classical = classical_selector.transform(X)
            classical_score = np.mean(cross_val_score(
                RandomForestClassifier(random_state=42),
                X_classical, y, cv=3
            ))

            results.append(BenchmarkResult(
                optimizer_name="Classical_Feature_Selection",
                problem_name="feature_selection",
                problem_size=n_features,
                best_score=classical_score,
                execution_time=classical_time,
                n_evaluations=n_features,  # Approximation
                converged=True
            ))

            # Compare
            speedup = classical_time / quantum_time if quantum_time > 0 else 0
            score_improvement = (quantum_score - classical_score) / classical_score * 100

            logger.info(f"  Quantum score: {quantum_score:.4f}, Time: {quantum_time:.2f}s")
            logger.info(f"  Classical score: {classical_score:.4f}, Time: {classical_time:.2f}s")
            logger.info(f"  Speedup: {speedup:.2f}x, Score improvement: {score_improvement:.2f}%")

        self.results.extend(results)
        return results

    def run_full_benchmark_suite(self) -> BenchmarkSummary:
        """
        Run complete benchmark suite

        Returns:
            BenchmarkSummary with all results
        """
        logger.info("\n" + "="*60)
        logger.info("RUNNING FULL BENCHMARK SUITE")
        logger.info("="*60)

        # Run all benchmarks
        self.benchmark_portfolio_optimization()
        self.benchmark_hyperparameter_tuning()
        self.benchmark_feature_selection()

        # Analyze results
        summary = self._analyze_results()

        # Save results
        self._save_results(summary)

        # Generate visualizations
        self._plot_results()

        return summary

    def _analyze_results(self) -> BenchmarkSummary:
        """
        Analyze benchmark results and generate summary

        Returns:
            BenchmarkSummary
        """
        logger.info("\n" + "="*60)
        logger.info("BENCHMARK SUMMARY")
        logger.info("="*60)

        # Group results by problem
        problems = set(r.problem_name for r in self.results)

        winner_by_quality = {}
        winner_by_speed = {}

        for problem in problems:
            problem_results = [r for r in self.results if r.problem_name == problem]

            # Group by problem size
            sizes = set(r.problem_size for r in problem_results)

            for size in sizes:
                size_results = [r for r in problem_results if r.problem_size == size]

                # Winner by quality
                best_quality = max(size_results, key=lambda r: r.best_score)
                key = f"{problem}_{size}"
                winner_by_quality[key] = best_quality.optimizer_name

                # Winner by speed
                fastest = min(size_results, key=lambda r: r.execution_time)
                winner_by_speed[key] = fastest.optimizer_name

                logger.info(f"\n{problem} (size {size}):")
                for result in size_results:
                    logger.info(f"  {result.optimizer_name}:")
                    logger.info(f"    Score: {result.best_score:.4f}")
                    logger.info(f"    Time: {result.execution_time:.2f}s")
                    logger.info(f"    Evaluations: {result.n_evaluations}")

        # Calculate average improvements
        quantum_results = [r for r in self.results if 'Quantum' in r.optimizer_name or 'QAOA' in r.optimizer_name]
        classical_results = [r for r in self.results if 'Classical' in r.optimizer_name or 'Grid' in r.optimizer_name]

        avg_quantum_time = np.mean([r.execution_time for r in quantum_results])
        avg_classical_time = np.mean([r.execution_time for r in classical_results])
        avg_speedup = avg_classical_time / avg_quantum_time if avg_quantum_time > 0 else 0

        logger.info(f"\nOverall Statistics:")
        logger.info(f"  Average Quantum Time: {avg_quantum_time:.2f}s")
        logger.info(f"  Average Classical Time: {avg_classical_time:.2f}s")
        logger.info(f"  Average Speedup: {avg_speedup:.2f}x")

        return BenchmarkSummary(
            results=self.results,
            winner_by_quality=winner_by_quality,
            winner_by_speed=winner_by_speed,
            average_speedup=avg_speedup,
            average_quality_improvement=0.0  # Calculate if needed
        )

    def _save_results(self, summary: BenchmarkSummary):
        """
        Save benchmark results to file

        Args:
            summary: BenchmarkSummary to save
        """
        # Save detailed results as JSON
        results_file = self.output_dir / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'results': [asdict(r) for r in self.results],
                'summary': {
                    'winner_by_quality': summary.winner_by_quality,
                    'winner_by_speed': summary.winner_by_speed,
                    'average_speedup': summary.average_speedup,
                    'average_quality_improvement': summary.average_quality_improvement
                }
            }, f, indent=2)

        logger.info(f"\nResults saved to: {results_file}")

        # Save summary as text
        summary_file = self.output_dir / "benchmark_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("QUANTUM OPTIMIZATION BENCHMARK SUMMARY\n")
            f.write("="*60 + "\n\n")

            f.write(f"Average Speedup: {summary.average_speedup:.2f}x\n\n")

            f.write("Winners by Quality:\n")
            for problem, winner in summary.winner_by_quality.items():
                f.write(f"  {problem}: {winner}\n")

            f.write("\nWinners by Speed:\n")
            for problem, winner in summary.winner_by_speed.items():
                f.write(f"  {problem}: {winner}\n")

        logger.info(f"Summary saved to: {summary_file}")

    def _plot_results(self):
        """Generate visualization plots of benchmark results"""
        try:
            # Group results by problem type
            problems = set(r.problem_name for r in self.results)

            for problem in problems:
                problem_results = [r for r in self.results if r.problem_name == problem]

                # Plot execution time comparison
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                optimizers = list(set(r.optimizer_name for r in problem_results))
                sizes = sorted(set(r.problem_size for r in problem_results))

                for optimizer in optimizers:
                    optimizer_results = [r for r in problem_results if r.optimizer_name == optimizer]
                    times = [r.execution_time for r in sorted(optimizer_results, key=lambda x: x.problem_size)]
                    scores = [r.best_score for r in sorted(optimizer_results, key=lambda x: x.problem_size)]

                    ax1.plot(sizes, times, marker='o', label=optimizer)
                    ax2.plot(sizes, scores, marker='s', label=optimizer)

                ax1.set_xlabel('Problem Size')
                ax1.set_ylabel('Execution Time (s)')
                ax1.set_title(f'{problem.replace("_", " ").title()}: Execution Time')
                ax1.legend()
                ax1.grid(True)

                ax2.set_xlabel('Problem Size')
                ax2.set_ylabel('Score')
                ax2.set_title(f'{problem.replace("_", " ").title()}: Solution Quality')
                ax2.legend()
                ax2.grid(True)

                plt.tight_layout()
                plot_file = self.output_dir / f"benchmark_{problem}.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to: {plot_file}")
                plt.close()

        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")


if __name__ == "__main__":
    # Run full benchmark suite
    benchmark = OptimizerBenchmark(output_dir="benchmarks/results")
    summary = benchmark.run_full_benchmark_suite()

    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    print(f"\nAverage speedup (Quantum vs Classical): {summary.average_speedup:.2f}x")
    print(f"\nResults saved to: benchmarks/results/")
