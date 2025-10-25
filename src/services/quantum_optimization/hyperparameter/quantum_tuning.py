from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Any, Optional
import logging
import numpy as np
import time

"""
Quantum-Inspired Hyperparameter Tuning

This module implements quantum-inspired hyperparameter optimization using
simulated quantum annealing techniques. It's designed to find optimal
hyperparameters for ML models faster than traditional grid/random search.

Key features:
- Simulated quantum annealing for hyperparameter search
- Parallel exploration of parameter space
- Adaptive temperature scheduling
- Automatic early stopping

Author: Quantum Optimization Team
Date: 2025-10-11
"""


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HyperparameterSpace:
    """Defines the hyperparameter search space"""
    name: str
    param_type: str  # 'continuous', 'discrete', 'categorical'
    bounds: Optional[Tuple[float, float]] = None  # For continuous/discrete
    values: Optional[List[Any]] = None  # For categorical
    log_scale: bool = False  # Use log scale for continuous params


@dataclass
class TuningResult:
    """Results from hyperparameter tuning"""
    best_params: Dict[str, Any]
    best_score: float
    all_trials: List[Dict]
    optimization_time: float
    n_evaluations: int
    convergence_history: List[float]


class QuantumAnnealingTuner:
    """
    Quantum-inspired hyperparameter tuner using simulated quantum annealing

    This tuner uses quantum annealing principles to efficiently explore
    the hyperparameter space and find optimal configurations.
    """

    def __init__(self,
                 param_spaces: List[HyperparameterSpace],
                 n_qubits: int = 10,
                 n_iterations: int = 100,
                 initial_temperature: float = 10.0,
                 cooling_rate: float = 0.95,
                 n_parallel: int = 4):
        """
        Initialize quantum annealing tuner

        Args:
            param_spaces: List of hyperparameter spaces to search
            n_qubits: Number of quantum-inspired search agents
            n_iterations: Maximum iterations
            initial_temperature: Starting temperature for annealing
            cooling_rate: Temperature decay rate
            n_parallel: Number of parallel evaluations
        """
        self.param_spaces = param_spaces
        self.n_qubits = n_qubits
        self.n_iterations = n_iterations
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.n_parallel = n_parallel
        self.best_params = None
        self.best_score = float('-inf')
        self.convergence_history = []
        self.all_trials = []

    def _initialize_qubits(self) -> List[Dict[str, Any]]:
        """
        Initialize quantum-inspired qubits (parameter configurations)

        Returns:
            List of initial parameter configurations
        """
        qubits = []
        for _ in range(self.n_qubits):
            config = {}
            for space in self.param_spaces:
                config[space.name] = self._sample_parameter(space)
            qubits.append(config)
        return qubits

    def _sample_parameter(self, space: HyperparameterSpace) -> Any:
        """
        Sample a parameter value from its space

        Args:
            space: Hyperparameter space definition

        Returns:
            Sampled parameter value
        """
        if space.param_type == 'continuous':
            if space.log_scale:
                log_bounds = np.log10(space.bounds)
                value = 10 ** np.random.uniform(log_bounds[0], log_bounds[1])
            else:
                value = np.random.uniform(space.bounds[0], space.bounds[1])
            return value

        elif space.param_type == 'discrete':
            return np.random.randint(space.bounds[0], space.bounds[1] + 1)

        elif space.param_type == 'categorical':
            return np.random.choice(space.values)

        else:
            raise ValueError(f"Unknown parameter type: {space.param_type}")

    def _quantum_tunneling(self,
                          current_config: Dict[str, Any],
                          temperature: float) -> Dict[str, Any]:
        """
        Apply quantum tunneling to explore parameter space

        Quantum tunneling allows escape from local minima by
        making larger jumps in parameter space.

        Args:
            current_config: Current parameter configuration
            temperature: Current temperature (controls jump size)

        Returns:
            New parameter configuration
        """
        new_config = current_config.copy()

        # Quantum tunneling: larger perturbations at high temperature
        tunneling_prob = np.exp(-1 / (temperature + 1e-10))

        for space in self.param_spaces:
            if np.random.random() < tunneling_prob:
                # Large quantum jump
                new_config[space.name] = self._sample_parameter(space)
            else:
                # Small classical perturbation
                new_config[space.name] = self._perturb_parameter(
                    current_config[space.name],
                    space,
                    temperature
                )

        return new_config

    def _perturb_parameter(self,
                          current_value: Any,
                          space: HyperparameterSpace,
                          temperature: float) -> Any:
        """
        Apply small perturbation to a parameter

        Args:
            current_value: Current parameter value
            space: Parameter space definition
            temperature: Current temperature (controls perturbation size)

        Returns:
            Perturbed parameter value
        """
        if space.param_type == 'continuous':
            # Gaussian perturbation scaled by temperature
            std = temperature * (space.bounds[1] - space.bounds[0]) / 10
            perturbed = current_value + np.random.normal(0, std)
            # Clip to bounds
            perturbed = np.clip(perturbed, space.bounds[0], space.bounds[1])
            return perturbed

        elif space.param_type == 'discrete':
            # Random walk
            step = int(np.ceil(temperature))
            perturbed = current_value + np.random.randint(-step, step + 1)
            perturbed = np.clip(perturbed, space.bounds[0], space.bounds[1])
            return int(perturbed)

        elif space.param_type == 'categorical':
            # Random choice with temperature-dependent probability
            if np.random.random() < temperature / self.initial_temperature:
                return np.random.choice(space.values)
            return current_value

        return current_value

    def _acceptance_probability(self,
                               current_score: float,
                               new_score: float,
                               temperature: float) -> float:
        """
        Calculate probability of accepting new configuration

        Uses quantum-inspired acceptance criterion.

        Args:
            current_score: Current score
            new_score: New score
            temperature: Current temperature

        Returns:
            Acceptance probability
        """
        if new_score > current_score:
            return 1.0

        # Quantum-inspired acceptance: accounts for quantum fluctuations
        delta = new_score - current_score
        return np.exp(delta / (temperature + 1e-10))

    def tune(self,
            objective_function: Callable[[Dict[str, Any]], float],
            maximize: bool = True) -> TuningResult:
        """
        Tune hyperparameters using quantum annealing

        Args:
            objective_function: Function to optimize (params -> score)
            maximize: Whether to maximize (True) or minimize (False) the objective

        Returns:
            TuningResult with best parameters and history
        """
        start_time = time.time()

        # Initialize qubits (parameter configurations)
        qubits = self._initialize_qubits()
        qubit_scores = [float('-inf')] * self.n_qubits

        # Evaluate initial configurations
        logger.info("Evaluating initial configurations...")
        for i, config in enumerate(qubits):
            score = objective_function(config)
            if not maximize:
                score = -score
            qubit_scores[i] = score
            self.all_trials.append({'params': config, 'score': score})

            if score > self.best_score:
                self.best_score = score
                self.best_params = config.copy()

        self.convergence_history.append(self.best_score)

        # Quantum annealing iterations
        temperature = self.initial_temperature
        n_evaluations = self.n_qubits

        logger.info(f"Starting quantum annealing optimization...")

        for iteration in range(self.n_iterations):
            # Generate new configurations via quantum tunneling
            new_qubits = []
            for qubit in qubits:
                new_config = self._quantum_tunneling(qubit, temperature)
                new_qubits.append(new_config)

            # Evaluate new configurations (in parallel)
            new_scores = []
            with ProcessPoolExecutor(max_workers=self.n_parallel) as executor:
                futures = {executor.submit(objective_function, config): i
                          for i, config in enumerate(new_qubits)}

                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        score = future.result()
                        if not maximize:
                            score = -score
                        new_scores.append((idx, score))
                        n_evaluations += 1
                    except Exception as e:
                        logger.warning(f"Evaluation failed: {e}")
                        new_scores.append((idx, float('-inf')))

            # Update qubits based on acceptance probability
            for idx, new_score in new_scores:
                config = new_qubits[idx]
                current_score = qubit_scores[idx]

                # Accept or reject new configuration
                if np.random.random() < self._acceptance_probability(
                    current_score, new_score, temperature):
                    qubits[idx] = config
                    qubit_scores[idx] = new_score

                    self.all_trials.append({'params': config, 'score': new_score})

                    # Update global best
                    if new_score > self.best_score:
                        self.best_score = new_score
                        self.best_params = config.copy()
                        logger.info(f"Iteration {iteration}: New best score = {self.best_score:.6f}")

            # Cool down temperature
            temperature *= self.cooling_rate

            # Track convergence
            self.convergence_history.append(self.best_score)

            # Early stopping check
            if iteration > 20:
                recent_improvement = (self.convergence_history[-1] -
                                     self.convergence_history[-20])
                if abs(recent_improvement) < 1e-6:
                    logger.info(f"Converged at iteration {iteration}")
                    break

            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}/{self.n_iterations}, "
                          f"Temperature: {temperature:.4f}, "
                          f"Best score: {self.best_score:.6f}")

        optimization_time = time.time() - start_time

        logger.info(f"\nOptimization complete!")
        logger.info(f"Total time: {optimization_time:.2f}s")
        logger.info(f"Total evaluations: {n_evaluations}")
        logger.info(f"Best score: {self.best_score:.6f}")
        logger.info(f"Best parameters: {self.best_params}")

        # Convert back to original scale if minimizing
        final_best_score = self.best_score if maximize else -self.best_score

        return TuningResult(
            best_params=self.best_params,
            best_score=final_best_score,
            all_trials=self.all_trials,
            optimization_time=optimization_time,
            n_evaluations=n_evaluations,
            convergence_history=self.convergence_history
        )


class GridSearchTuner:
    """Classical grid search for comparison"""

    def __init__(self, param_spaces: List[HyperparameterSpace], n_points: int = 5):
        """
        Initialize grid search tuner

        Args:
            param_spaces: List of hyperparameter spaces
            n_points: Number of grid points per dimension
        """
        self.param_spaces = param_spaces
        self.n_points = n_points

    def _generate_grid(self) -> List[Dict[str, Any]]:
        """Generate grid of parameter configurations"""
        from itertools import product

        param_grids = []
        for space in self.param_spaces:
            if space.param_type == 'continuous':
                if space.log_scale:
                    log_bounds = np.log10(space.bounds)
                    grid = np.logspace(log_bounds[0], log_bounds[1], self.n_points)
                else:
                    grid = np.linspace(space.bounds[0], space.bounds[1], self.n_points)
                param_grids.append(list(grid))

            elif space.param_type == 'discrete':
                grid = np.linspace(space.bounds[0], space.bounds[1],
                                 min(self.n_points, space.bounds[1] - space.bounds[0] + 1))
                param_grids.append([int(x) for x in grid])

            elif space.param_type == 'categorical':
                param_grids.append(space.values)

        # Generate all combinations
        configs = []
        for values in product(*param_grids):
            config = {space.name: value
                     for space, value in zip(self.param_spaces, values)}
            configs.append(config)

        return configs

    def tune(self,
            objective_function: Callable[[Dict[str, Any]], float],
            maximize: bool = True) -> TuningResult:
        """
        Tune hyperparameters using grid search

        Args:
            objective_function: Function to optimize
            maximize: Whether to maximize the objective

        Returns:
            TuningResult
        """
        start_time = time.time()

        configs = self._generate_grid()
        logger.info(f"Grid search: evaluating {len(configs)} configurations...")

        best_score = float('-inf') if maximize else float('inf')
        best_params = None
        all_trials = []

        for i, config in enumerate(configs):
            score = objective_function(config)
            all_trials.append({'params': config, 'score': score})

            if (maximize and score > best_score) or (not maximize and score < best_score):
                best_score = score
                best_params = config

            if (i + 1) % 10 == 0:
                logger.info(f"Evaluated {i + 1}/{len(configs)} configurations")

        optimization_time = time.time() - start_time

        logger.info(f"\nGrid search complete!")
        logger.info(f"Total time: {optimization_time:.2f}s")
        logger.info(f"Best score: {best_score:.6f}")
        logger.info(f"Best parameters: {best_params}")

        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            all_trials=all_trials,
            optimization_time=optimization_time,
            n_evaluations=len(configs),
            convergence_history=[best_score]
        )


if __name__ == "__main__":
    # Example: Tune a simple function
    def objective(params):
        """Rosenbrock function (complex optimization landscape)"""
        x = params['x']
        y = params['y']
        a = params['a']
        return -(100 * (y - x**2)**2 + (a - x)**2)  # Negative for minimization

    # Define search space
    param_spaces = [
        HyperparameterSpace('x', 'continuous', bounds=(-5, 5)),
        HyperparameterSpace('y', 'continuous', bounds=(-5, 5)),
        HyperparameterSpace('a', 'discrete', bounds=(1, 5))
    ]

    # Quantum annealing tuner
    logger.info("=== Quantum Annealing Tuner ===")
    qa_tuner = QuantumAnnealingTuner(
        param_spaces=param_spaces,
        n_qubits=10,
        n_iterations=50,
        n_parallel=2
    )
    qa_result = qa_tuner.tune(objective, maximize=False)

    # Grid search for comparison
    logger.info("\n=== Grid Search Tuner ===")
    grid_tuner = GridSearchTuner(param_spaces, n_points=5)
    grid_result = grid_tuner.tune(objective, maximize=False)

    print(f"\nQuantum Annealing: Best score = {qa_result.best_score:.6f} "
          f"in {qa_result.optimization_time:.2f}s ({qa_result.n_evaluations} evals)")
    print(f"Grid Search: Best score = {grid_result.best_score:.6f} "
          f"in {grid_result.optimization_time:.2f}s ({grid_result.n_evaluations} evals)")
