from dataclasses import dataclass
from functools import lru_cache
from sklearn.base import BaseEstimator
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import cross_val_score
from typing import List, Dict, Tuple, Callable, Optional
import logging
import numpy as np


"""
Quantum-Inspired Feature Selection

This module implements quantum-inspired algorithms for feature selection in ML models.
Uses quantum superposition and entanglement concepts to efficiently explore the
exponentially large space of feature combinations.

Key techniques:
- Quantum-inspired genetic algorithm
- Amplitude amplification for important features
- Quantum-inspired mutual information maximization

Author: Quantum Optimization Team
Date: 2025-10-11
"""


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeatureSelectionResult:
    """Results from feature selection"""
    selected_features: List[int]
    feature_scores: np.ndarray
    selection_history: List[List[int]]
    optimization_time: float
    n_evaluations: int
    final_score: float


class QuantumFeatureSelector:
    """
    Quantum-inspired feature selector using superposition and amplitude amplification

    This selector treats features as qubits in superposition and uses
    quantum-inspired techniques to find optimal feature subsets.
    """

    def __init__(self,
                 n_features_to_select: Optional[int] = None,
                 n_iterations: int = 50,
                 population_size: int = 20,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7):
        """
        Initialize quantum feature selector

        Args:
            n_features_to_select: Target number of features (None = auto)
            n_iterations: Number of optimization iterations
            population_size: Size of population (feature subsets)
            mutation_rate: Probability of quantum mutation
            crossover_rate: Probability of quantum crossover
        """
        self.n_features_to_select = n_features_to_select
        self.n_iterations = n_iterations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.feature_importance_ = None

    def _initialize_population(self, n_features: int) -> np.ndarray:
        """
        Initialize population of feature subsets using quantum superposition

        Each individual is in superposition of feature states (selected/not selected)

        Args:
            n_features: Total number of features

        Returns:
            Population matrix (population_size x n_features) with binary values
        """
        if self.n_features_to_select is None:
            # Auto-select: start with 50% features
            n_selected = n_features // 2
        else:
            n_selected = self.n_features_to_select

        population = np.zeros((self.population_size, n_features), dtype=bool)

        for i in range(self.population_size):
            # Quantum superposition: probabilistic initialization
            selected_indices = np.random.choice(
                n_features,
                size=n_selected,
                replace=False
            )
            population[i, selected_indices] = True

        return population

    def _quantum_amplitude_amplification(self,
                                        feature_scores: np.ndarray,
                                        individual: np.ndarray) -> np.ndarray:
        """
        Apply quantum amplitude amplification to boost important features

        Inspired by Grover's amplitude amplification algorithm.

        Args:
            feature_scores: Importance score for each feature
            individual: Current feature selection (binary array)

        Returns:
            Amplified feature selection
        """
        n_features = len(individual)

        # Calculate amplification probabilities based on feature importance
        amplification_probs = np.exp(feature_scores) / np.sum(np.exp(feature_scores))

        # Apply amplitude amplification: boost high-scoring features
        amplified = individual.copy()
        for i in range(n_features):
            if not amplified[i]:
                # Probabilistically add important unselected features
                if np.random.random() < amplification_probs[i] * 0.1:
                    amplified[i] = True
            else:
                # Probabilistically remove unimportant selected features
                if np.random.random() < (1 - amplification_probs[i]) * 0.1:
                    amplified[i] = False

        return amplified

    def _quantum_crossover(self,
                          parent1: np.ndarray,
                          parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quantum-inspired crossover: entangle feature states from two parents

        Uses quantum entanglement concept to create correlated offspring.

        Args:
            parent1: First parent feature selection
            parent2: Second parent feature selection

        Returns:
            Two offspring feature selections
        """
        n_features = len(parent1)

        # Quantum entanglement: create superposition of parent states
        # Each position has probability based on parent agreement
        agreement = (parent1 == parent2)

        offspring1 = np.zeros(n_features, dtype=bool)
        offspring2 = np.zeros(n_features, dtype=bool)

        for i in range(n_features):
            if agreement[i]:
                # Strong entanglement: both offspring inherit same state
                offspring1[i] = offspring2[i] = parent1[i]
            else:
                # Weak entanglement: superposition state collapses randomly
                if np.random.random() < 0.5:
                    offspring1[i] = parent1[i]
                    offspring2[i] = parent2[i]
                else:
                    offspring1[i] = parent2[i]
                    offspring2[i] = parent1[i]

        return offspring1, offspring2

    def _quantum_mutation(self, individual: np.ndarray) -> np.ndarray:
        """
        Quantum mutation: apply quantum tunneling to explore feature space

        Args:
            individual: Feature selection to mutate

        Returns:
            Mutated feature selection
        """
        mutated = individual.copy()
        n_features = len(individual)

        # Quantum tunneling: flip feature states with quantum probability
        for i in range(n_features):
            if np.random.random() < self.mutation_rate:
                mutated[i] = not mutated[i]

        # Ensure at least one feature is selected
        if not np.any(mutated):
            mutated[np.random.randint(n_features)] = True

        return mutated

    def _evaluate_fitness(self,
                         individual: np.ndarray,
                         X: np.ndarray,
                         y: np.ndarray,
                         estimator: Optional[BaseEstimator] = None) -> float:
        """
        Evaluate fitness of a feature subset

        Args:
            individual: Feature selection (binary mask)
            X: Feature matrix
            y: Target variable
            estimator: Optional ML estimator for cross-validation

        Returns:
            Fitness score (higher is better)
        """
        selected_features = np.where(individual)[0]

        if len(selected_features) == 0:
            return 0.0

        X_selected = X[:, selected_features]

        if estimator is not None:
            # Use cross-validation score
            try:
                scores = cross_val_score(estimator, X_selected, y, cv=3, scoring='accuracy')
                fitness = np.mean(scores)
            except:
                # Fallback to mutual information if CV fails
                fitness = self._mutual_information_fitness(X_selected, y)
        else:
            # Use mutual information
            fitness = self._mutual_information_fitness(X_selected, y)

        # Penalty for selecting too many features (Occam's razor)
        n_features_penalty = len(selected_features) / X.shape[1]
        fitness -= 0.1 * n_features_penalty

        return fitness

    def _mutual_information_fitness(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate fitness based on mutual information with target

        Args:
            X: Selected features
            y: Target variable

        Returns:
            Average mutual information score
        """
        n_features = X.shape[1]
        mi_scores = []

        for i in range(n_features):
            mi = mutual_info_score(y, X[:, i])
            mi_scores.append(mi)

        return np.mean(mi_scores)

    def fit(self,
           X: np.ndarray,
           y: np.ndarray,
           estimator: Optional[BaseEstimator] = None) -> 'QuantumFeatureSelector':
        """
        Fit feature selector using quantum-inspired optimization

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target variable (n_samples,)
            estimator: Optional ML estimator for fitness evaluation

        Returns:
            Self
        """
        import time
        start_time = time.time()

        n_features = X.shape[1]

        # Calculate initial feature importance scores
        logger.info("Calculating feature importance scores...")
        feature_scores = np.zeros(n_features)
        for i in range(n_features):
            feature_scores[i] = mutual_info_score(y, X[:, i])

        self.feature_importance_ = feature_scores

        # Initialize population
        logger.info("Initializing quantum population...")
        population = self._initialize_population(n_features)

        # Evaluate initial fitness
        fitness_scores = np.array([
            self._evaluate_fitness(ind, X, y, estimator)
            for ind in population
        ])

        best_individual = population[np.argmax(fitness_scores)].copy()
        best_fitness = np.max(fitness_scores)
        selection_history = [np.where(best_individual)[0].tolist()]

        logger.info(f"Initial best fitness: {best_fitness:.4f}")

        # Quantum evolutionary optimization
        for iteration in range(self.n_iterations):
            new_population = []

            # Selection: quantum measurement collapses to high-fitness states
            selection_probs = fitness_scores / np.sum(fitness_scores)

            # Generate offspring through quantum operations
            for _ in range(self.population_size // 2):
                # Select parents
                parent_indices = np.random.choice(
                    self.population_size,
                    size=2,
                    p=selection_probs,
                    replace=False
                )
                parent1 = population[parent_indices[0]]
                parent2 = population[parent_indices[1]]

                # Quantum crossover
                if np.random.random() < self.crossover_rate:
                    offspring1, offspring2 = self._quantum_crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1.copy(), parent2.copy()

                # Quantum mutation
                offspring1 = self._quantum_mutation(offspring1)
                offspring2 = self._quantum_mutation(offspring2)

                # Amplitude amplification
                offspring1 = self._quantum_amplitude_amplification(feature_scores, offspring1)
                offspring2 = self._quantum_amplitude_amplification(feature_scores, offspring2)

                new_population.extend([offspring1, offspring2])

            # Evaluate new population
            population = np.array(new_population)
            fitness_scores = np.array([
                self._evaluate_fitness(ind, X, y, estimator)
                for ind in population
            ])

            # Update best solution
            current_best_fitness = np.max(fitness_scores)
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[np.argmax(fitness_scores)].copy()
                logger.info(f"Iteration {iteration}: New best fitness = {best_fitness:.4f}")

            selection_history.append(np.where(best_individual)[0].tolist())

            # Early stopping
            if iteration > 10:
                recent_history = selection_history[-10:]
                if all(set(h) == set(recent_history[0]) for h in recent_history):
                    logger.info(f"Converged at iteration {iteration}")
                    break

        optimization_time = time.time() - start_time

        self.selected_features_ = np.where(best_individual)[0]
        self.selection_history_ = selection_history
        self.final_score_ = best_fitness
        self.optimization_time_ = optimization_time
        self.n_evaluations_ = len(population) * (iteration + 1)

        logger.info(f"\nFeature selection complete!")
        logger.info(f"Selected {len(self.selected_features_)} features: {self.selected_features_}")
        logger.info(f"Final fitness: {best_fitness:.4f}")
        logger.info(f"Optimization time: {optimization_time:.2f}s")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using selected features

        Args:
            X: Feature matrix

        Returns:
            Transformed feature matrix with selected features only
        """
        if not hasattr(self, 'selected_features_'):
            raise ValueError("Feature selector not fitted. Call fit() first.")

        return X[:, self.selected_features_]

    def fit_transform(self,
                     X: np.ndarray,
                     y: np.ndarray,
                     estimator: Optional[BaseEstimator] = None) -> np.ndarray:
        """
        Fit selector and transform data in one step

        Args:
            X: Feature matrix
            y: Target variable
            estimator: Optional ML estimator

        Returns:
            Transformed feature matrix
        """
        self.fit(X, y, estimator)
        return self.transform(X)

    @lru_cache(maxsize=128)

    def get_support(self, indices: bool = False) -> np.ndarray:
        """
        Get selected features

        Args:
            indices: If True, return feature indices; if False, return boolean mask

        Returns:
            Selected features as indices or mask
        """
        if not hasattr(self, 'selected_features_'):
            raise ValueError("Feature selector not fitted. Call fit() first.")

        if indices:
            return self.selected_features_
        else:
            mask = np.zeros(len(self.feature_importance_), dtype=bool)
            mask[self.selected_features_] = True
            return mask


class ClassicalFeatureSelector:
    """Classical feature selection using recursive feature elimination for comparison"""

    def __init__(self, n_features_to_select: Optional[int] = None):
        """
        Initialize classical selector

        Args:
            n_features_to_select: Target number of features
        """
        self.n_features_to_select = n_features_to_select

    def fit(self,
           X: np.ndarray,
           y: np.ndarray,
           estimator: Optional[BaseEstimator] = None) -> 'ClassicalFeatureSelector':
        """
        Fit using mutual information ranking

        Args:
            X: Feature matrix
            y: Target variable
            estimator: Unused (for API compatibility)

        Returns:
            Self
        """
        import time
        start_time = time.time()

        n_features = X.shape[1]

        # Calculate mutual information scores
        feature_scores = np.array([
            mutual_info_score(y, X[:, i])
            for i in range(n_features)
        ])

        # Select top features
        if self.n_features_to_select is None:
            n_selected = n_features // 2
        else:
            n_selected = self.n_features_to_select

        self.selected_features_ = np.argsort(feature_scores)[-n_selected:]
        self.feature_importance_ = feature_scores
        self.optimization_time_ = time.time() - start_time

        logger.info(f"Classical selection: Selected {len(self.selected_features_)} features")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using selected features"""
        return X[:, self.selected_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray, estimator=None) -> np.ndarray:
        """Fit and transform"""
        self.fit(X, y, estimator)
        return self.transform(X)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

    # Generate sample data
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42
    )

    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Quantum feature selection
    logger.info("\n=== Quantum Feature Selector ===")
    quantum_selector = QuantumFeatureSelector(
        n_features_to_select=10,
        n_iterations=30,
        population_size=20
    )
    quantum_selector.fit(X, y, RandomForestClassifier())

    # Classical feature selection
    logger.info("\n=== Classical Feature Selector ===")
    classical_selector = ClassicalFeatureSelector(n_features_to_select=10)
    classical_selector.fit(X, y)

    print(f"\nQuantum selected features: {quantum_selector.selected_features_}")
    print(f"Classical selected features: {classical_selector.selected_features_}")
    print(f"\nQuantum optimization time: {quantum_selector.optimization_time_:.2f}s")
    print(f"Classical optimization time: {classical_selector.optimization_time_:.2f}s")
