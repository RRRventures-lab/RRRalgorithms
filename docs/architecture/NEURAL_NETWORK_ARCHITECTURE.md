# Neural Network Architecture

## Overview

The RRRalgorithms trading system employs a sophisticated ensemble of neural networks, each specialized for different aspects of trading decision-making. This document outlines the complete neural network architecture, training pipelines, and deployment strategies.

## High-Level Architecture

```
Market Data Stream
        ↓
┌───────────────────────────────────────────────────────┐
│              Feature Engineering Pipeline              │
│  (Raw data → Normalized features → Embedding space)   │
└───────────────┬───────────────────────────────────────┘
                ↓
    ┌───────────┴───────────┐
    │                       │
    ▼                       ▼
┌─────────┐           ┌─────────┐
│ Price   │           │Sentiment│
│Prediction│          │Analysis │
│ Network │           │ Network │
└────┬────┘           └────┬────┘
     │                     │
     └─────────┬───────────┘
               ▼
      ┌────────────────┐
      │   Portfolio    │
      │  Optimization  │
      │    Network     │
      └───────┬────────┘
              ▼
      ┌────────────────┐
      │   Execution    │
      │   Strategy     │
      │    Network     │
      │      (RL)      │
      └───────┬────────┘
              ▼
         Trading Signal
              ↓
       Order Execution
```

## Neural Network Components

### 1. Price Prediction Network

**Purpose**: Predict future price movements and volatility

**Architecture**: Transformer-based Sequence Model

#### Model Specification

```python
class PricePredictionTransformer(nn.Module):
    """
    Transformer model for multi-horizon price prediction
    """
    def __init__(
        self,
        input_dim: int = 128,          # Feature dimension
        d_model: int = 512,             # Model dimension
        nhead: int = 8,                 # Attention heads
        num_encoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 1000,     # Maximum sequence length
        prediction_horizons: List[int] = [1, 5, 15, 60],  # Minutes ahead
    ):
        super().__init__()

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Prediction heads (one per horizon)
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 3)  # [price_change, volatility, confidence]
            )
            for _ in prediction_horizons
        ])

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_length, input_dim)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, mask=mask)

        # Use last time step for predictions
        last_hidden = x[:, -1, :]

        # Multi-horizon predictions
        predictions = [head(last_hidden) for head in self.prediction_heads]
        return predictions
```

#### Input Features (128 dimensions)

1. **Price Features (20 dims)**
   - OHLCV (5 dims)
   - Log returns (1, 5, 15, 60 min)
   - Price momentum indicators

2. **Technical Indicators (30 dims)**
   - Moving averages (SMA, EMA)
   - RSI, MACD, Bollinger Bands
   - Stochastic oscillator
   - ATR (volatility)

3. **Order Book Features (25 dims)**
   - Bid-ask spread
   - Order book imbalance
   - Depth at various levels
   - Large order detection

4. **Volume Profile (15 dims)**
   - Volume-weighted metrics
   - VWAP deviation
   - Volume trend indicators

5. **Market Microstructure (20 dims)**
   - Trade arrival rate
   - Effective spread
   - Price impact
   - Realized volatility

6. **Sentiment Features (10 dims)**
   - News sentiment score (from Perplexity)
   - Social media sentiment
   - Fear & Greed Index

7. **Temporal Features (8 dims)**
   - Time of day
   - Day of week
   - Market session
   - Holiday indicator

#### Training Strategy

```python
# Training configuration
training_config = {
    "optimizer": "AdamW",
    "learning_rate": 1e-4,
    "batch_size": 64,
    "sequence_length": 500,  # 500 time steps
    "num_epochs": 100,
    "early_stopping_patience": 10,

    # Loss function: Multi-task learning
    "loss_weights": {
        "price_prediction": 1.0,      # MSE loss for price
        "volatility_prediction": 0.5,  # MSE loss for volatility
        "direction_accuracy": 0.8,     # Cross-entropy for direction
    },

    # Regularization
    "weight_decay": 1e-5,
    "dropout": 0.1,
    "label_smoothing": 0.1,

    # Data augmentation
    "noise_injection": 0.01,
    "mixup_alpha": 0.2,
}
```

#### Evaluation Metrics

```python
evaluation_metrics = {
    "MAE": "Mean Absolute Error for price prediction",
    "RMSE": "Root Mean Squared Error",
    "Direction_Accuracy": "Percentage of correct direction predictions",
    "Sharpe_Ratio": "Sharpe ratio of trading strategy using predictions",
    "Sortino_Ratio": "Downside risk-adjusted returns",
    "Max_Drawdown": "Maximum peak-to-trough decline",
    "Profit_Factor": "Gross profit / gross loss",
}
```

---

### 2. Sentiment Analysis Network

**Purpose**: Analyze market sentiment from news, social media, and research

**Architecture**: BERT-based NLP Model (FinBERT fine-tuned)

#### Model Specification

```python
from transformers import BertModel, BertTokenizer

class SentimentAnalysisNetwork(nn.Module):
    """
    FinBERT-based sentiment analysis for financial text
    """
    def __init__(
        self,
        bert_model_name: str = "ProsusAI/finbert",
        num_sentiment_classes: int = 3,  # Positive, Neutral, Negative
        dropout: float = 0.1
    ):
        super().__init__()

        # Load pre-trained FinBERT
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_sentiment_classes)
        )

        # Additional outputs
        self.confidence_head = nn.Linear(512, 1)  # Confidence score
        self.impact_head = nn.Linear(512, 1)      # Expected price impact

    def forward(self, input_ids, attention_mask):
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Intermediate representation
        intermediate = self.classifier[:-1](cls_output)  # All but last layer

        # Predictions
        sentiment = self.classifier[-1](intermediate)
        confidence = torch.sigmoid(self.confidence_head(intermediate))
        impact = torch.tanh(self.impact_head(intermediate))

        return {
            "sentiment": sentiment,       # Logits for 3 classes
            "confidence": confidence,     # 0-1 confidence score
            "price_impact": impact        # -1 to 1 expected impact
        }
```

#### Data Sources for Sentiment

1. **Perplexity AI** (Primary)
   - Real-time news aggregation
   - Research report synthesis
   - Event detection

2. **Twitter/X API** (Secondary)
   - Crypto influencer tweets
   - Trading community sentiment

3. **Reddit API** (Tertiary)
   - r/cryptocurrency, r/Bitcoin sentiment
   - r/wallstreetbets for meme stock trading

4. **News APIs**
   - CryptoPanic API
   - CoinMarketCap news

#### Training Data

```python
# Sentiment dataset structure
sentiment_training_data = {
    "source": "perplexity",                  # Data source
    "text": "Bitcoin surges past $70k...",   # Article/tweet text
    "timestamp": "2025-10-11T14:30:00Z",
    "asset": "BTC",
    "sentiment_label": "positive",            # Ground truth
    "price_change_1h": 0.023,                # Price change after 1h
    "price_change_4h": 0.045,
    "volume_surge": 1.5,                     # Volume relative to average
}
```

---

### 3. Portfolio Optimization Network

**Purpose**: Optimize portfolio weights using quantum-inspired algorithms

**Architecture**: Quantum-Inspired Neural Network (QAOA-style)

#### Model Specification

```python
class QuantumInspiredPortfolioOptimizer(nn.Module):
    """
    Quantum-inspired portfolio optimization using variational circuits
    """
    def __init__(
        self,
        num_assets: int = 50,
        num_layers: int = 4,
        embedding_dim: int = 64
    ):
        super().__init__()

        # Asset embedding
        self.asset_embeddings = nn.Embedding(num_assets, embedding_dim)

        # Variational quantum circuit simulation
        self.quantum_layers = nn.ModuleList([
            QuantumInspiredLayer(embedding_dim) for _ in range(num_layers)
        ])

        # Classical post-processing
        self.weight_decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softmax(dim=0)  # Portfolio weights sum to 1
        )

    def forward(self, asset_features, constraints):
        # asset_features: (num_assets, feature_dim)
        # constraints: dict with budget, max_position_size, etc.

        batch_size = asset_features.shape[0]

        # Embed assets
        embedded = self.asset_embeddings(torch.arange(batch_size))
        combined = torch.cat([embedded, asset_features], dim=-1)

        # Apply quantum-inspired transformations
        quantum_state = combined
        for layer in self.quantum_layers:
            quantum_state = layer(quantum_state)

        # Decode to portfolio weights
        weights = self.weight_decoder(quantum_state).squeeze(-1)

        # Apply constraints
        weights = self.apply_constraints(weights, constraints)

        return weights

    def apply_constraints(self, weights, constraints):
        """Apply portfolio constraints (budget, position limits, etc.)"""
        # Max position size
        max_weight = constraints.get("max_position_size", 0.2)
        weights = torch.clamp(weights, max=max_weight)

        # Normalize to sum to 1
        weights = weights / weights.sum()

        return weights

class QuantumInspiredLayer(nn.Module):
    """Simulates a layer of a quantum variational circuit"""
    def __init__(self, dim: int):
        super().__init__()
        # Rotation gates (RX, RY, RZ)
        self.rotation_params = nn.Parameter(torch.randn(dim, 3))
        # Entangling gates (simulated)
        self.entangle_weights = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x):
        # Rotation: simulate quantum rotation gates
        rotated = self.apply_rotations(x, self.rotation_params)

        # Entanglement: simulate quantum entanglement
        entangled = torch.matmul(rotated, self.entangle_weights)

        return entangled

    def apply_rotations(self, x, params):
        """Apply rotation-like transformations"""
        rx = x * torch.cos(params[:, 0]) + torch.sin(params[:, 0])
        ry = rx * torch.cos(params[:, 1]) + torch.sin(params[:, 1])
        rz = ry * torch.cos(params[:, 2]) + torch.sin(params[:, 2])
        return rz
```

#### Optimization Objectives

```python
# Multi-objective optimization
optimization_objectives = {
    "maximize_return": {
        "weight": 1.0,
        "function": lambda weights, returns: (weights * returns).sum()
    },
    "minimize_risk": {
        "weight": 0.5,
        "function": lambda weights, cov_matrix: (weights @ cov_matrix @ weights.T)
    },
    "minimize_turnover": {
        "weight": 0.2,
        "function": lambda weights, prev_weights: (weights - prev_weights).abs().sum()
    },
    "maximize_sharpe": {
        "weight": 0.8,
        "function": lambda weights, returns, cov: (weights * returns).sum() / torch.sqrt(weights @ cov @ weights.T)
    }
}
```

---

### 4. Execution Strategy Network (Reinforcement Learning)

**Purpose**: Learn optimal order execution strategies to minimize market impact

**Architecture**: Proximal Policy Optimization (PPO) with Actor-Critic

#### Model Specification

```python
import torch.nn as nn
from torch.distributions import Categorical

class ExecutionPolicyNetwork(nn.Module):
    """
    PPO-based order execution strategy
    Learns when and how to place orders to minimize market impact
    """
    def __init__(
        self,
        state_dim: int = 64,      # Market state features
        action_dim: int = 10,      # Discretized order sizes
        hidden_dim: int = 256
    ):
        super().__init__()

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state):
        features = self.feature_extractor(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value

    def act(self, state):
        """Sample action from policy"""
        action_probs, _ = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate(self, state, action):
        """Evaluate state-action pair for PPO update"""
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_log_probs, state_value, dist_entropy
```

#### State Space (Execution Environment)

```python
state_features = {
    "order_book_state": {
        "bid_ask_spread": float,
        "order_book_imbalance": float,
        "depth_at_best": float,
        "large_order_presence": bool
    },
    "execution_progress": {
        "remaining_quantity": float,       # How much left to execute
        "time_remaining": int,              # Seconds left
        "executed_quantity": float,
        "average_execution_price": float
    },
    "market_conditions": {
        "volatility": float,
        "volume_trend": float,
        "price_momentum": float
    },
    "slippage": {
        "current_slippage": float,          # vs VWAP
        "predicted_impact": float
    }
}
```

#### Action Space

```python
# Discretized order placement actions
action_space = {
    0: "wait",                    # Don't place order this step
    1: "market_order_10%",        # Execute 10% of remaining with market order
    2: "market_order_25%",
    3: "market_order_50%",
    4: "limit_order_aggressive",  # Limit order at best bid/ask
    5: "limit_order_passive",     # Limit order deeper in book
    6: "iceberg_order",           # Large hidden order
    7: "twap_slice",              # Time-weighted slice
    8: "vwap_slice",              # Volume-weighted slice
    9: "adaptive_slice",          # ML-determined size
}
```

#### Reward Function

```python
def execution_reward(state, action, next_state):
    """
    Reward function for execution strategy
    Encourages minimizing:
    - Slippage
    - Market impact
    - Execution time (with time penalty)
    """
    slippage_cost = (
        state["executed_price"] - state["arrival_price"]
    ) / state["arrival_price"]

    market_impact = estimate_market_impact(action, state["order_book"])

    time_penalty = -0.01 * state["time_elapsed"]  # Encourage faster execution

    completion_bonus = 10.0 if state["remaining_quantity"] == 0 else 0.0

    reward = (
        -100.0 * abs(slippage_cost) +
        -50.0 * market_impact +
        time_penalty +
        completion_bonus
    )

    return reward
```

#### Training with PPO

```python
# PPO training configuration
ppo_config = {
    "gamma": 0.99,                    # Discount factor
    "lambda_gae": 0.95,               # GAE parameter
    "clip_epsilon": 0.2,              # PPO clip parameter
    "c1": 1.0,                        # Value loss coefficient
    "c2": 0.01,                       # Entropy coefficient
    "num_epochs": 10,                 # PPO epochs per update
    "batch_size": 64,
    "learning_rate": 3e-4,
    "max_grad_norm": 0.5,

    # Simulation environment
    "num_parallel_envs": 16,          # Parallel execution simulations
    "steps_per_update": 2048,
}
```

---

## Training Infrastructure

### Distributed Training with Ray

```python
# Training orchestration with Ray
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

# Initialize Ray
ray.init(num_gpus=4)

# Hyperparameter search
config = {
    "price_prediction": {
        "learning_rate": tune.loguniform(1e-5, 1e-3),
        "num_layers": tune.choice([4, 6, 8]),
        "d_model": tune.choice([256, 512, 1024]),
    },
    "sentiment": {
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "dropout": tune.uniform(0.1, 0.3),
    },
    "execution_rl": PPOTrainer.get_default_config(),
}

# Launch parallel training
analysis = tune.run(
    train_model,
    config=config,
    num_samples=20,
    resources_per_trial={"gpu": 1}
)
```

### Model Versioning with MLflow

```python
import mlflow

# Log model training
with mlflow.start_run(run_name="price_prediction_v1.2"):
    mlflow.log_params(training_config)

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader)
        val_loss = validate(model, val_loader)

        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss
        }, step=epoch)

    # Save model
    mlflow.pytorch.log_model(model, "model")
    mlflow.log_artifacts("./plots")
```

---

## Inference Pipeline

### Real-Time Inference Architecture

```
Market Data → Feature Pipeline → Model Ensemble → Signal Aggregation → Trading Decision
     (1ms)         (5ms)             (10ms)            (2ms)              (Total: 18ms)
```

### Ensemble Prediction

```python
class ModelEnsemble:
    """Ensemble of all neural networks for final decision"""

    def __init__(self):
        self.price_predictor = load_model("price_prediction", version="latest")
        self.sentiment_analyzer = load_model("sentiment", version="latest")
        self.portfolio_optimizer = load_model("portfolio_opt", version="latest")
        self.execution_policy = load_model("execution_rl", version="latest")

    async def predict(self, market_state):
        # Parallel inference
        price_pred, sentiment, portfolio_weights = await asyncio.gather(
            self.price_predictor.predict(market_state["features"]),
            self.sentiment_analyzer.analyze(market_state["news"]),
            self.portfolio_optimizer.optimize(market_state["positions"])
        )

        # Aggregate signals
        trading_signal = self.aggregate_signals(
            price_pred, sentiment, portfolio_weights
        )

        # Execution strategy
        execution_plan = self.execution_policy.plan(
            trading_signal, market_state["order_book"]
        )

        return execution_plan
```

---

**Last Updated**: 2025-10-11
**Maintained By**: Machine Learning Team
