"""
Loss Functions for Neural Network Training

This module implements various loss functions optimized for cryptocurrency
price prediction and trading applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Reduces loss for well-classified examples and focuses on hard examples.
    Reference: https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        """
        Initialize focal loss.

        Args:
            alpha: Class weights [num_classes]
            gamma: Focusing parameter (default: 2.0)
            reduction: Reduction method ('none', 'mean', 'sum')
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Loss value
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction='none',
            label_smoothing=self.label_smoothing
        )

        # Compute focal weight
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma

        # Apply focal weight
        loss = focal_weight * ce_loss

        # Apply class weights
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy with label smoothing.

    Prevents overconfidence by distributing probability mass to other classes.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = 'mean'
    ):
        """
        Initialize label smoothing cross entropy.

        Args:
            smoothing: Smoothing factor (0.0 = no smoothing, 1.0 = uniform)
            reduction: Reduction method
        """
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss with label smoothing.

        Args:
            inputs: Predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Loss value
        """
        num_classes = inputs.size(-1)

        # Compute log probabilities
        log_probs = F.log_softmax(inputs, dim=-1)

        # One-hot encode targets with smoothing
        targets_one_hot = torch.zeros_like(log_probs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        # Apply smoothing
        targets_smooth = (1 - self.smoothing) * targets_one_hot + \
                        self.smoothing / num_classes

        # Compute loss
        loss = -(targets_smooth * log_probs).sum(dim=-1)

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MultiHorizonLoss(nn.Module):
    """
    Multi-horizon loss for predicting multiple time horizons simultaneously.

    Combines losses from different prediction horizons with optional weighting.
    """

    def __init__(
        self,
        horizons: list = None,
        horizon_weights: Optional[Dict[str, float]] = None,
        loss_type: str = 'focal',
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        class_weights: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Initialize multi-horizon loss.

        Args:
            horizons: List of prediction horizons
            horizon_weights: Weights for each horizon
            loss_type: Type of loss ('focal', 'ce', 'smooth_ce')
            focal_gamma: Gamma parameter for focal loss
            label_smoothing: Label smoothing factor
            class_weights: Class weights for each horizon
        """
        super().__init__()

        self.horizons = horizons or ['5min', '15min', '1hr']
        self.loss_type = loss_type

        # Default equal weights
        if horizon_weights is None:
            horizon_weights = {h: 1.0 for h in self.horizons}
        self.horizon_weights = horizon_weights

        # Create loss functions for each horizon
        self.loss_functions = nn.ModuleDict()

        for horizon in self.horizons:
            alpha = class_weights[horizon] if class_weights and horizon in class_weights else None

            if loss_type == 'focal':
                self.loss_functions[horizon] = FocalLoss(
                    alpha=alpha,
                    gamma=focal_gamma,
                    label_smoothing=label_smoothing
                )
            elif loss_type == 'smooth_ce':
                self.loss_functions[horizon] = LabelSmoothingCrossEntropy(
                    smoothing=label_smoothing
                )
            else:  # Standard cross entropy
                self.loss_functions[horizon] = nn.CrossEntropyLoss(
                    weight=alpha,
                    label_smoothing=label_smoothing
                )

    def forward(
        self,
        predictions: Dict[str, Dict[str, torch.Tensor]],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-horizon loss.

        Args:
            predictions: Dictionary of predictions for each horizon
                        {horizon: {'logits': tensor, ...}}
            targets: Dictionary of target labels for each horizon
                    {horizon: tensor}

        Returns:
            Total loss and per-horizon losses
        """
        total_loss = 0.0
        horizon_losses = {}

        for horizon in self.horizons:
            if horizon in predictions and horizon in targets:
                # Get logits and targets
                logits = predictions[horizon]['logits']
                target = targets[horizon]

                # Compute loss
                loss = self.loss_functions[horizon](logits, target)

                # Apply horizon weight
                weighted_loss = self.horizon_weights[horizon] * loss

                total_loss += weighted_loss
                horizon_losses[horizon] = loss.item()

        return total_loss, horizon_losses


class TradingLoss(nn.Module):
    """
    Trading-aware loss that considers profitability and directional accuracy.

    Penalizes predictions that would lead to losing trades.
    """

    def __init__(
        self,
        transaction_cost: float = 0.001,
        risk_penalty: float = 0.5,
        confidence_weight: float = 0.2
    ):
        """
        Initialize trading loss.

        Args:
            transaction_cost: Transaction cost (0.1% = 0.001)
            risk_penalty: Penalty for risky predictions
            confidence_weight: Weight for confidence calibration
        """
        super().__init__()
        self.transaction_cost = transaction_cost
        self.risk_penalty = risk_penalty
        self.confidence_weight = confidence_weight

    def forward(
        self,
        predictions: Dict[str, Dict[str, torch.Tensor]],
        targets: Dict[str, torch.Tensor],
        prices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute trading loss.

        Args:
            predictions: Model predictions
            targets: True labels
            prices: Current prices (for computing returns)

        Returns:
            Loss value
        """
        total_loss = 0.0

        for horizon in predictions.keys():
            if horizon not in targets:
                continue

            probs = predictions[horizon]['probs']
            logits = predictions[horizon]['logits']
            target = targets[horizon]
            confidence = predictions[horizon].get('confidence', None)

            # Classification loss
            ce_loss = F.cross_entropy(logits, target)

            # Directional accuracy penalty
            pred_direction = torch.argmax(probs, dim=-1)
            target_direction = target

            # Map to trading signals: 0=sell, 1=hold, 2=buy
            # Penalize more for wrong direction (0 vs 2, 2 vs 0)
            wrong_direction = ((pred_direction == 0) & (target_direction == 2)) | \
                            ((pred_direction == 2) & (target_direction == 0))

            directional_penalty = wrong_direction.float().mean() * self.risk_penalty

            # Confidence calibration
            if confidence is not None:
                # High confidence on wrong predictions should be penalized
                correct = (pred_direction == target_direction).float()
                confidence_loss = torch.abs(confidence.squeeze() - correct).mean()
            else:
                confidence_loss = 0.0

            # Combine losses
            horizon_loss = ce_loss + directional_penalty + \
                          self.confidence_weight * confidence_loss

            total_loss += horizon_loss

        return total_loss / len(predictions)


class UncertaintyLoss(nn.Module):
    """
    Loss function that incorporates aleatoric and epistemic uncertainty.

    Helps model learn when it's uncertain about predictions.
    """

    def __init__(self, num_classes: int = 3):
        """
        Initialize uncertainty loss.

        Args:
            num_classes: Number of output classes
        """
        super().__init__()
        self.num_classes = num_classes

    def forward(
        self,
        predictions: Dict[str, Dict[str, torch.Tensor]],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute uncertainty-aware loss.

        Args:
            predictions: Model predictions with uncertainty estimates
            targets: True labels

        Returns:
            Total loss and uncertainty metrics
        """
        total_loss = 0.0
        uncertainty_metrics = {}

        for horizon in predictions.keys():
            if horizon not in targets:
                continue

            logits = predictions[horizon]['logits']
            target = targets[horizon]

            # Get uncertainty if available
            if 'uncertainty' in predictions[horizon]:
                uncertainty = predictions[horizon]['uncertainty']
                aleatoric = uncertainty[:, 0]  # Data uncertainty
                epistemic = uncertainty[:, 1]   # Model uncertainty

                # Compute negative log likelihood with uncertainty
                ce_loss = F.cross_entropy(logits, target, reduction='none')

                # Uncertainty-weighted loss
                # Higher uncertainty -> lower weight on loss
                uncertainty_weight = 1.0 / (1.0 + aleatoric + epistemic)
                weighted_loss = (ce_loss * uncertainty_weight).mean()

                # Regularization to prevent uncertainty from becoming too large
                uncertainty_reg = (aleatoric.mean() + epistemic.mean()) * 0.01

                horizon_loss = weighted_loss + uncertainty_reg

                # Track metrics
                uncertainty_metrics[f'{horizon}_aleatoric'] = aleatoric.mean().item()
                uncertainty_metrics[f'{horizon}_epistemic'] = epistemic.mean().item()
            else:
                # Standard cross entropy
                horizon_loss = F.cross_entropy(logits, target)

            total_loss += horizon_loss

        return total_loss, uncertainty_metrics


if __name__ == "__main__":
    # Test loss functions
    print("Testing Loss Functions...")

    batch_size = 32
    num_classes = 3

    # Create dummy predictions and targets
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    # Test FocalLoss
    print("\n1. Testing FocalLoss...")
    focal_loss = FocalLoss(gamma=2.0, label_smoothing=0.1)
    loss1 = focal_loss(logits, targets)
    print(f"   Focal Loss: {loss1.item():.4f}")

    # Test LabelSmoothingCrossEntropy
    print("\n2. Testing LabelSmoothingCrossEntropy...")
    smooth_ce = LabelSmoothingCrossEntropy(smoothing=0.1)
    loss2 = smooth_ce(logits, targets)
    print(f"   Smooth CE Loss: {loss2.item():.4f}")

    # Test MultiHorizonLoss
    print("\n3. Testing MultiHorizonLoss...")
    predictions = {
        '5min': {'logits': logits, 'probs': F.softmax(logits, dim=-1)},
        '15min': {'logits': logits, 'probs': F.softmax(logits, dim=-1)},
        '1hr': {'logits': logits, 'probs': F.softmax(logits, dim=-1)}
    }
    targets_dict = {
        '5min': targets,
        '15min': targets,
        '1hr': targets
    }

    multi_loss = MultiHorizonLoss(
        horizons=['5min', '15min', '1hr'],
        loss_type='focal',
        focal_gamma=2.0
    )
    total_loss, horizon_losses = multi_loss(predictions, targets_dict)
    print(f"   Total Loss: {total_loss.item():.4f}")
    print(f"   Horizon Losses: {horizon_losses}")

    # Test TradingLoss
    print("\n4. Testing TradingLoss...")
    predictions_with_conf = {
        '5min': {
            'logits': logits,
            'probs': F.softmax(logits, dim=-1),
            'confidence': torch.rand(batch_size, 1)
        }
    }
    trading_loss = TradingLoss()
    loss4 = trading_loss(predictions_with_conf, {'5min': targets})
    print(f"   Trading Loss: {loss4.item():.4f}")

    # Test UncertaintyLoss
    print("\n5. Testing UncertaintyLoss...")
    predictions_with_uncertainty = {
        '5min': {
            'logits': logits,
            'probs': F.softmax(logits, dim=-1),
            'uncertainty': torch.rand(batch_size, 2)
        }
    }
    uncertainty_loss = UncertaintyLoss()
    loss5, metrics = uncertainty_loss(predictions_with_uncertainty, {'5min': targets})
    print(f"   Uncertainty Loss: {loss5.item():.4f}")
    print(f"   Uncertainty Metrics: {metrics}")

    print("\nLoss function tests completed successfully!")
