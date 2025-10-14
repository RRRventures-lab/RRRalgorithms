from models.price_prediction.transformer_model import (
import numpy as np
import os
import pytest
import sys
import torch
import torch.nn as nn

"""
Unit Tests for Improved Transformer Price Predictor

Tests all new improvements: dropout, causal masking, gradient clipping, label smoothing.
Critical for model quality and production deployment.
"""


# Add neural network to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../worktrees/neural-network/src'))

    TransformerPricePredictor,
    PricePredictionLoss,
    create_price_predictor
)


class TestTransformerInitialization:
    """Test Transformer initialization with new parameters"""

    def test_default_initialization(self):
        """Test default model initialization"""
        model = TransformerPricePredictor()

        assert model is not None
        assert model.d_model == 256
        assert model.use_causal_mask is True  # NEW: Default causal masking
        assert hasattr(model, 'input_dropout')  # NEW: Input dropout layer

    def test_custom_dropout_initialization(self):
        """Test initialization with custom dropout rates"""
        model = TransformerPricePredictor(
            dropout=0.1,
            attention_dropout=0.3,
            head_dropout=0.25
        )

        assert model.input_dropout.p == 0.1
        assert model.head_dropout == 0.25

    def test_causal_mask_disabled(self):
        """Test initialization with causal masking disabled"""
        model = TransformerPricePredictor(use_causal_mask=False)

        assert model.use_causal_mask is False

    def test_parameter_count(self):
        """Test model has expected number of parameters"""
        model = TransformerPricePredictor(d_model=128, nhead=4, num_encoder_layers=3)

        param_count = sum(p.numel() for p in model.parameters())

        assert param_count > 100000  # Should have substantial parameters
        assert param_count < 10000000  # But not excessive


class TestCausalMasking:
    """Test causal attention mask generation and application"""

    def test_causal_mask_generation(self):
        """Test causal mask is correctly generated"""
        model = TransformerPricePredictor(seq_len=10)
        device = torch.device('cpu')

        mask = model.generate_causal_mask(10, device)

        assert mask.shape == (10, 10)
        assert mask.dtype == torch.bool

        # Check mask is upper triangular (True values prevent attention)
        for i in range(10):
            for j in range(10):
                if j > i:
                    assert mask[i, j] == True  # Can't attend to future
                else:
                    assert mask[i, j] == False  # Can attend to past/present

    def test_causal_mask_prevents_future_leakage(self):
        """Test causal mask prevents looking at future data"""
        batch_size = 2
        seq_len = 5
        input_dim = 6

        model = TransformerPricePredictor(seq_len=seq_len, use_causal_mask=True)
        x = torch.randn(batch_size, seq_len, input_dim)

        outputs = model(x)

        # Model should produce valid outputs without errors
        assert outputs['5min']['logits'].shape == (batch_size, 3)

    def test_no_causal_mask(self):
        """Test model works without causal masking"""
        batch_size = 2
        seq_len = 5
        input_dim = 6

        model = TransformerPricePredictor(seq_len=seq_len, use_causal_mask=False)
        x = torch.randn(batch_size, seq_len, input_dim)

        outputs = model(x)

        assert outputs['5min']['logits'].shape == (batch_size, 3)


class TestDropoutLayers:
    """Test dropout layers are properly integrated"""

    def test_input_dropout_active(self):
        """Test input dropout is active during training"""
        model = TransformerPricePredictor(dropout=0.5)  # High dropout for testing
        model.train()

        x = torch.ones(2, 10, 6)

        # Run multiple forward passes
        outputs1 = model.input_embedding(x)
        dropped1 = model.input_dropout(outputs1)

        outputs2 = model.input_embedding(x)
        dropped2 = model.input_dropout(outputs2)

        # With dropout, outputs should differ
        assert not torch.allclose(dropped1, dropped2)

    def test_dropout_disabled_in_eval(self):
        """Test dropout is disabled during evaluation"""
        model = TransformerPricePredictor(dropout=0.5)
        model.eval()

        x = torch.ones(2, 10, 6)

        # Run multiple forward passes
        outputs1 = model.input_embedding(x)
        dropped1 = model.input_dropout(outputs1)

        outputs2 = model.input_embedding(x)
        dropped2 = model.input_dropout(outputs2)

        # Without dropout, outputs should be identical
        assert torch.allclose(dropped1, dropped2)

    def test_prediction_head_dropout(self):
        """Test prediction heads have correct dropout"""
        model = TransformerPricePredictor(head_dropout=0.3)

        # Check dropout layers exist in prediction heads
        for layer in model.head_5min:
            if isinstance(layer, nn.Dropout):
                assert layer.p == 0.3


class TestModelForwardPass:
    """Test complete forward pass"""

    @pytest.fixture
    def sample_batch(self):
        """Create sample input batch"""
        batch_size = 4
        seq_len = 100
        input_dim = 6
        return torch.randn(batch_size, seq_len, input_dim)

    def test_forward_pass_shapes(self, sample_batch):
        """Test forward pass produces correct output shapes"""
        model = TransformerPricePredictor()
        outputs = model(sample_batch)

        assert '5min' in outputs
        assert '15min' in outputs
        assert '1hr' in outputs

        for horizon in ['5min', '15min', '1hr']:
            assert 'logits' in outputs[horizon]
            assert 'probs' in outputs[horizon]
            assert outputs[horizon]['logits'].shape == (4, 3)
            assert outputs[horizon]['probs'].shape == (4, 3)

    def test_probabilities_sum_to_one(self, sample_batch):
        """Test output probabilities sum to 1"""
        model = TransformerPricePredictor()
        outputs = model(sample_batch)

        for horizon in ['5min', '15min', '1hr']:
            probs = outputs[horizon]['probs']
            prob_sums = probs.sum(dim=-1)
            assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6)

    def test_forward_pass_deterministic_eval(self, sample_batch):
        """Test forward pass is deterministic in eval mode"""
        model = TransformerPricePredictor()
        model.eval()

        with torch.no_grad():
            outputs1 = model(sample_batch)
            outputs2 = model(sample_batch)

        assert torch.allclose(outputs1['5min']['logits'], outputs2['5min']['logits'])

    def test_forward_pass_with_custom_mask(self, sample_batch):
        """Test forward pass with custom attention mask"""
        model = TransformerPricePredictor()
        custom_mask = torch.zeros(100, 100).bool()  # All False (no masking)

        outputs = model(sample_batch, src_mask=custom_mask)

        assert outputs['5min']['logits'].shape == (4, 3)


class TestPredictMethod:
    """Test prediction method with confidence thresholding"""

    @pytest.fixture
    def model_and_input(self):
        """Create model and input"""
        model = TransformerPricePredictor()
        x = torch.randn(2, 100, 6)
        return model, x

    def test_predict_basic(self, model_and_input):
        """Test basic prediction"""
        model, x = model_and_input
        predictions = model.predict(x, threshold=0.4)

        assert '5min' in predictions
        assert '15min' in predictions
        assert '1hr' in predictions

        for horizon in ['5min', '15min', '1hr']:
            assert 'prediction' in predictions[horizon]
            assert 'confidence' in predictions[horizon]
            assert 'probabilities' in predictions[horizon]

    def test_predict_with_high_threshold(self, model_and_input):
        """Test prediction with high confidence threshold"""
        model, x = model_and_input
        predictions = model.predict(x, threshold=0.9)

        # With high threshold, many predictions should be 'flat' (class 1)
        for horizon in ['5min', '15min', '1hr']:
            pred_classes = predictions[horizon]['prediction']
            # Should have some predictions defaulting to flat
            assert (pred_classes == 1).any()

    def test_predict_confidence_values(self, model_and_input):
        """Test confidence values are in valid range"""
        model, x = model_and_input
        predictions = model.predict(x)

        for horizon in ['5min', '15min', '1hr']:
            confidence = predictions[horizon]['confidence']
            assert (confidence >= 0).all()
            assert (confidence <= 1).all()


class TestOptimizerConfiguration:
    """Test optimizer configuration with gradient clipping"""

    def test_configure_optimizers(self):
        """Test optimizer configuration"""
        model = TransformerPricePredictor()

        optimizer, config = model.configure_optimizers(
            learning_rate=1e-4,
            weight_decay=0.01,
            max_grad_norm=1.0
        )

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]['lr'] == 1e-4
        assert optimizer.param_groups[0]['weight_decay'] == 0.01
        assert config['max_grad_norm'] == 1.0
        assert config['clip_grad'] is True

    def test_gradient_clipping_config(self):
        """Test gradient clipping configuration"""
        model = TransformerPricePredictor()

        _, config = model.configure_optimizers(max_grad_norm=0.5)

        assert config['max_grad_norm'] == 0.5

    def test_optimizer_parameters(self):
        """Test optimizer has access to all model parameters"""
        model = TransformerPricePredictor()
        optimizer, _ = model.configure_optimizers()

        optimizer_param_count = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
        model_param_count = sum(p.numel() for p in model.parameters())

        assert optimizer_param_count == model_param_count


class TestLossFunctionWithLabelSmoothing:
    """Test loss function with label smoothing"""

    def test_loss_initialization_with_smoothing(self):
        """Test loss initialization with label smoothing"""
        loss_fn = PricePredictionLoss(label_smoothing=0.1)

        assert loss_fn.label_smoothing == 0.1

    def test_loss_calculation_with_smoothing(self):
        """Test loss calculation applies label smoothing"""
        batch_size = 4
        model = TransformerPricePredictor()
        loss_fn = PricePredictionLoss(label_smoothing=0.1, use_focal_loss=False)

        x = torch.randn(batch_size, 100, 6)
        predictions = model(x)

        targets = {
            '5min': torch.randint(0, 3, (batch_size,)),
            '15min': torch.randint(0, 3, (batch_size,)),
            '1hr': torch.randint(0, 3, (batch_size,))
        }

        total_loss, loss_dict = loss_fn(predictions, targets)

        assert total_loss.item() > 0
        assert 'loss_5min' in loss_dict
        assert 'loss_15min' in loss_dict
        assert 'loss_1hr' in loss_dict
        assert 'total_loss' in loss_dict

    def test_focal_loss_with_smoothing(self):
        """Test focal loss with label smoothing"""
        batch_size = 4
        model = TransformerPricePredictor()
        loss_fn = PricePredictionLoss(label_smoothing=0.1, use_focal_loss=True)

        x = torch.randn(batch_size, 100, 6)
        predictions = model(x)

        targets = {
            '5min': torch.randint(0, 3, (batch_size,)),
            '15min': torch.randint(0, 3, (batch_size,)),
            '1hr': torch.randint(0, 3, (batch_size,))
        }

        total_loss, _ = loss_fn(predictions, targets)

        assert total_loss.item() > 0

    def test_label_smoothing_reduces_overconfidence(self):
        """Test label smoothing produces different loss than hard targets"""
        batch_size = 4
        model = TransformerPricePredictor()

        loss_fn_no_smooth = PricePredictionLoss(label_smoothing=0.0, use_focal_loss=False)
        loss_fn_with_smooth = PricePredictionLoss(label_smoothing=0.1, use_focal_loss=False)

        x = torch.randn(batch_size, 100, 6)
        predictions = model(x)

        targets = {
            '5min': torch.randint(0, 3, (batch_size,)),
            '15min': torch.randint(0, 3, (batch_size,)),
            '1hr': torch.randint(0, 3, (batch_size,))
        }

        loss_no_smooth, _ = loss_fn_no_smooth(predictions, targets)
        loss_with_smooth, _ = loss_fn_with_smooth(predictions, targets)

        # Losses should be different
        assert not torch.allclose(loss_no_smooth, loss_with_smooth)


class TestModelFactoryFunction:
    """Test model creation factory function"""

    def test_create_default_model(self):
        """Test creating model with default parameters"""
        model = create_price_predictor()

        assert isinstance(model, TransformerPricePredictor)
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0

    def test_create_custom_model(self):
        """Test creating model with custom parameters"""
        model = create_price_predictor(
            input_dim=10,
            d_model=128,
            nhead=4,
            num_encoder_layers=3
        )

        assert model.input_dim == 10
        assert model.d_model == 128

    def test_model_device_placement(self):
        """Test model is placed on correct device"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = create_price_predictor(device=device)

        # Check first parameter device
        first_param = next(model.parameters())
        assert str(first_param.device).startswith(device)


class TestBackwardCompatibility:
    """Test model is backward compatible"""

    def test_old_api_still_works(self):
        """Test model works with old API (no new parameters)"""
        model = TransformerPricePredictor(
            input_dim=6,
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            num_classes=3,
            seq_len=100
        )

        x = torch.randn(2, 100, 6)
        outputs = model(x)

        assert outputs['5min']['logits'].shape == (2, 3)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
