# AI-to-AI Validation Protocol

## Version 1.0.0
**Last Updated**: 2025-10-11
**Status**: Production Ready

---

## Table of Contents
1. [Overview](#overview)
2. [Protocol Architecture](#protocol-architecture)
3. [Message Format Specification](#message-format-specification)
4. [Communication Flow](#communication-flow)
5. [Error Handling](#error-handling)
6. [Performance Requirements](#performance-requirements)
7. [Security](#security)
8. [Examples](#examples)

---

## Overview

### Purpose
This protocol defines how AI systems communicate with the AI Validation system to ensure decision quality, prevent hallucinations, and maintain system integrity.

### Participants
- **Decision-Making AI**: Any AI system making trading decisions (neural networks, RL agents, etc.)
- **Validation AI**: The AI Psychology Team's validation system
- **Audit Logger**: Immutable audit trail system

### Design Principles
1. **Low Latency**: Validation must complete in <10ms (p95)
2. **Non-Blocking**: Trading systems can operate if validation is unavailable (with safety limits)
3. **Comprehensive**: All decision context must be captured
4. **Immutable**: All validations are logged immutably
5. **Explainable**: All rejections must have clear reasoning

---

## Protocol Architecture

```
┌─────────────────────┐
│  Decision-Making AI │
│  (Neural Network)   │
└──────────┬──────────┘
           │
           │ 1. Validation Request
           ↓
┌─────────────────────┐
│   Validation API    │
│   (REST/gRPC)       │
└──────────┬──────────┘
           │
           │ 2. Multi-Layer Validation
           ↓
┌─────────────────────┐      ┌──────────────────┐
│   AI Validator      │──────▶│ Hallucination    │
│   (Coordinator)     │      │ Detector         │
└──────────┬──────────┘      └──────────────────┘
           │                  ┌──────────────────┐
           │──────────────────▶│ Data Authenticity│
           │                  │ Validator        │
           │                  └──────────────────┘
           │                  ┌──────────────────┐
           │──────────────────▶│ Logic Coherence  │
           │                  │ Validator        │
           │                  └──────────────────┘
           │
           │ 3. Validation Response
           ↓
┌─────────────────────┐
│  Decision Auditor   │
│  (Immutable Log)    │
└─────────────────────┘
```

---

## Message Format Specification

### 1. Validation Request

```json
{
  "request_id": "uuid-v4",
  "timestamp": "2025-10-11T10:30:00.123Z",
  "request_type": "TRADE_DECISION",

  "model_info": {
    "model_name": "transformer_v2.1.0",
    "model_version": "2.1.0",
    "model_type": "transformer",
    "training_date": "2025-10-01",
    "last_validation_date": "2025-10-10"
  },

  "decision": {
    "decision_id": "dec_abc123",
    "decision_type": "BUY",
    "symbol": "BTC-USD",
    "quantity": 0.001,
    "price": 50000.00,
    "confidence": 0.72,
    "urgency": "NORMAL",
    "timeout_ms": 50
  },

  "inputs": {
    "features": [0.23, -0.45, 0.67, ...],
    "feature_names": ["rsi", "macd", "volume_ratio", ...],
    "current_price": 50000.00,
    "historical_prices": [49800, 49900, 50000, ...],
    "order_book": {
      "bids": [[49990, 1.5], [49980, 2.0], ...],
      "asks": [[50010, 1.2], [50020, 1.8], ...]
    },
    "market_context": {
      "volatility": 0.025,
      "volume_24h": 15000000000,
      "trend": "upward",
      "regime": "bull"
    }
  },

  "reasoning": {
    "primary_signal": "RSI indicates oversold condition",
    "supporting_signals": [
      "Bullish divergence on 4H chart",
      "Positive sentiment score: 0.65",
      "Strong support at 49500"
    ],
    "feature_importance": {
      "rsi": 0.35,
      "macd": 0.25,
      "sentiment": 0.20,
      "volume": 0.15,
      "other": 0.05
    },
    "alternative_outcomes": [
      {
        "scenario": "price_continues_down",
        "probability": 0.28,
        "action": "stop_loss_at_49000"
      }
    ]
  },

  "ensemble_predictions": [
    {"model": "transformer_1", "prediction": 51000, "confidence": 0.70},
    {"model": "transformer_2", "prediction": 51200, "confidence": 0.75},
    {"model": "lstm_1", "prediction": 50800, "confidence": 0.68}
  ],

  "data_sources": [
    {
      "source_id": "coinbase_websocket",
      "source_url": "wss://ws-feed.exchange.coinbase.com",
      "data_type": "price",
      "timestamp": "2025-10-11T10:29:59.950Z",
      "latency_ms": 50,
      "checksum": "sha256:abc123..."
    },
    {
      "source_id": "tradingview_indicators",
      "source_url": "https://api.tradingview.com/v1/indicators",
      "data_type": "technical_indicators",
      "timestamp": "2025-10-11T10:29:55.000Z",
      "latency_ms": 5000,
      "checksum": "sha256:def456..."
    }
  ],

  "risk_assessment": {
    "expected_value": 150.00,
    "max_loss": 2000.00,
    "probability_success": 0.72,
    "kelly_fraction": 0.015,
    "position_size_usd": 50.00,
    "portfolio_exposure": 0.05,
    "risk_reward_ratio": 3.0
  }
}
```

### 2. Validation Response

```json
{
  "request_id": "uuid-v4",
  "response_timestamp": "2025-10-11T10:30:00.150Z",
  "processing_time_ms": 8.5,

  "validation_status": "APPROVED",
  "execution_allowed": true,
  "confidence": 0.96,

  "validations": {
    "hallucination_check": {
      "passed": true,
      "confidence": 0.99,
      "layer_results": {
        "statistical_plausibility": {"passed": true, "confidence": 1.0},
        "historical_consistency": {"passed": true, "confidence": 0.98},
        "ensemble_agreement": {"passed": true, "confidence": 0.95},
        "logical_coherence": {"passed": true, "confidence": 0.99},
        "source_attribution": {"passed": true, "confidence": 1.0}
      },
      "details": "All hallucination checks passed"
    },

    "data_authenticity": {
      "passed": true,
      "confidence": 0.98,
      "details": "All data sources verified and authenticated",
      "source_scores": {
        "coinbase_websocket": 1.0,
        "tradingview_indicators": 0.95
      }
    },

    "decision_logic": {
      "passed": true,
      "confidence": 0.95,
      "details": "Decision logic is sound and explainable",
      "coherence_score": 0.92,
      "explainability_score": 0.88
    },

    "adversarial_robustness": {
      "passed": true,
      "confidence": 0.92,
      "details": "No adversarial patterns detected",
      "perturbation_tests_passed": 15
    },

    "confidence_calibration": {
      "passed": true,
      "confidence": 0.90,
      "details": "Model confidence is well-calibrated",
      "expected_calibration_error": 0.03
    }
  },

  "concerns": [],

  "recommendations": [
    "Consider reducing position size by 20% due to elevated market volatility"
  ],

  "audit_info": {
    "audit_entry_id": "audit_xyz789",
    "audit_logged": true,
    "audit_hash": "sha256:fedcba987..."
  }
}
```

### 3. Rejection Response

```json
{
  "request_id": "uuid-v4",
  "response_timestamp": "2025-10-11T10:30:00.145Z",
  "processing_time_ms": 7.2,

  "validation_status": "REJECTED",
  "execution_allowed": false,
  "confidence": 0.98,

  "rejection_reason": "CRITICAL_HALLUCINATION_DETECTED",
  "rejection_details": "Prediction is 8.5-sigma outlier, exceeds plausibility threshold",

  "validations": {
    "hallucination_check": {
      "passed": false,
      "confidence": 0.98,
      "layer_results": {
        "statistical_plausibility": {
          "passed": false,
          "confidence": 0.98,
          "failure_reason": "Prediction is 8.5-sigma outlier",
          "evidence": {
            "prediction": 75000,
            "current_price": 50000,
            "historical_mean": 50200,
            "historical_std": 2000,
            "z_score": 8.5,
            "threshold": 5.0
          }
        }
      }
    }
  },

  "concerns": [
    {
      "severity": "CRITICAL",
      "type": "STATISTICAL_OUTLIER",
      "description": "Prediction exceeds 5-sigma threshold"
    }
  ],

  "recommendations": [
    "Review model for potential bugs or data quality issues",
    "Retrain model with recent data",
    "Use ensemble median instead of single model prediction"
  ],

  "audit_info": {
    "audit_entry_id": "audit_xyz789",
    "audit_logged": true,
    "audit_hash": "sha256:fedcba987..."
  }
}
```

---

## Communication Flow

### Synchronous Flow (Default)

```
Decision AI                    Validation API                 AI Validator
    │                                │                              │
    │  1. POST /validate/decision    │                              │
    ├────────────────────────────────▶                              │
    │                                │  2. Validate Request         │
    │                                ├──────────────────────────────▶
    │                                │                              │
    │                                │  3. Multi-Layer Validation   │
    │                                │                              │
    │                                │  4. Validation Response      │
    │                                ◀──────────────────────────────┤
    │  5. Response                   │                              │
    ◀────────────────────────────────┤                              │
    │                                │                              │
    │  6. Execute if approved        │                              │
    │                                │                              │
```

### Asynchronous Flow (For Batch Validation)

```
Decision AI                    Validation API                 AI Validator
    │                                │                              │
    │  1. POST /validate/batch       │                              │
    ├────────────────────────────────▶                              │
    │  2. Return job_id immediately  │                              │
    ◀────────────────────────────────┤                              │
    │                                │  3. Async Validation         │
    │                                ├──────────────────────────────▶
    │                                │                              │
    │  4. Poll: GET /validate/job_id │                              │
    ├────────────────────────────────▶                              │
    │  5. Return results when ready  │                              │
    ◀────────────────────────────────┤                              │
    │                                │                              │
```

---

## Error Handling

### Timeout Handling
```json
{
  "error_code": "VALIDATION_TIMEOUT",
  "error_message": "Validation exceeded 50ms timeout",
  "default_action": "REJECT",
  "recommendation": "Retry with higher timeout or check validator health"
}
```

### Validator Unavailable
```json
{
  "error_code": "VALIDATOR_UNAVAILABLE",
  "error_message": "Validation service is unavailable",
  "default_action": "ALLOW_WITH_CAUTION",
  "fallback_validations": ["basic_sanity_checks"],
  "recommendation": "Monitor closely and reduce position sizes"
}
```

### Invalid Request
```json
{
  "error_code": "INVALID_REQUEST",
  "error_message": "Missing required field: 'inputs.current_price'",
  "default_action": "REJECT",
  "recommendation": "Fix request format and retry"
}
```

---

## Performance Requirements

### Latency Targets
- **p50**: < 5ms
- **p95**: < 10ms
- **p99**: < 50ms
- **Timeout**: 100ms (hard limit)

### Throughput
- **Minimum**: 10,000 validations/second
- **Target**: 50,000 validations/second
- **Peak**: 100,000 validations/second

### Availability
- **Uptime**: 99.99% during market hours
- **Degraded Mode**: System continues with basic validations if primary validator fails

---

## Security

### Authentication
- API key authentication for all requests
- JWT tokens for internal service-to-service communication
- Rate limiting: 10,000 requests per second per API key

### Data Protection
- All communication over TLS 1.3
- Sensitive data (API keys, credentials) never logged
- Audit logs encrypted at rest

### Audit Trail
- All validations logged immutably
- Cryptographic hash chaining prevents tampering
- Retention: 7 years for regulatory compliance

---

## Examples

### Example 1: Successful Trade Validation

```bash
curl -X POST https://api.rrr-trading.com/v1/validate/decision \
  -H "Authorization: Bearer <api_key>" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "req_123",
    "timestamp": "2025-10-11T10:30:00Z",
    "decision": {
      "decision_type": "BUY",
      "symbol": "BTC-USD",
      "quantity": 0.001,
      "price": 50000,
      "confidence": 0.75
    },
    "inputs": {
      "current_price": 50000,
      "historical_prices": [49800, 49900, 50000]
    }
  }'
```

**Response:**
```json
{
  "request_id": "req_123",
  "validation_status": "APPROVED",
  "execution_allowed": true,
  "confidence": 0.95,
  "processing_time_ms": 8.2
}
```

### Example 2: Rejected Due to Hallucination

```bash
# Same request but with impossible prediction
curl -X POST https://api.rrr-trading.com/v1/validate/decision \
  -H "Authorization: Bearer <api_key>" \
  -H "Content-Type: application/json" \
  -d '{
    "decision": {
      "decision_type": "BUY",
      "predicted_price": 150000,
      "current_price": 50000
    }
  }'
```

**Response:**
```json
{
  "validation_status": "REJECTED",
  "execution_allowed": false,
  "rejection_reason": "STATISTICAL_OUTLIER",
  "rejection_details": "Prediction is 50% jump, exceeds 10x volatility threshold",
  "concerns": [
    {
      "severity": "HIGH",
      "type": "UNREALISTIC_PREDICTION"
    }
  ]
}
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-10-11 | Initial protocol specification |

---

## Contact

- **Team**: AI Psychology Team
- **Email**: ai-validation@rrrventures.com
- **Slack**: #ai-psychology-team
- **On-Call**: PagerDuty rotation

---

**Document Status**: Production Ready
**Next Review**: 2025-11-11
