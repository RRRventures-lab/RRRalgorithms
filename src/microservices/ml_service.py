from datetime import datetime
from fastapi import FastAPI, HTTPException
from functools import lru_cache
from pydantic import BaseModel
from src.core.memory_cache import get_cache
from src.core.redis_cache import get_redis_cache
from src.neural_network.production_predictor import ProductionPredictor, PredictionResult
from typing import Dict, List, Optional, Any
import asyncio
import json
import logging


"""
ML Service
==========

Microservice for machine learning predictions and model management.
Handles model inference, feature engineering, and prediction caching.

Author: RRR Ventures
Date: 2025-10-12
"""




class PredictionRequest(BaseModel):
    """Prediction request model."""
    symbol: str
    market_data: Dict[str, Any]
    horizon_minutes: int = 60


class PredictionResponse(BaseModel):
    """Prediction response model."""
    symbol: str
    predicted_price: float
    predicted_direction: str
    confidence: float
    model_version: str
    timestamp: float
    prediction_horizon: int


class MLService:
    """
    ML service for machine learning predictions.
    
    Features:
    - Model inference and predictions
    - Feature engineering
    - Prediction caching
    - Model versioning
    - Performance monitoring
    """
    
    def __init__(self, port: int = 8002):
        """
        Initialize ML service.
        
        Args:
            port: Service port
        """
        self.port = port
        self.app = FastAPI(title="ML Service", version="1.0.0")
        
        # ML components
        self.predictor = None
        self.redis_cache = None
        self.memory_cache = None
        
        # Performance metrics
        self.metrics = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_inference_time': 0.0,
            'max_inference_time': 0.0,
            'min_inference_time': float('inf')
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """Setup FastAPI routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Service health check."""
            return {
                "status": "healthy",
                "service": "ml-service",
                "timestamp": datetime.now().isoformat(),
                "metrics": self.metrics
            }
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get service metrics."""
            return self.metrics
        
        @self.app.get("/models/status")
        async def get_model_status():
            """Get model status."""
            if self.predictor:
                return self.predictor.get_status()
            return {"error": "Predictor not initialized"}
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            """Make prediction for a symbol."""
            try:
                # Check cache first
                cache_key = f"prediction:{request.symbol}:{request.horizon_minutes}"
                cached_prediction = None
                
                if self.memory_cache:
                    cached_prediction = self.memory_cache.get(cache_key)
                    if cached_prediction:
                        self.metrics['cache_hits'] += 1
                        return PredictionResponse(**cached_prediction)
                
                if self.redis_cache:
                    cached_prediction = await self.redis_cache.get_cached_prediction(cache_key)
                    if cached_prediction:
                        self.metrics['cache_hits'] += 1
                        return PredictionResponse(**cached_prediction)
                
                # Make prediction
                self.metrics['cache_misses'] += 1
                start_time = asyncio.get_event_loop().time()
                
                prediction = await self.predictor.predict(
                    symbol=request.symbol,
                    market_data=request.market_data,
                    horizon_minutes=request.horizon_minutes
                )
                
                inference_time = asyncio.get_event_loop().time() - start_time
                self._update_metrics(inference_time, True)
                
                # Convert to response
                response = PredictionResponse(
                    symbol=prediction.symbol,
                    predicted_price=prediction.predicted_price,
                    predicted_direction=prediction.predicted_direction,
                    confidence=prediction.confidence,
                    model_version=prediction.model_version,
                    timestamp=prediction.timestamp,
                    prediction_horizon=prediction.prediction_horizon
                )
                
                # Cache prediction
                prediction_dict = response.dict()
                if self.memory_cache:
                    self.memory_cache.set(cache_key, prediction_dict, ttl=300)
                
                if self.redis_cache:
                    await self.redis_cache.cache_prediction(
                        cache_key,
                        prediction_dict,
                        ttl=600
                    )
                
                return response
                
            except Exception as e:
                self.logger.error(f"Prediction failed for {request.symbol}: {e}")
                self._update_metrics(0, False)
                raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
        @self.app.post("/predict/batch")
        async def predict_batch(requests: List[PredictionRequest]):
            """Make batch predictions."""
            try:
                results = []
                
                # Process predictions in parallel
                tasks = []
                for request in requests:
                    task = asyncio.create_task(self._predict_single(request))
                    tasks.append(task)
                
                predictions = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, prediction in enumerate(predictions):
                    if isinstance(prediction, Exception):
                        self.logger.error(f"Batch prediction {i} failed: {prediction}")
                        results.append({
                            "error": str(prediction),
                            "symbol": requests[i].symbol
                        })
                    else:
                        results.append(prediction.dict())
                
                return {"predictions": results}
                
            except Exception as e:
                self.logger.error(f"Batch prediction failed: {e}")
                raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
        
        @self.app.get("/predictions/{symbol}")
        async def get_predictions(symbol: str, limit: int = 10):
            """Get recent predictions for a symbol."""
            try:
                # Try cache first
                cache_key = f"predictions:{symbol}"
                cached_predictions = None
                
                if self.memory_cache:
                    cached_predictions = self.memory_cache.get(cache_key)
                
                if cached_predictions:
                    return {"predictions": cached_predictions[:limit]}
                
                # Return empty if no cached predictions
                return {"predictions": []}
                
            except Exception as e:
                self.logger.error(f"Error getting predictions for {symbol}: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
    
    async def _predict_single(self, request: PredictionRequest) -> PredictionResponse:
        """Make a single prediction."""
        # Check cache first
        cache_key = f"prediction:{request.symbol}:{request.horizon_minutes}"
        
        if self.memory_cache:
            cached_prediction = self.memory_cache.get(cache_key)
            if cached_prediction:
                self.metrics['cache_hits'] += 1
                return PredictionResponse(**cached_prediction)
        
        # Make prediction
        self.metrics['cache_misses'] += 1
        start_time = asyncio.get_event_loop().time()
        
        prediction = await self.predictor.predict(
            symbol=request.symbol,
            market_data=request.market_data,
            horizon_minutes=request.horizon_minutes
        )
        
        inference_time = asyncio.get_event_loop().time() - start_time
        self._update_metrics(inference_time, True)
        
        # Convert to response
        response = PredictionResponse(
            symbol=prediction.symbol,
            predicted_price=prediction.predicted_price,
            predicted_direction=prediction.predicted_direction,
            confidence=prediction.confidence,
            model_version=prediction.model_version,
            timestamp=prediction.timestamp,
            prediction_horizon=prediction.prediction_horizon
        )
        
        # Cache prediction
        prediction_dict = response.dict()
        if self.memory_cache:
            self.memory_cache.set(cache_key, prediction_dict, ttl=300)
        
        return response
    
    async def initialize(self) -> None:
        """Initialize ML service components."""
        try:
            # Initialize ML predictor
            self.predictor = ProductionPredictor()
            await self.predictor.initialize()
            
            # Initialize caches
            self.redis_cache = await get_redis_cache()
            self.memory_cache = get_cache()
            
            self.logger.info("ML service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ML service: {e}")
            raise
    
    def _update_metrics(self, inference_time: float, success: bool) -> None:
        """Update performance metrics."""
        self.metrics['total_predictions'] += 1
        
        if success:
            self.metrics['successful_predictions'] += 1
        else:
            self.metrics['failed_predictions'] += 1
        
        # Update timing metrics
        if success and inference_time > 0:
            if self.metrics['total_predictions'] == 1:
                self.metrics['avg_inference_time'] = inference_time
            else:
                # Exponential moving average
                alpha = 0.1
                self.metrics['avg_inference_time'] = (
                    alpha * inference_time + 
                    (1 - alpha) * self.metrics['avg_inference_time']
                )
            
            self.metrics['max_inference_time'] = max(self.metrics['max_inference_time'], inference_time)
            self.metrics['min_inference_time'] = min(self.metrics['min_inference_time'], inference_time)
    
    async def start(self) -> None:
        """Start the ML service."""
        self.logger.info(f"Starting ML Service on port {self.port}")
        
        # Initialize components
        await self.initialize()
        
        # Start FastAPI server
        import uvicorn
        config = uvicorn.Config(
            app=self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    @lru_cache(maxsize=128)
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            'service': 'ml-service',
            'port': self.port,
            'metrics': self.metrics
        }


# Global ML service instance
_ml_service: Optional[MLService] = None


@lru_cache(maxsize=128)


def get_ml_service() -> MLService:
    """Get the global ML service instance."""
    global _ml_service
    
    if _ml_service is None:
        _ml_service = MLService()
    
    return _ml_service


if __name__ == "__main__":
    # Run ML service
    service = MLService()
    asyncio.run(service.start())