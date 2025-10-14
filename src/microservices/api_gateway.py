from dataclasses import dataclass
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from functools import lru_cache
from pydantic import BaseModel
from src.core.exceptions import AuthenticationError, AuthorizationError, APIError
from typing import Dict, List, Optional, Any, Callable
import asyncio
import httpx
import json
import jwt
import logging
import time


"""
API Gateway
===========

High-performance API gateway with authentication, rate limiting,
load balancing, and service discovery for microservices architecture.

Author: RRR Ventures
Date: 2025-10-12
"""




@dataclass
class ServiceEndpoint:
    """Service endpoint configuration."""
    name: str
    url: str
    health_check_url: str
    weight: int = 1
    max_connections: int = 100
    timeout: float = 30.0
    retry_attempts: int = 3


@dataclass
class RateLimit:
    """Rate limiting configuration."""
    requests_per_minute: int = 60
    burst_size: int = 10
    window_size: int = 60  # seconds


class ServiceRegistry:
    """Service registry for service discovery."""
    
    def __init__(self):
        """Initialize service registry."""
        self.services: Dict[str, List[ServiceEndpoint]] = {}
        self.health_status: Dict[str, bool] = {}
        self.load_balancers: Dict[str, int] = {}  # Round-robin counters
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def register_service(self, service_name: str, endpoint: ServiceEndpoint) -> None:
        """Register a service endpoint."""
        if service_name not in self.services:
            self.services[service_name] = []
        
        self.services[service_name].append(endpoint)
        self.health_status[f"{service_name}:{endpoint.url}"] = True
        self.load_balancers[service_name] = 0
        
        self.logger.info(f"Registered service {service_name} at {endpoint.url}")
    
    @lru_cache(maxsize=128)
    
    def get_service_endpoint(self, service_name: str) -> Optional[ServiceEndpoint]:
        """Get a healthy service endpoint using round-robin load balancing."""
        if service_name not in self.services:
            return None
        
        healthy_endpoints = [
            ep for ep in self.services[service_name]
            if self.health_status.get(f"{service_name}:{ep.url}", False)
        ]
        
        if not healthy_endpoints:
            return None
        
        # Round-robin selection
        index = self.load_balancers[service_name] % len(healthy_endpoints)
        self.load_balancers[service_name] += 1
        
        return healthy_endpoints[index]
    
    async def health_check(self, service_name: str, endpoint: ServiceEndpoint) -> bool:
        """Perform health check on service endpoint."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(endpoint.health_check_url)
                is_healthy = response.status_code == 200
                
                self.health_status[f"{service_name}:{endpoint.url}"] = is_healthy
                return is_healthy
                
        except Exception as e:
            self.logger.warning(f"Health check failed for {service_name}: {e}")
            self.health_status[f"{service_name}:{endpoint.url}"] = False
            return False
    
    async def check_all_services(self) -> None:
        """Check health of all registered services."""
        tasks = []
        for service_name, endpoints in self.services.items():
            for endpoint in endpoints:
                task = asyncio.create_task(
                    self.health_check(service_name, endpoint)
                )
                tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    @lru_cache(maxsize=128)
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services."""
        return {
            'services': {
                name: {
                    'endpoints': len(endpoints),
                    'healthy_endpoints': len([
                        ep for ep in endpoints
                        if self.health_status.get(f"{name}:{ep.url}", False)
                    ])
                }
                for name, endpoints in self.services.items()
            },
            'health_status': self.health_status
        }


class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self):
        """Initialize rate limiter."""
        self.requests: Dict[str, List[float]] = {}
        self.rate_limits: Dict[str, RateLimit] = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def set_rate_limit(self, key: str, rate_limit: RateLimit) -> None:
        """Set rate limit for a key."""
        self.rate_limits[key] = rate_limit
    
    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed based on rate limit."""
        if key not in self.rate_limits:
            return True  # No rate limit set
        
        rate_limit = self.rate_limits[key]
        now = time.time()
        
        # Clean old requests
        if key in self.requests:
            self.requests[key] = [
                req_time for req_time in self.requests[key]
                if now - req_time < rate_limit.window_size
            ]
        else:
            self.requests[key] = []
        
        # Check if under limit
        if len(self.requests[key]) < rate_limit.requests_per_minute:
            self.requests[key].append(now)
            return True
        
        return False
    
    @lru_cache(maxsize=128)
    
    def get_remaining_requests(self, key: str) -> int:
        """Get remaining requests for a key."""
        if key not in self.rate_limits:
            return float('inf')
        
        rate_limit = self.rate_limits[key]
        now = time.time()
        
        if key in self.requests:
            self.requests[key] = [
                req_time for req_time in self.requests[key]
                if now - req_time < rate_limit.window_size
            ]
            return max(0, rate_limit.requests_per_minute - len(self.requests[key]))
        
        return rate_limit.requests_per_minute


class APIGateway:
    """
    High-performance API gateway for microservices.
    
    Features:
    - Service discovery and load balancing
    - Authentication and authorization
    - Rate limiting and throttling
    - Request/response logging
    - Health checks and monitoring
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        jwt_secret: str = "your-secret-key",
        jwt_algorithm: str = "HS256"
    ):
        """
        Initialize API gateway.
        
        Args:
            host: Gateway host
            port: Gateway port
            jwt_secret: JWT secret key
            jwt_algorithm: JWT algorithm
        """
        self.host = host
        self.port = port
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        
        # Initialize components
        self.service_registry = ServiceRegistry()
        self.rate_limiter = RateLimiter()
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Setup FastAPI app
        self.app = FastAPI(
            title="RRRalgorithms API Gateway",
            description="High-performance API gateway for trading microservices",
            version="1.0.0"
        )
        
        # Setup middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limited_requests': 0,
            'avg_response_time': 0.0,
            'max_response_time': 0.0,
            'min_response_time': float('inf')
        }
    
    def _setup_middleware(self) -> None:
        """Setup FastAPI middleware."""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Trusted host middleware
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]
        )
        
        # Request logging middleware
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()
            
            # Rate limiting
            client_ip = request.client.host
            if not self.rate_limiter.is_allowed(client_ip):
                self.metrics['rate_limited_requests'] += 1
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded"}
                )
            
            # Process request
            response = await call_next(request)
            
            # Update metrics
            process_time = time.time() - start_time
            self._update_metrics(process_time, response.status_code < 400)
            
            # Add response headers
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Rate-Limit-Remaining"] = str(
                self.rate_limiter.get_remaining_requests(client_ip)
            )
            
            return response
    
    def _setup_routes(self) -> None:
        """Setup API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Gateway health check."""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "services": self.service_registry.get_service_status()
            }
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get gateway metrics."""
            return self.metrics
        
        @self.app.post("/auth/login")
        async def login(credentials: dict):
            """Authenticate user and return JWT token."""
            username = credentials.get("username")
            password = credentials.get("password")
            
            # Simple authentication (in production, use proper auth service)
            if username == "admin" and password == "password":
                token = jwt.encode(
                    {
                        "username": username,
                        "exp": datetime.utcnow() + timedelta(hours=24)
                    },
                    self.jwt_secret,
                    algorithm=self.jwt_algorithm
                )
                return {"access_token": token, "token_type": "bearer"}
            
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        @self.app.get("/services/{service_name}/{path:path}")
        async def proxy_request(
            service_name: str,
            path: str,
            request: Request,
            credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
        ):
            """Proxy request to microservice."""
            # Verify JWT token
            try:
                payload = jwt.decode(
                    credentials.credentials,
                    self.jwt_secret,
                    algorithms=[self.jwt_algorithm]
                )
                username = payload.get("username")
            except jwt.InvalidTokenError:
                raise HTTPException(status_code=401, detail="Invalid token")
            
            # Get service endpoint
            endpoint = self.service_registry.get_service_endpoint(service_name)
            if not endpoint:
                raise HTTPException(status_code=503, detail="Service unavailable")
            
            # Build target URL
            target_url = f"{endpoint.url}/{path}"
            if request.query_params:
                target_url += f"?{request.query_params}"
            
            # Forward request
            try:
                headers = dict(request.headers)
                headers["X-User"] = username
                
                response = await self.http_client.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    content=await request.body()
                )
                
                return JSONResponse(
                    content=response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
                    status_code=response.status_code,
                    headers=dict(response.headers)
                )
                
            except Exception as e:
                self.logger.error(f"Proxy request failed: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
    
    def register_service(
        self,
        service_name: str,
        url: str,
        health_check_url: str,
        weight: int = 1,
        max_connections: int = 100,
        timeout: float = 30.0
    ) -> None:
        """Register a microservice."""
        endpoint = ServiceEndpoint(
            name=service_name,
            url=url,
            health_check_url=health_check_url,
            weight=weight,
            max_connections=max_connections,
            timeout=timeout
        )
        
        self.service_registry.register_service(service_name, endpoint)
    
    def set_rate_limit(self, key: str, requests_per_minute: int) -> None:
        """Set rate limit for a key."""
        rate_limit = RateLimit(requests_per_minute=requests_per_minute)
        self.rate_limiter.set_rate_limit(key, rate_limit)
    
    def _update_metrics(self, response_time: float, success: bool) -> None:
        """Update performance metrics."""
        self.metrics['total_requests'] += 1
        
        if success:
            self.metrics['successful_requests'] += 1
        else:
            self.metrics['failed_requests'] += 1
        
        # Update timing metrics
        if self.metrics['total_requests'] == 1:
            self.metrics['avg_response_time'] = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics['avg_response_time'] = (
                alpha * response_time + 
                (1 - alpha) * self.metrics['avg_response_time']
            )
        
        self.metrics['max_response_time'] = max(self.metrics['max_response_time'], response_time)
        self.metrics['min_response_time'] = min(self.metrics['min_response_time'], response_time)
    
    async def start_health_checks(self) -> None:
        """Start periodic health checks."""
        while True:
            try:
                await self.service_registry.check_all_services()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(30)
    
    async def start(self) -> None:
        """Start the API gateway."""
        self.logger.info(f"Starting API Gateway on {self.host}:{self.port}")
        
        # Start health checks
        asyncio.create_task(self.start_health_checks())
        
        # Start FastAPI server
        import uvicorn
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    @lru_cache(maxsize=128)
    
    def get_status(self) -> Dict[str, Any]:
        """Get gateway status."""
        return {
            'host': self.host,
            'port': self.port,
            'services': self.service_registry.get_service_status(),
            'metrics': self.metrics
        }


# Global API gateway instance
_gateway: Optional[APIGateway] = None


@lru_cache(maxsize=128)


def get_api_gateway() -> APIGateway:
    """Get the global API gateway instance."""
    global _gateway
    
    if _gateway is None:
        _gateway = APIGateway()
    
    return _gateway


__all__ = [
    'APIGateway',
    'ServiceRegistry',
    'RateLimiter',
    'ServiceEndpoint',
    'get_api_gateway',
]