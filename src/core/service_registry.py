from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Dict, Optional, Any, List
import asyncio
import logging


"""
Service Registry
================

Central registry for all microservices in the trading system.
Manages service lifecycle and dependencies.

Author: RRR Ventures
Date: 2025-10-12
"""


logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class ServiceInfo:
    """Service information."""
    name: str
    status: ServiceStatus
    dependencies: List[str]
    config: Dict[str, Any]
    instance: Optional["BaseService"]


class BaseService(ABC):
    """Abstract base class for all services."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.status = ServiceStatus.STOPPED
        self.logger = logging.getLogger(f"service.{name}")

    @abstractmethod
    async def start(self):
        """Start the service."""
        pass

    @abstractmethod
    async def stop(self):
        """Stop the service."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check service health."""
        pass

    async def restart(self):
        """Restart the service."""
        await self.stop()
        await self.start()


class ServiceRegistry:
    """Central registry for all microservices."""

    def __init__(self):
        self._services: Dict[str, ServiceInfo] = {}
        self._startup_order: List[str] = []
        self.logger = logger

    def register(self,
                 name: str,
                 service_class: type,
                 config: Dict[str, Any],
                 dependencies: Optional[List[str]] = None):
        """Register a service."""
        dependencies = dependencies or []

        # Validate dependencies
        for dep in dependencies:
            if dep not in self._services and dep != name:
                self.logger.warning(f"Service {name} depends on unregistered service {dep}")

        # Create service info
        info = ServiceInfo(
            name=name,
            status=ServiceStatus.STOPPED,
            dependencies=dependencies,
            config=config,
            instance=None
        )

        self._services[name] = info
        self._update_startup_order()

        self.logger.info(f"Registered service: {name}")

    def _update_startup_order(self):
        """Update service startup order based on dependencies."""
        # Simple topological sort
        visited = set()
        order = []

        def visit(name: str):
            if name in visited:
                return
            visited.add(name)

            if name in self._services:
                for dep in self._services[name].dependencies:
                    if dep in self._services:
                        visit(dep)
                order.append(name)

        for name in self._services:
            visit(name)

        self._startup_order = order

    async def start_all(self):
        """Start all registered services in dependency order."""
        self.logger.info("Starting all services...")

        for name in self._startup_order:
            await self.start_service(name)

        self.logger.info("All services started")

    async def start_service(self, name: str):
        """Start a specific service."""
        if name not in self._services:
            raise ValueError(f"Service {name} not registered")

        info = self._services[name]

        if info.status == ServiceStatus.RUNNING:
            self.logger.info(f"Service {name} already running")
            return

        # Start dependencies first
        for dep in info.dependencies:
            if dep in self._services:
                await self.start_service(dep)

        # Create and start service instance
        try:
            info.status = ServiceStatus.STARTING

            # Dynamic import based on service name
            module_name = f"src.services.{name.replace('-', '_')}"
            module = __import__(module_name, fromlist=['Service'])
            service_class = getattr(module, 'Service')

            info.instance = service_class(name, info.config)
            await info.instance.start()

            info.status = ServiceStatus.RUNNING
            self.logger.info(f"Service {name} started successfully")

        except Exception as e:
            info.status = ServiceStatus.ERROR
            self.logger.error(f"Failed to start service {name}: {e}")
            raise

    async def stop_all(self):
        """Stop all services in reverse dependency order."""
        self.logger.info("Stopping all services...")

        for name in reversed(self._startup_order):
            await self.stop_service(name)

        self.logger.info("All services stopped")

    async def stop_service(self, name: str):
        """Stop a specific service."""
        if name not in self._services:
            return

        info = self._services[name]

        if info.status != ServiceStatus.RUNNING:
            return

        try:
            info.status = ServiceStatus.STOPPING

            if info.instance:
                await info.instance.stop()

            info.status = ServiceStatus.STOPPED
            self.logger.info(f"Service {name} stopped")

        except Exception as e:
            self.logger.error(f"Error stopping service {name}: {e}")

    @lru_cache(maxsize=128)

    def get_service(self, name: str) -> Optional[BaseService]:
        """Get a service instance."""
        if name in self._services:
            return self._services[name].instance
        return None

    @lru_cache(maxsize=128)

    def get_status(self, name: Optional[str] = None) -> Dict[str, ServiceStatus]:
        """Get service status."""
        if name:
            if name in self._services:
                return {name: self._services[name].status}
            return {}

        return {name: info.status for name, info in self._services.items()}

    async def health_check(self) -> Dict[str, bool]:
        """Run health checks on all services."""
        results = {}

        for name, info in self._services.items():
            if info.instance and info.status == ServiceStatus.RUNNING:
                try:
                    results[name] = await info.instance.health_check()
                except Exception as e:
                    self.logger.error(f"Health check failed for {name}: {e}")
                    results[name] = False
            else:
                results[name] = False

        return results


# Global registry instance
registry = ServiceRegistry()


@lru_cache(maxsize=128)


def get_registry() -> ServiceRegistry:
    """Get the global service registry."""
    return registry
