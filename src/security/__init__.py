"""
Security module for RRRalgorithms
Handles secrets management, encryption, authentication, and audit logging
"""

from .keychain_manager import KeychainManager
from .secrets_manager import SecretsManager, get_secrets_manager
from .auth import JWTManager, get_jwt_manager, get_current_user, get_current_active_user, get_current_admin_user, require_scopes
from .middleware import SecurityHeadersMiddleware, RateLimitMiddleware, AuditLoggingMiddleware


__all__ = [
    'SecretsManager',
    'get_secrets_manager',
    'KeychainManager',
    'JWTManager',
    'get_jwt_manager',
    'get_current_user',
    'get_current_active_user',
    'get_current_admin_user',
    'require_scopes',
    'SecurityHeadersMiddleware',
    'RateLimitMiddleware',
    'AuditLoggingMiddleware'
]
