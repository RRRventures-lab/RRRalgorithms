"""
JWT Authentication and Authorization
Provides token-based authentication for API endpoints
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import logging

from .secrets_manager import get_secrets_manager

logger = logging.getLogger(__name__)

# Security schemes
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TokenData(BaseModel):
    """JWT Token payload"""
    sub: str  # Subject (user ID)
    exp: datetime  # Expiration
    iat: datetime  # Issued at
    type: str  # Token type (access/refresh)
    scopes: list[str] = []  # Permissions


class User(BaseModel):
    """User model"""
    id: str
    username: str
    email: Optional[str] = None
    scopes: list[str] = []
    is_active: bool = True
    is_admin: bool = False


class JWTManager:
    """
    JWT token manager

    Features:
    - Access token generation
    - Refresh token generation
    - Token validation
    - Token revocation (via blacklist)
    """

    def __init__(self):
        secrets = get_secrets_manager()
        jwt_config = secrets.get_secret("JWT_SECRET")

        if not jwt_config:
            import secrets as sec
            jwt_config = sec.token_urlsafe(64)
            logger.warning("JWT_SECRET not set, using random secret (NOT PRODUCTION SAFE)")

        self.secret_key = jwt_config
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 60
        self.refresh_token_expire_days = 30

        # Token blacklist (in production, use Redis)
        self.blacklisted_tokens: set[str] = set()

    def create_access_token(
        self,
        user_id: str,
        scopes: list[str] = None,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a new access token

        Args:
            user_id: User ID
            scopes: Permission scopes
            expires_delta: Custom expiration time

        Returns:
            JWT token string
        """
        if expires_delta is None:
            expires_delta = timedelta(minutes=self.access_token_expire_minutes)

        expire = datetime.utcnow() + expires_delta
        iat = datetime.utcnow()

        payload = {
            "sub": user_id,
            "exp": expire,
            "iat": iat,
            "type": "access",
            "scopes": scopes or []
        }

        encoded_jwt = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.info(f"Created access token for user {user_id}")
        return encoded_jwt

    def create_refresh_token(
        self,
        user_id: str,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a new refresh token

        Args:
            user_id: User ID
            expires_delta: Custom expiration time

        Returns:
            JWT refresh token string
        """
        if expires_delta is None:
            expires_delta = timedelta(days=self.refresh_token_expire_days)

        expire = datetime.utcnow() + expires_delta
        iat = datetime.utcnow()

        payload = {
            "sub": user_id,
            "exp": expire,
            "iat": iat,
            "type": "refresh",
            "scopes": []
        }

        encoded_jwt = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.info(f"Created refresh token for user {user_id}")
        return encoded_jwt

    def verify_token(self, token: str, expected_type: str = "access") -> TokenData:
        """
        Verify and decode a JWT token

        Args:
            token: JWT token string
            expected_type: Expected token type (access/refresh)

        Returns:
            TokenData object

        Raises:
            HTTPException: If token is invalid or expired
        """
        # Check if token is blacklisted
        if token in self.blacklisted_tokens:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked",
                headers={"WWW-Authenticate": "Bearer"},
            )

        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            user_id: str = payload.get("sub")
            token_type: str = payload.get("type")
            scopes: list = payload.get("scopes", [])

            if user_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            if token_type != expected_type:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid token type: expected {expected_type}",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            return TokenData(
                sub=user_id,
                exp=datetime.fromtimestamp(payload.get("exp")),
                iat=datetime.fromtimestamp(payload.get("iat")),
                type=token_type,
                scopes=scopes
            )

        except JWTError as e:
            logger.warning(f"JWT verification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def revoke_token(self, token: str):
        """Add token to blacklist"""
        self.blacklisted_tokens.add(token)
        logger.info("Token revoked")

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)


# Global JWT manager instance
_jwt_manager: Optional[JWTManager] = None


def get_jwt_manager() -> JWTManager:
    """Get the global JWT manager instance"""
    global _jwt_manager
    if _jwt_manager is None:
        _jwt_manager = JWTManager()
    return _jwt_manager


# FastAPI dependency for authentication
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """
    FastAPI dependency to get current authenticated user

    Usage:
        @app.get("/protected")
        async def protected_route(user: User = Depends(get_current_user)):
            return {"user_id": user.id}
    """
    token = credentials.credentials
    jwt_manager = get_jwt_manager()

    # Verify token
    token_data = jwt_manager.verify_token(token, expected_type="access")

    # In a real app, would fetch user from database
    # For now, create user from token data
    user = User(
        id=token_data.sub,
        username=token_data.sub,
        scopes=token_data.scopes,
        is_active=True,
        is_admin="admin" in token_data.scopes
    )

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user (requires user to be active)"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


async def get_current_admin_user(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Get current admin user (requires admin scope)"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


def require_scopes(*required_scopes: str):
    """
    Dependency to require specific scopes

    Usage:
        @app.get("/admin/users")
        async def list_users(user: User = Depends(require_scopes("admin", "users:read"))):
            return {"users": [...]}
    """
    async def check_scopes(user: User = Depends(get_current_active_user)) -> User:
        for scope in required_scopes:
            if scope not in user.scopes:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing required scope: {scope}"
                )
        return user

    return check_scopes
