"""
Polymarket CLOB authentication.

Two-level auth system:
- L1: EIP-712 wallet signing (create/derive API keys)
- L2: HMAC-SHA256 with derived API credentials (trading)

Usage:
    auth = PolymarketAuth(private_key="0x...")
    auth.initialize()  # derives API creds
    client = auth.client  # ready to trade
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds

from config.settings import (
    CLOB_API,
    POLYGON_CHAIN_ID,
    POLYMARKET_PRIVATE_KEY,
    POLYMARKET_FUNDER_ADDRESS,
)

logger = logging.getLogger(__name__)


# Signature types for Polymarket
SIG_TYPE_EOA = 0          # Standard EOA wallet
SIG_TYPE_GNOSIS = 1       # Gnosis Safe / multi-sig


@dataclass
class AuthStatus:
    """Authentication state."""
    is_l1: bool = False
    is_l2: bool = False
    address: str = ""
    api_key: str = ""
    error: str = ""


class PolymarketAuth:
    """
    Manages Polymarket CLOB authentication.
    
    Handles:
    - Client initialization at L1 (wallet signing)
    - API credential derivation for L2 (HMAC trading)
    - Credential persistence to avoid re-deriving
    - Health checks
    """
    
    def __init__(
        self,
        private_key: Optional[str] = None,
        funder_address: Optional[str] = None,
        host: str = CLOB_API,
        chain_id: int = POLYGON_CHAIN_ID,
        signature_type: int = SIG_TYPE_EOA,
    ):
        self.private_key = private_key or POLYMARKET_PRIVATE_KEY
        self.funder_address = funder_address or POLYMARKET_FUNDER_ADDRESS
        self.host = host
        self.chain_id = chain_id
        self.signature_type = signature_type
        self._client: Optional[ClobClient] = None
        self._creds: Optional[ApiCreds] = None
        self._status = AuthStatus()
    
    @property
    def client(self) -> ClobClient:
        """Get the authenticated ClobClient."""
        if self._client is None:
            raise RuntimeError("Auth not initialized. Call initialize() first.")
        return self._client
    
    @property
    def status(self) -> AuthStatus:
        return self._status
    
    @property
    def address(self) -> str:
        return self._status.address
    
    @property
    def is_ready(self) -> bool:
        """Whether the client is fully authenticated (L2)."""
        return self._status.is_l2
    
    def initialize(self, creds: Optional[ApiCreds] = None) -> AuthStatus:
        """
        Initialize authentication.
        
        1. Creates ClobClient with private key (L1)
        2. Derives or uses provided API credentials (L2)
        
        Args:
            creds: Pre-existing API credentials (skips derivation)
        
        Returns:
            AuthStatus with current state
        """
        if not self.private_key:
            self._status.error = "No private key configured"
            logger.error(self._status.error)
            return self._status
        
        try:
            # Step 1: L1 - Create client with wallet signing
            self._client = ClobClient(
                host=self.host,
                chain_id=self.chain_id,
                key=self.private_key,
                signature_type=self.signature_type,
                funder=self.funder_address or None,
            )
            
            self._status.address = self._client.get_address() or ""
            self._status.is_l1 = True
            logger.info(f"L1 auth OK — address: {self._status.address}")
            
            # Step 2: L2 - Derive or set API credentials
            if creds:
                self._creds = creds
            else:
                # Try to load from env first
                env_key = os.getenv("POLYMARKET_API_KEY", "")
                env_secret = os.getenv("POLYMARKET_API_SECRET", "")
                env_passphrase = os.getenv("POLYMARKET_API_PASSPHRASE", "")
                
                if env_key and env_secret and env_passphrase:
                    self._creds = ApiCreds(
                        api_key=env_key,
                        api_secret=env_secret,
                        api_passphrase=env_passphrase,
                    )
                    logger.info("Using API credentials from environment")
                else:
                    # Derive new credentials
                    logger.info("Deriving API credentials...")
                    self._creds = self._client.create_or_derive_api_creds()
                    logger.info(f"API credentials derived — key: {self._creds.api_key[:8]}...")
            
            self._client.set_api_creds(self._creds)
            self._status.is_l2 = True
            self._status.api_key = self._creds.api_key
            logger.info("L2 auth OK — full trading access")
            
        except Exception as e:
            self._status.error = str(e)
            logger.error(f"Auth failed: {e}")
        
        return self._status
    
    def health_check(self) -> dict:
        """
        Verify authentication by calling CLOB endpoints.
        
        Returns dict with:
            server_ok: bool
            server_time: str
            address: str
            api_keys_count: int
            balance_allowance: dict (if available)
        """
        result = {
            'server_ok': False,
            'server_time': '',
            'address': self._status.address,
            'api_keys_count': 0,
            'error': '',
        }
        
        if not self._client:
            result['error'] = "Client not initialized"
            return result
        
        try:
            # Basic health check
            ok = self._client.get_ok()
            result['server_ok'] = ok == "OK"
            
            # Server time
            result['server_time'] = self._client.get_server_time()
            
            # API keys (L2)
            if self._status.is_l2:
                keys = self._client.get_api_keys()
                result['api_keys_count'] = len(keys) if isinstance(keys, list) else 0
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def get_balance_allowance(self) -> dict:
        """
        Check USDC balance and exchange allowance.
        
        Returns:
            {balance: str, allowance: str} for COLLATERAL type
        """
        if not self.is_ready:
            return {'error': 'Not authenticated'}
        
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
            result = self._client.get_balance_allowance(
                BalanceAllowanceParams(
                    asset_type=AssetType.COLLATERAL,
                    signature_type=self.signature_type,
                )
            )
            return result if isinstance(result, dict) else {'raw': str(result)}
        except Exception as e:
            return {'error': str(e)}
    
    def get_creds_for_env(self) -> dict:
        """
        Get credentials formatted for .env file.
        Useful for persisting after first derivation.
        """
        if not self._creds:
            return {}
        return {
            'POLYMARKET_API_KEY': self._creds.api_key,
            'POLYMARKET_API_SECRET': self._creds.api_secret,
            'POLYMARKET_API_PASSPHRASE': self._creds.api_passphrase,
        }
