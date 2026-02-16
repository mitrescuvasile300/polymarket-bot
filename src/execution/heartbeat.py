"""
Heartbeat manager for Polymarket CLOB.

CRITICAL: If heartbeats are started and one isn't sent within 10 seconds,
Polymarket CANCELS ALL open orders automatically. This module ensures
heartbeats are sent reliably via asyncio task (no thread safety issues).

Usage:
    hb = HeartbeatManager(clob_client)
    hb.start()           # Begin heartbeat loop (asyncio task)
    ...
    await hb.stop()      # Clean shutdown
"""

import asyncio
import logging
import time
import uuid
from typing import Optional

from py_clob_client.client import ClobClient

logger = logging.getLogger(__name__)

# Polymarket cancels orders if no heartbeat within 10s
# We send every 5s for safety margin
DEFAULT_INTERVAL = 5.0
MAX_FAILURES = 3  # Stop after this many consecutive failures


class HeartbeatManager:
    """
    Sends periodic heartbeats to keep open orders alive.
    
    Uses asyncio task (not threading) to avoid sharing ClobClient
    across threads which could corrupt request state.
    """
    
    def __init__(
        self,
        client: ClobClient,
        interval: float = DEFAULT_INTERVAL,
    ):
        self.client = client
        self.interval = interval
        self.heartbeat_id: str = str(uuid.uuid4())
        
        self._task: Optional[asyncio.Task] = None
        self._is_running = False
        self._consecutive_failures = 0
        self._total_sent = 0
        self._last_sent: float = 0.0
        self._last_error: str = ""
    
    @property
    def is_running(self) -> bool:
        return self._is_running
    
    @property
    def seconds_since_last(self) -> float:
        if self._last_sent == 0:
            return float('inf')
        return time.time() - self._last_sent
    
    def start(self):
        """Start the heartbeat asyncio task."""
        if self._is_running:
            logger.warning("Heartbeat already running")
            return
        
        self._is_running = True
        self._consecutive_failures = 0
        self.heartbeat_id = str(uuid.uuid4())
        
        self._task = asyncio.ensure_future(self._heartbeat_loop())
        logger.info(f"Heartbeat started (id={self.heartbeat_id[:8]}..., interval={self.interval}s)")
    
    async def stop(self):
        """Stop the heartbeat task gracefully."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"Heartbeat stopped (sent {self._total_sent} total)")
    
    async def _heartbeat_loop(self):
        """Main heartbeat loop as asyncio coroutine."""
        while self._is_running:
            try:
                # Run blocking CLOB call in thread pool to not block event loop
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, self.client.post_heartbeat, self.heartbeat_id
                )
                self._total_sent += 1
                self._last_sent = time.time()
                self._consecutive_failures = 0
                
                if self._total_sent % 100 == 0:
                    logger.debug(f"Heartbeat #{self._total_sent} OK")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._consecutive_failures += 1
                self._last_error = str(e)
                logger.error(
                    f"Heartbeat failed ({self._consecutive_failures}/{MAX_FAILURES}): {e}"
                )
                
                if self._consecutive_failures >= MAX_FAILURES:
                    logger.critical(
                        f"Heartbeat DEAD after {MAX_FAILURES} failures! "
                        f"All open orders will be cancelled by Polymarket."
                    )
                    self._is_running = False
                    break
            
            try:
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break
    
    def status(self) -> dict:
        """Get heartbeat status."""
        return {
            'is_running': self._is_running,
            'heartbeat_id': self.heartbeat_id[:8],
            'total_sent': self._total_sent,
            'seconds_since_last': round(self.seconds_since_last, 1),
            'consecutive_failures': self._consecutive_failures,
            'last_error': self._last_error,
        }
