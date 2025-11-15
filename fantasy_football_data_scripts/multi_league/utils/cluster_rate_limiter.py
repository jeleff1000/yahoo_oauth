"""
Cluster-aware rate limiter with shared token store.

Allows multiple processes to coordinate rate limiting through a shared
file-based token store, preventing thundering herd when running parallel
workers.

Usage:
    from cluster_rate_limiter import ClusterRateLimiter

    # Single process (uses in-memory tokens)
    limiter = ClusterRateLimiter(rate=4.0)
    limiter.acquire()

    # Multi-process (uses shared file)
    limiter = ClusterRateLimiter(rate=4.0, shared_store="/tmp/rate_limit_tokens.json")
    limiter.acquire()

Alternative: Use a coordinator process
    from cluster_rate_limiter import RateLimitCoordinator

    # Run coordinator as a separate process
    coordinator = RateLimitCoordinator(rate=4.0, port=5555)
    coordinator.start()

    # Workers connect to coordinator
    limiter = ClusterRateLimiter(rate=4.0, coordinator_url="http://localhost:5555")
    limiter.acquire()
"""
from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from pathlib import Path
from time import monotonic
from typing import Optional
import random


class ClusterRateLimiter:
    """
    Rate limiter that can coordinate across multiple processes.

    Supports three modes:
    1. In-memory (single process): No shared_store or coordinator_url
    2. File-based (multiple processes): shared_store path provided
    3. Coordinator-based (multiple processes): coordinator_url provided
    """

    def __init__(
        self,
        rate: float = 4.0,
        burst: Optional[float] = None,
        shared_store: Optional[Union[str, Path]] = None,
        coordinator_url: Optional[str] = None
    ):
        """
        Initialize cluster-aware rate limiter.

        Args:
            rate: Requests per second
            burst: Max burst size (default: same as rate)
            shared_store: Path to shared token store file (for file-based coordination)
            coordinator_url: URL of coordinator service (for coordinator-based)
        """
        self.rate = rate
        self.burst = burst if burst is not None else rate
        self.shared_store = Path(shared_store) if shared_store else None
        self.coordinator_url = coordinator_url

        # In-memory state (used when no coordination)
        self._tokens = self.burst
        self._last = monotonic()
        self._lock = threading.Lock()

        # Determine coordination mode
        if coordinator_url:
            self.mode = 'coordinator'
        elif shared_store:
            self.mode = 'file'
            self._ensure_shared_store()
        else:
            self.mode = 'memory'

    def _ensure_shared_store(self) -> None:
        """Create shared store file if it doesn't exist."""
        if self.shared_store and not self.shared_store.exists():
            self.shared_store.parent.mkdir(parents=True, exist_ok=True)
            with open(self.shared_store, 'w') as f:
                json.dump({
                    'tokens': self.burst,
                    'last_update': monotonic(),
                    'rate': self.rate,
                    'burst': self.burst
                }, f)

    def acquire(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens, blocking if necessary.

        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait (None = wait forever)

        Returns:
            True if tokens acquired, False if timeout
        """
        if self.mode == 'coordinator':
            return self._acquire_coordinator(tokens, timeout)
        elif self.mode == 'file':
            return self._acquire_file(tokens, timeout)
        else:
            return self._acquire_memory(tokens, timeout)

    def _acquire_memory(self, tokens: float, timeout: Optional[float]) -> bool:
        """Acquire tokens using in-memory store (single process)."""
        start = monotonic() if timeout else None

        with self._lock:
            while True:
                now = monotonic()

                # Replenish tokens
                elapsed = now - self._last
                self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
                self._last = now

                # Check if we have enough tokens
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True

                # Check timeout
                if timeout and (monotonic() - start) >= timeout:
                    return False

                # Calculate wait time
                tokens_needed = tokens - self._tokens
                wait_time = tokens_needed / self.rate
                time.sleep(min(wait_time, 0.1))

    def _acquire_file(self, tokens: float, timeout: Optional[float]) -> bool:
        """Acquire tokens using file-based store (multi-process)."""
        start = monotonic() if timeout else None

        while True:
            # Try to acquire tokens
            acquired = self._try_acquire_file(tokens)
            if acquired:
                return True

            # Check timeout
            if timeout and (monotonic() - start) >= timeout:
                return False

            # Back off with jitter
            backoff = (1.0 / self.rate) * (0.5 + random.random() * 0.5)
            time.sleep(backoff)

    def _try_acquire_file(self, tokens: float) -> bool:
        """
        Try to acquire tokens from file store.

        Returns True if acquired, False if not enough tokens available.
        """
        try:
            # Open file with exclusive lock
            with open(self.shared_store, 'r+') as f:
                # Acquire exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)

                try:
                    # Read current state
                    f.seek(0)
                    state = json.load(f)

                    current_tokens = state['tokens']
                    last_update = state['last_update']
                    now = monotonic()

                    # Replenish tokens
                    elapsed = now - last_update
                    current_tokens = min(self.burst, current_tokens + elapsed * self.rate)

                    # Check if we have enough tokens
                    if current_tokens >= tokens:
                        # Acquire tokens
                        current_tokens -= tokens

                        # Write updated state
                        state['tokens'] = current_tokens
                        state['last_update'] = now

                        f.seek(0)
                        f.truncate()
                        json.dump(state, f)

                        return True
                    else:
                        # Not enough tokens
                        # Update timestamp anyway to keep replenishment accurate
                        state['tokens'] = current_tokens
                        state['last_update'] = now

                        f.seek(0)
                        f.truncate()
                        json.dump(state, f)

                        return False

                finally:
                    # Release lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        except Exception as e:
            # If file locking fails, fall back to sleep
            print(f"Warning: File-based rate limiting failed: {e}")
            time.sleep(1.0 / self.rate)
            return True

    def _acquire_coordinator(self, tokens: float, timeout: Optional[float]) -> bool:
        """Acquire tokens from coordinator service."""
        import requests

        start = monotonic() if timeout else None

        while True:
            try:
                response = requests.post(
                    f"{self.coordinator_url}/acquire",
                    json={'tokens': tokens, 'timeout': 5.0},
                    timeout=10.0
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get('acquired', False)
                elif response.status_code == 408:  # Timeout
                    pass  # Retry
                else:
                    print(f"Warning: Coordinator returned {response.status_code}")
                    time.sleep(1.0 / self.rate)
                    return True

            except Exception as e:
                print(f"Warning: Failed to contact coordinator: {e}")
                time.sleep(1.0 / self.rate)
                return True

            # Check timeout
            if timeout and (monotonic() - start) >= timeout:
                return False

            time.sleep(0.1)

    def acquire_with_jitter(
        self,
        tokens: float = 1.0,
        jitter: float = 0.3,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Acquire tokens with random jitter to prevent thundering herd.

        Args:
            tokens: Number of tokens to acquire
            jitter: Jitter factor (0.0 to 1.0)
            timeout: Maximum time to wait

        Returns:
            True if tokens acquired, False if timeout
        """
        # Add random pre-delay
        if jitter > 0:
            delay = random.uniform(0, jitter / self.rate)
            time.sleep(delay)

        return self.acquire(tokens, timeout)


class RateLimitCoordinator:
    """
    Centralized rate limit coordinator service.

    Runs as a separate process and manages token bucket for all workers.
    Workers connect via HTTP to acquire tokens.

    This is the most robust solution for multi-process rate limiting,
    but requires running a coordinator process.
    """

    def __init__(self, rate: float = 4.0, burst: Optional[float] = None, port: int = 5555):
        """
        Initialize coordinator.

        Args:
            rate: Requests per second
            burst: Max burst size
            port: Port to listen on
        """
        self.rate = rate
        self.burst = burst if burst is not None else rate
        self.port = port

        # Token bucket state
        self._tokens = self.burst
        self._last = monotonic()
        self._lock = threading.Lock()

    def start(self):
        """Start coordinator service."""
        from flask import Flask, request, jsonify

        app = Flask(__name__)

        @app.route('/acquire', methods=['POST'])
        def acquire():
            data = request.json
            tokens = data.get('tokens', 1.0)
            timeout = data.get('timeout', None)

            acquired = self._acquire(tokens, timeout)

            if acquired:
                return jsonify({'acquired': True}), 200
            else:
                return jsonify({'acquired': False}), 408

        @app.route('/status', methods=['GET'])
        def status():
            with self._lock:
                return jsonify({
                    'tokens': self._tokens,
                    'rate': self.rate,
                    'burst': self.burst
                }), 200

        print(f"Starting rate limit coordinator on port {self.port}")
        print(f"Rate: {self.rate} req/sec, Burst: {self.burst}")
        app.run(host='0.0.0.0', port=self.port)

    def _acquire(self, tokens: float, timeout: Optional[float]) -> bool:
        """Acquire tokens from coordinator's token bucket."""
        start = monotonic() if timeout else None

        with self._lock:
            while True:
                now = monotonic()

                # Replenish tokens
                elapsed = now - self._last
                self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
                self._last = now

                # Check if we have enough tokens
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True

                # Check timeout
                if timeout and (monotonic() - start) >= timeout:
                    return False

                # Wait
                tokens_needed = tokens - self._tokens
                wait_time = tokens_needed / self.rate
                time.sleep(min(wait_time, 0.1))


def get_rate_limiter(
    rate: float = 4.0,
    mode: str = 'auto',
    **kwargs
) -> ClusterRateLimiter:
    """
    Factory function to get appropriate rate limiter.

    Args:
        rate: Requests per second
        mode: 'auto', 'memory', 'file', or 'coordinator'
        **kwargs: Additional arguments for ClusterRateLimiter

    Returns:
        ClusterRateLimiter instance

    Example:
        >>> # Auto-detect based on environment
        >>> limiter = get_rate_limiter(rate=4.0)
        >>>
        >>> # Force file-based for multi-process
        >>> limiter = get_rate_limiter(
        ...     rate=4.0,
        ...     mode='file',
        ...     shared_store='/tmp/rate_limit.json'
        ... )
    """
    if mode == 'auto':
        # Check environment for hints
        coordinator_url = os.environ.get('RATE_LIMIT_COORDINATOR_URL')
        if coordinator_url:
            return ClusterRateLimiter(rate, coordinator_url=coordinator_url, **kwargs)

        shared_store = os.environ.get('RATE_LIMIT_SHARED_STORE')
        if shared_store:
            return ClusterRateLimiter(rate, shared_store=shared_store, **kwargs)

        # Default to in-memory
        return ClusterRateLimiter(rate, **kwargs)

    elif mode == 'memory':
        return ClusterRateLimiter(rate, **kwargs)

    elif mode == 'file':
        shared_store = kwargs.get('shared_store')
        if not shared_store:
            # Default location
            shared_store = Path.home() / '.fantasy_football' / 'rate_limit.json'
            kwargs['shared_store'] = shared_store
        return ClusterRateLimiter(rate, **kwargs)

    elif mode == 'coordinator':
        coordinator_url = kwargs.get('coordinator_url')
        if not coordinator_url:
            coordinator_url = 'http://localhost:5555'
            kwargs['coordinator_url'] = coordinator_url
        return ClusterRateLimiter(rate, **kwargs)

    else:
        raise ValueError(f"Unknown mode: {mode}")
