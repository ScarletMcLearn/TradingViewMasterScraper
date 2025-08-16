## safe_requests.py
## Thoroughly commented version explaining each line and decision.

# from __future__ import annotations
# ^ Optional in Python 3.11+: would postpone evaluation of type hints.
#   Left commented because this file runs fine without it and Kaggle often uses 3.10/3.11.

import logging                 # Standard logging for diagnostics.
import time                    # Used for monotonic clocks and sleeps.
import threading               # Ensures our rate limiter is thread-safe.
from typing import Any, Callable, Iterable, Optional  # Type hints for clarity.
from email.utils import parsedate_to_datetime         # Parse HTTP-date Retry-After.
from datetime import datetime, timezone               # For timezone-aware math.

import requests                # The underlying HTTP client we wrap.
from requests.adapters import HTTPAdapter             # To configure connection pooling.
from requests.exceptions import (                     # Network/HTTP exceptions to retry.
    ConnectionError,
    Timeout,
    ReadTimeout,
    ChunkedEncodingError,
    SSLError,  # retry transient TLS hiccups
)
from tenacity import (                                 # Tenacity drives retry/backoff.
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log,
)

# Create a module-level logger; users can pass their own to SafeSession if desired.
logger = logging.getLogger("safe_requests")
logger.addHandler(logging.NullHandler())  # Avoid "No handler found" warnings by default.


class RetryableStatusError(Exception):
    """Internal marker to signal retry for a response status."""
    def __init__(self, response: requests.Response):
        # Include the HTTP code in the message for easy debugging.
        super().__init__(f"Retryable HTTP {response.status_code}")
        # Keep the actual Response so we can surface it if retries exhaust.
        self.response = response


# -------------------------
# Rate limiter (token bucket)
# -------------------------
class TokenBucket:
    """
    Thread-safe token bucket rate limiter.

    rate = max_calls / per_seconds
    capacity = burst (max tokens you can accumulate).
    """
    def __init__(self, max_calls: int, per_seconds: float, burst: Optional[int] = None):
        # Validate that we were given sensible, positive numbers.
        if max_calls <= 0 or per_seconds <= 0:
            raise ValueError("max_calls and per_seconds must be > 0")
        # tokens per second we earn back into the bucket
        self.rate = float(max_calls) / float(per_seconds)
        # max number of tokens the bucket can hold (burst), default to max_calls
        self.capacity = float(burst if burst is not None else max_calls)
        # start full so we can immediately burst up to capacity
        self.tokens = self.capacity
        # monotonic clock to avoid issues if system clock changes
        self.last = time.monotonic()
        # mutex to make acquire() safe across threads
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> None:
        """Blocks until `tokens` are available."""
        # Loop until we can deduct the requested tokens.
        while True:
            with self._lock:
                # Current monotonic time to compute elapsed since last refill.
                now = time.monotonic()
                elapsed = now - self.last
                if elapsed > 0:
                    # Add earned tokens (elapsed * rate), capped by capacity.
                    self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                    # Move the reference point to now.
                    self.last = now

                # If we have enough tokens, take them and return immediately.
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return

                # Otherwise, compute how long to sleep to earn enough tokens.
                needed = tokens - self.tokens
                sleep_for = needed / self.rate

            # Sleep outside the lock so other threads can progress/refill.
            time.sleep(sleep_for)


# -------------------------
# Utilities
# -------------------------

# HTTP methods that are safe/idempotent by spec or convention (okay to retry).
_IDEMPOTENT_METHODS = {"GET", "HEAD", "OPTIONS", "DELETE"}

# Default set of HTTP status codes we consider transient/ retryable.
_DEFAULT_STATUS_FORCELIST = {
    408,  # Request Timeout
    425,  # Too Early
    429,  # Too Many Requests
    500, 502, 503, 504,  # Server errors / gateways
    521, 522, 523,       # Common CDN edge errors
}

def _parse_retry_after_seconds(value: str) -> Optional[float]:
    """Support both delta-seconds and HTTP-date formats."""
    # If header missing/blank, nothing to parse.
    if not value:
        return None
    # Normalize whitespace.
    value = value.strip()
    # The simple case: an integer number of seconds.
    if value.isdigit():
        return max(0.0, float(value))
    # Otherwise try HTTP-date (e.g., "Wed, 21 Oct 2015 07:28:00 GMT").
    try:
        dt = parsedate_to_datetime(value)
        # Some servers omit tzinfo; assume UTC to make it comparable.
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        # Compute seconds from "now" to the target date; clamp at 0.
        delta = (dt - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, delta)
    except Exception:
        # If parsing fails, just ignore header and fall back to backoff.
        return None


def _should_retry_response(
    resp: requests.Response,
    method: str,
    allow_unsafe_method_retry: bool,
    status_forcelist: Iterable[int],
) -> bool:
    # Guard: if somehow we got None, don't retry.
    if resp is None:
        return False
    # Read the HTTP code once.
    code = resp.status_code
    # Only consider retries if the code is in our retryable set.
    if code not in status_forcelist:
        return False
    # If it's an idempotent method, we can retry.
    if method in _IDEMPOTENT_METHODS:
        return True
    # For unsafe methods (POST/PUT/PATCH), only retry if explicitly allowed
    # or the request carries an Idempotency-Key header.
    if allow_unsafe_method_retry or "Idempotency-Key" in (resp.request.headers or {}):
        return True
    # Otherwise, don't retry.
    return False


def _combined_wait_with_retry_after(min_seconds: float, max_seconds: float) -> Callable:
    """
    Tenacity wait function that prefers Retry-After (on either a retryable response
    wrapped in RetryableStatusError or a previous successful Response), otherwise
    uses randomized exponential backoff.
    """
    # Base wait uses exponential backoff with jitter up to max_seconds.
    base = wait_random_exponential(multiplier=1, max=max_seconds)

    def _wait(retry_state) -> float:
        # Start with the default exponential backoff delay.
        default_wait = base(retry_state)
        try:
            # Tenacity stores the last attempt outcome (result or exception).
            outcome = retry_state.outcome
            if outcome:
                if outcome.failed:
                    # If last attempt failed with an exception…
                    exc = outcome.exception()
                    # …and that exception wraps a retryable response…
                    if isinstance(exc, RetryableStatusError):
                        # …respect its Retry-After header if present.
                        ra = exc.response.headers.get("Retry-After")
                        secs = _parse_retry_after_seconds(ra) if ra else None
                        if secs is not None:
                            # Return at least min_seconds; never negative.
                            return max(secs, min_seconds, 0.0)
                else:
                    # If last attempt returned a Response successfully (rare path),
                    # we can still check for a Retry-After header to be polite.
                    resp = outcome.result()
                    if isinstance(resp, requests.Response):
                        ra = resp.headers.get("Retry-After")
                        secs = _parse_retry_after_seconds(ra) if ra else None
                        if secs is not None:
                            return max(secs, min_seconds, 0.0)
        except Exception:
            # Never let wait calculation itself crash retry.
            pass
        # Fall back to exponential backoff (respect min_seconds).
        return max(default_wait, min_seconds)

    # Return the callable Tenacity will invoke between attempts.
    return _wait


# --------------------------------------
# The wrapper: a safer, rate-limited Session
# --------------------------------------
class SafeSession(requests.Session):
    """
    A drop-in `requests.Session` with:

    - Token-bucket rate limiting
    - Tenacity-based retries with exponential backoff + jitter
    - Respect for Retry-After
    - Default timeouts
    - Connection pooling (HTTPAdapter)
    - Pass-through `requests` API
    """

    def __init__(
        self,
        # Rate limiting
        max_calls: int = 10,                 # Allowed calls per period (see per_seconds).
        per_seconds: float = 1.0,            # Window length for the rate.
        burst: Optional[int] = None,         # Optional burst capacity (defaults to max_calls).

        # Retry / backoff
        max_attempts: int = 6,               # Total attempts (1 initial + retries).
        backoff_min: float = 0.5,            # Minimum sleep between retries.
        backoff_max: float = 60.0,           # Maximum backoff cap.
        status_forcelist: Iterable[int] = _DEFAULT_STATUS_FORCELIST,  # Retryable HTTP codes.

        # Networking
        default_timeout: float = 10.0,       # Per-request default timeout unless overridden.
        pool_connections: int = 32,          # Connection pool size (adapters manage pooling).
        pool_maxsize: int = 128,             # Max pool size across hosts.
        user_agent: str = "safe-requests/1.0 (+https://example.com)",  # Default UA.

        # Logging
        log: Optional[logging.Logger] = None,  # Optional custom logger.
    ):
        # Initialize base requests.Session.
        super().__init__()

        # Build the shared token bucket limiter for this session.
        self._rate_limiter = TokenBucket(max_calls=max_calls, per_seconds=per_seconds, burst=burst)
        # Store retryable statuses as a set for fast membership checks.
        self._status_forcelist = set(status_forcelist)
        # Keep defaults configurable per-session.
        self._default_timeout = default_timeout
        self._max_attempts = max_attempts
        self._backoff_min = backoff_min
        self._backoff_max = backoff_max
        # Use provided logger or the module logger.
        self._logger = log or logger

        # Configure HTTPAdapter explicitly:
        # - max_retries=0 so urllib3 doesn't double-retry; Tenacity handles retries.
        # - pool sizes to enable aggressive connection reuse.
        adapter = HTTPAdapter(pool_connections=pool_connections, pool_maxsize=pool_maxsize, max_retries=0)
        # Mount adapter for both http and https schemes.
        self.mount("http://", adapter)
        self.mount("https://", adapter)

        # Set sensible default headers at the session level.
        self.headers.update({
            "User-Agent": user_agent,
            "Accept": "application/json, text/plain, */*",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        })

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        Extra optional kwargs:

        - timeout: float | (connect, read)
        - raise_on_status: bool = True
        - allow_unsafe_method_retry: bool = False
        - max_attempts: int
        - status_forcelist: Iterable[int]
        """
        # Normalize method to upper-case for comparisons (e.g., "GET").
        method_upper = method.upper()

        # Extract custom controls from kwargs so we don't pass unknown keys to requests.
        raise_on_status: bool = kwargs.pop("raise_on_status", True)
        allow_unsafe_method_retry: bool = kwargs.pop("allow_unsafe_method_retry", False)
        max_attempts: int = kwargs.pop("max_attempts", self._max_attempts)
        status_forcelist = set(kwargs.pop("status_forcelist", self._status_forcelist))

        # Ensure we always set a timeout to avoid hanging forever.
        if "timeout" not in kwargs or kwargs["timeout"] is None:
            kwargs["timeout"] = self._default_timeout

        # Small helper that checks whether the response *should* trigger a retry.
        def _retry_on_resp(resp: requests.Response) -> bool:
            return _should_retry_response(
                resp=resp,
                method=method_upper,
                allow_unsafe_method_retry=allow_unsafe_method_retry,
                status_forcelist=status_forcelist,
            )

        # Perform a single HTTP attempt (Tenacity will call this multiple times).
        def _send_once() -> requests.Response:
            # Throttle according to the token bucket before each attempt.
            self._rate_limiter.acquire(1.0)
            # Delegate the actual HTTP call to the base Session.
            resp = super(SafeSession, self).request(method_upper, url, **kwargs)
            # If the response is retryable (e.g., 429/5xx), raise our internal
            # exception so Tenacity treats it like other retryable failures.
            if _retry_on_resp(resp):
                raise RetryableStatusError(resp)
            # Otherwise, return the response immediately.
            return resp

        # Build the retry policy: retries are driven by exceptions only.
        retry_on_ex = retry_if_exception_type((
            ConnectionError,         # Network-level failure.
            Timeout,                 # Overall timeout exceeded.
            ReadTimeout,             # Server didn't send data in time.
            ChunkedEncodingError,    # Broken transfer encoding / connection.
            SSLError,                # Transient TLS issues (often recoverable).
            RetryableStatusError,    # Our wrapper for retryable HTTP statuses.
        ))
        # Compose a wait strategy that honors Retry-After and adds jittered backoff.
        wait = _combined_wait_with_retry_after(self._backoff_min, self._backoff_max)

        # Create the Tenacity Retrying controller (iterator-style API).
        retrying = Retrying(
            stop=stop_after_attempt(max_attempts),  # Stop after N total attempts.
            wait=wait,                              # Sleep policy between attempts.
            retry=retry_on_ex,                      # Only exceptions trigger retries.
            reraise=True,                           # Re-raise the last error if we give up.
            before_sleep=before_sleep_log(self._logger, logging.WARNING),  # Log before sleeping.
        )

        # Execute with retries using the iterator pattern (Tenacity 7/8/9 compatible).
        resp = None
        try:
            # Each iteration corresponds to one attempt. Entering the context allows tenacity
            # to capture exceptions and decide whether to retry.
            for attempt in retrying:
                with attempt:
                    resp = _send_once()
        except RetryableStatusError as e:
            # If we exhausted attempts on retryable HTTP statuses, expose the final Response
            # so callers can inspect status/text as usual.
            resp = e.response

        # If the caller wants exceptions on error codes, raise now (after retries).
        if raise_on_status:
            try:
                resp.raise_for_status()
            except requests.HTTPError as e:
                # Log a short snippet of the body to help debugging without huge logs.
                text_snippet = ""
                try:
                    text_snippet = resp.text[:512]
                except Exception:
                    pass
                self._logger.error("HTTPError %s %s -> %s\n%s", method_upper, url, e, text_snippet)
                # Re-raise so the caller can handle it.
                raise

        # Return the successful (or non-raising) response to the caller.
        return resp


# -------------------------
# Convenience factory
# -------------------------
def create_session(**kwargs) -> SafeSession:
    """s = create_session(max_calls=5, per_seconds=1, burst=10, ...)"""
    # Simply forward kwargs to the SafeSession constructor.
    return SafeSession(**kwargs)





# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":  # Only run this block when the file is executed directly (not imported)
    logging.basicConfig(level=logging.INFO)  # Show INFO+ logs (Tenacity retry sleeps show at WARNING)

    # Create a SafeSession with rate limiting + retries + sane defaults
    s = create_session(
        max_calls=5,          # Allow at most 5 requests per `per_seconds` window...
        per_seconds=1.0,      # ...where the window is 1 second (so: 5 req/second)
        burst=10,             # Token bucket capacity: allows short bursts up to 10 immediate calls
        max_attempts=6,       # Total attempts per request: 1 initial + up to 5 retries
        backoff_min=0.5,      # Minimum backoff between retries (seconds)
        backoff_max=30.0,     # Maximum backoff cap (seconds). Retry-After can override if larger.
        user_agent="my-app/0.1 (+https://myapp.example)",  # Custom UA for server logs/diagnostics
    )

    # ---------------- GET demo ----------------
    # This endpoint randomly returns one of 200, 500, 500, 200.
    # Our session will retry on 5xx per the status_forcelist.
    # `raise_on_status=False` avoids raising if the final status is still an error,
    # so we can just print the observed outcome for demo purposes.
    r = s.get("https://httpbin.org/status/200,500,500,200", timeout=5, raise_on_status=False)
    print("GET ok:", r.status_code)  # Print final HTTP status (e.g., 200 or 500)

    # ---------------- POST demo (two variants) ----------------
    # By default, POST/PUT/PATCH are *not* retried (to avoid duplicate side effects).
    # If your endpoint is idempotent or supports Idempotency-Key, you can safely opt in.

    # Variant A: Show how you would *disable raising* to inspect final status manually.
    # (Left commented so the script doesn't run it by default.)
    # r = s.post(
    #     "https://httpbin.org/status/500,500,200",  # Randomly returns 500, then maybe 200
    #     json={"foo": "bar"},                        # Example JSON payload
    #     allow_unsafe_method_retry=False,            # Do NOT retry POST (default behavior)
    #     max_attempts=6,                             # Would only matter if retries were enabled
    #     raise_on_status=False,                      # Don’t raise even if final status is 5xx
    # )

    # Variant B: Safe retries for POST using Idempotency-Key (server must support the semantics).
    # (Left commented out; the active call below shows the same pattern.)
    # r = s.post(
    #     "https://httpbin.org/status/500,500,200",
    #     json={"foo": "bar"},
    #     headers={"Idempotency-Key": "a-unique-uuid"},  # Reuse same key on retries for same logical op
    #     allow_unsafe_method_retry=True,                # Opt-in to retry POST
    # )
    # print("POST final:", r.status_code)

    # Active POST call: demonstrate safe retries with Idempotency-Key + no exception on final error.
    r = s.post(
        "https://httpbin.org/status/500,500,200",  # Random statuses: good to demo retry/backoff
        json={"foo": "bar"},                       # JSON request body
        headers={"Idempotency-Key": "uuid"},       # Use a *unique* value in real code (e.g., uuid4())
        allow_unsafe_method_retry=True,            # Enable retries for this POST since it's idempotent-safe
        max_attempts=6,                            # Up to 6 total attempts (1 + 5 retries)
        raise_on_status=False,                     # Don’t raise; just let us inspect the final status
    )
    print(r.status_code)  # Print the final HTTP status (200 on success, maybe 500 if it never succeeded)



    # test_safe_requests.py
    import time
    import threading
    import uuid
    from contextlib import contextmanager

    import requests
    import responses

    # --- ALWAYS reload the module in Kaggle ---
    import importlib, inspect
    # import safe_requests
    # importlib.reload(safe_requests)
    # from safe_requests import (
    #     create_session, SafeSession, TokenBucket, _parse_retry_after_seconds, RetryableStatusError
    # )

    print("Uses RetryableStatusError in request():",
        "RetryableStatusError" in inspect.getsource(SafeSession.request))

    # Tiny helper to run a block and report
    @contextmanager
    def check(name):
        try:
            yield
            print(f"✅ {name}")
        except AssertionError as e:
            print(f"❌ {name} -> ASSERTION FAILED:", e)
        except Exception as e:
            print(f"❌ {name} -> EXCEPTION:", repr(e))

    def make_session(**kw):
        # lower waits for faster tests
        return create_session(
            max_calls=10, per_seconds=1.0, burst=10,
            max_attempts=4, backoff_min=0.05, backoff_max=0.2,
            user_agent="safe-requests-test/0.1",
            **kw
        )

    # --------------------------------------------------------------------
    # 0) Pure utility & TokenBucket tests (no network)
    # --------------------------------------------------------------------
    with check("TokenBucket: respects rate and burst"):
        tb = TokenBucket(max_calls=2, per_seconds=1.0, burst=2)  # 2 tokens/sec, burst 2
        t0 = time.perf_counter()
        for _ in range(4):  # need 4 tokens
            tb.acquire()
        elapsed = time.perf_counter() - t0
        assert elapsed >= 0.9, elapsed  # allow small timing wiggle

    with check("_parse_retry_after_seconds: delta seconds"):
        assert _parse_retry_after_seconds("2") in (2.0, 2), "expected 2s"

    # --------------------------------------------------------------------
    # 1) Retry behavior on idempotent GET: 500, 500, 200
    #    (set raise_on_status=False so the test never crashes even if misconfigured)
    # --------------------------------------------------------------------
    with responses.RequestsMock(assert_all_requests_are_fired=True) as rsps, check("GET retries on 5xx then succeeds"):
        url = "https://example.com/idempotent"
        rsps.add(responses.GET, url, status=500)
        rsps.add(responses.GET, url, status=500)
        rsps.add(responses.GET, url, json={"ok": True}, status=200)

        s = make_session()
        r = s.get(url, raise_on_status=False)
        assert r.status_code == 200
        assert len(rsps.calls) == 3

    # --------------------------------------------------------------------
    # 2) POST should NOT retry by default (unsafe)
    #    IMPORTANT: register only ONE stub here.
    # --------------------------------------------------------------------
    with responses.RequestsMock(assert_all_requests_are_fired=True) as rsps, check("POST does NOT retry unless allowed"):
        url = "https://example.com/unsafe"
        rsps.add(responses.POST, url, status=500)  # only one response

        s = make_session()
        r = s.post(url, json={"x": 1}, raise_on_status=False)
        assert r.status_code == 500
        assert len(rsps.calls) == 1

    # --------------------------------------------------------------------
    # 3) POST WITH retries when explicitly allowed + Idempotency-Key
    # --------------------------------------------------------------------
    with responses.RequestsMock(assert_all_requests_are_fired=True) as rsps, check("POST retries when allowed and idempotency present"):
        url = "https://example.com/unsafe-allowed"
        rsps.add(responses.POST, url, status=500)
        rsps.add(responses.POST, url, status=500)
        rsps.add(responses.POST, url, json={"ok": True}, status=200)

        s = make_session()
        r = s.post(
            url,
            json={"x": 1},
            headers={"Idempotency-Key": str(uuid.uuid4())},
            allow_unsafe_method_retry=True,
            raise_on_status=False,
        )
        assert r.status_code == 200
        assert len(rsps.calls) == 3

    # --------------------------------------------------------------------
    # 4) Respect Retry-After: 429 with header, then 200
    # --------------------------------------------------------------------
    with responses.RequestsMock(assert_all_requests_are_fired=True) as rsps, check("Retry-After respected (>= header wait)"):
        url = "https://example.com/too-many"
        rsps.add(responses.GET, url, headers={"Retry-After": "1"}, status=429)
        rsps.add(responses.GET, url, json={"ok": True}, status=200)

        s = make_session()
        t0 = time.perf_counter()
        r = s.get(url, raise_on_status=False)
        elapsed = time.perf_counter() - t0

        assert r.status_code == 200
        assert elapsed >= 0.95, elapsed  # allow tiny slack

    # --------------------------------------------------------------------
    # 5) Per-call override: status_forcelist (don’t retry 429)
    # --------------------------------------------------------------------
    with responses.RequestsMock(assert_all_requests_are_fired=True) as rsps, check("Per-call status_forcelist override prevents retry"):
        url = "https://example.com/no-retry-429"
        rsps.add(responses.GET, url, status=429)  # <- only one stub

        s = make_session()
        r = s.get(url, status_forcelist={500}, raise_on_status=False)  # only retry 500s, not 429
        assert r.status_code == 429
        assert len(rsps.calls) == 1


    # --------------------------------------------------------------------
    # 6) 404 is NOT in forcelist: should not retry
    # --------------------------------------------------------------------
    with responses.RequestsMock(assert_all_requests_are_fired=True) as rsps, check("404 not retried by default"):
        url = "https://example.com/not-found"
        rsps.add(responses.GET, url, status=404)
        s = make_session()
        r = s.get(url, raise_on_status=False)
        assert r.status_code == 404
        assert len(rsps.calls) == 1

    # --------------------------------------------------------------------
    # 7) exceptions (ReadTimeout) are retried up to max_attempts
    # --------------------------------------------------------------------
    with responses.RequestsMock(assert_all_requests_are_fired=True) as rsps, check("Exceptions (ReadTimeout) are retried"):
        url = "https://example.com/timeout-then-ok"
        rsps.add(responses.GET, url, body=requests.exceptions.ReadTimeout("rt1"))
        rsps.add(responses.GET, url, body=requests.exceptions.ReadTimeout("rt2"))
        rsps.add(responses.GET, url, json={"ok": True}, status=200)

        s = make_session()
        r = s.get(url, raise_on_status=False)
        assert r.status_code == 200
        assert len(rsps.calls) == 3

    # --------------------------------------------------------------------
    # 8) status_forcelist per-call can make 404 retryable
    # --------------------------------------------------------------------
    with responses.RequestsMock(assert_all_requests_are_fired=True) as rsps, check("404 retried when included in forcelist"):
        url = "https://example.com/now-retry-404"
        rsps.add(responses.GET, url, status=404)
        rsps.add(responses.GET, url, json={"ok": True}, status=200)

        s = make_session()
        r = s.get(url, status_forcelist={404, 500}, raise_on_status=False)
        assert r.status_code == 200
        assert len(rsps.calls) == 2

    # --------------------------------------------------------------------
    # 9) timeout kwarg passes through
    # --------------------------------------------------------------------
    with responses.RequestsMock(assert_all_requests_are_fired=True) as rsps, check("Timeout kwarg passes through"):
        url = "https://example.com/quick"
        rsps.add(responses.GET, url, json={"ok": True}, status=200)
        s = make_session()
        r = s.get(url, timeout=0.3, raise_on_status=False)
        assert r.status_code == 200

    # --------------------------------------------------------------------
    # 10) User-Agent header present
    # --------------------------------------------------------------------
    with responses.RequestsMock(assert_all_requests_are_fired=True) as rsps, check("Custom User-Agent is sent"):
        url = "https://example.com/ua"
        seen_ua = {"val": None}

        def cb(req):
            seen_ua["val"] = req.headers.get("User-Agent")
            return (200, {}, '{"ok": true}')

        rsps.add_callback(responses.GET, url, callback=lambda req: cb(req))
        s = make_session()
        r = s.get(url, raise_on_status=False)
        assert r.status_code == 200
        assert seen_ua["val"] and "safe-requests-test/0.1" in seen_ua["val"]

    # --------------------------------------------------------------------
    # 11) Query params are forwarded correctly
    # --------------------------------------------------------------------
    with responses.RequestsMock(assert_all_requests_are_fired=True) as rsps, check("Query params forwarded"):
        base = "https://example.com/echo"
        url = base + "?a=1&b=2"
        rsps.add(responses.GET, url, json={"ok": True}, status=200)

        s = make_session()
        r = s.get(base, params={"a": 1, "b": 2}, raise_on_status=False)
        assert r.status_code == 200
        assert len(rsps.calls) == 1

    # --------------------------------------------------------------------
    # 12) Non-forcelist status like 418 is not retried
    # --------------------------------------------------------------------
    with responses.RequestsMock(assert_all_requests_are_fired=True) as rsps, check("418 not retried by default"):
        url = "https://example.com/teapot"
        rsps.add(responses.GET, url, status=418)
        s = make_session()
        r = s.get(url, raise_on_status=False)
        assert r.status_code == 418
        assert len(rsps.calls) == 1

    # --------------------------------------------------------------------
    # 13) Per-call max_attempts override works
    # --------------------------------------------------------------------
    with responses.RequestsMock(assert_all_requests_are_fired=True) as rsps, check("Per-call max_attempts respected"):
        url = "https://example.com/limited-retries"
        rsps.add(responses.GET, url, status=500)
        rsps.add(responses.GET, url, status=500)
        # (remove the 200 stub because max_attempts=2 means we never reach it)

        s = make_session()
        try:
            # Only 2 total attempts (1 original + 1 retry) allowed here
            s.get(url, max_attempts=2)  # final still 500 -> raises
            assert False, "expected HTTPError"
        except requests.HTTPError:
            pass

        # Should have called exactly twice
        assert len(rsps.calls) == 2


    # --------------------------------------------------------------------
    # 14) Concurrency + bucket: N parallel requests constrained by rate
    # --------------------------------------------------------------------
    with responses.RequestsMock(assert_all_requests_are_fired=True) as rsps, check("Thread-safety & rate limit under concurrency"):
        url = "https://example.com/concurrent"
        for _ in range(6):
            rsps.add(responses.GET, url, json={"ok": True}, status=200)

        s = create_session(max_calls=2, per_seconds=1.0, burst=2, max_attempts=1,
                        backoff_min=0.05, backoff_max=0.1, user_agent="safe-requests-test/0.1")

        t0 = time.perf_counter()
        out = []

        def worker():
            r = s.get(url, raise_on_status=False)
            out.append(r.status_code)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for th in threads: th.start()
        for th in threads: th.join()

        elapsed = time.perf_counter() - t0
        assert len(out) == 6 and all(code == 200 for code in out)
        assert elapsed >= 1.6, elapsed   # ~2s expected; allow slack


    # Drop this in your suite as an extra check
    with responses.RequestsMock(assert_all_requests_are_fired=True) as rsps, check("send_once called 3 times for 500,500,200"):
        url = "https://example.com/callback"
        attempts = {"n": 0}

        def cb(req):
            attempts["n"] += 1
            if attempts["n"] < 3:
                return (500, {}, "err")
            return (200, {"Content-Type": "application/json"}, '{"ok": true}')

        rsps.add_callback(responses.GET, url, callback=cb)

        s = make_session()
        r = s.get(url, raise_on_status=False)
        assert r.status_code == 200
        assert attempts["n"] == 3          # <- _send_once invoked 3 times
        assert len(rsps.calls) == 3



    with responses.RequestsMock(assert_all_requests_are_fired=True) as rsps, check("POST no-retry proven via callback counter"):
        url = "https://example.com/post-no-retry"
        attempts = {"n": 0}

        def cb(req):
            attempts["n"] += 1
            return (500, {}, "boom")

        rsps.add_callback(responses.POST, url, callback=cb)

        s = make_session()
        r = s.post(url, json={"x": 1}, raise_on_status=False)
        assert r.status_code == 500
        assert attempts["n"] == 1           # only one call



    # integration_jsonplaceholder.py
    import uuid
    import logging
    # from safe_requests import create_session

    logging.basicConfig(level=logging.WARNING)  # see retry backoff logs if any

    # Create a session with sane limits/backoff
    s = create_session(
        max_calls=5, per_seconds=1.0, burst=5,   # rate limit
        max_attempts=4, backoff_min=0.3, backoff_max=4.0,
        user_agent="safe-requests-integration/1.0",
    )

    def p(label, resp):
        print(f"{label}: {resp.request.method} {resp.url} -> {resp.status_code}")
        ct = resp.headers.get("Content-Type", "")
        if "application/json" in ct:
            try:
                print(resp.json())
            except Exception:
                print(resp.text[:300])
        else:
            print(resp.text[:200])

    base = "https://jsonplaceholder.typicode.com"

    print("\n=== GET list (with params) ===")
    r = s.get(f"{base}/posts", params={"userId": 1}, timeout=10)
    p("GET /posts?userId=1", r)

    print("\n=== GET single ===")
    r = s.get(f"{base}/posts/1", timeout=10)
    p("GET /posts/1", r)

    print("\n=== HEAD (metadata only) ===")
    r = s.head(f"{base}/posts/1", raise_on_status=False, timeout=10)
    print("HEAD /posts/1 ->", r.status_code, "len:", r.headers.get("Content-Length"))

    print("\n=== OPTIONS (allowed methods) ===")
    r = s.options(f"{base}/posts/1", raise_on_status=False, timeout=10)
    print("OPTIONS /posts/1 ->", r.status_code,
        "allow:", r.headers.get("Allow") or r.headers.get("Access-Control-Allow-Methods"))

    # NOTE: JSONPlaceholder fakes writes (it doesn't actually persist), but returns 201/200 with echo.
    idem = str(uuid.uuid4())

    print("\n=== POST (create; safe retries via Idempotency-Key) ===")
    r = s.post(
        f"{base}/posts",
        json={"title": "foo", "body": "bar", "userId": 1},
        headers={"Idempotency-Key": idem},
        allow_unsafe_method_retry=True,
        timeout=10,
    )
    p("POST /posts", r)

    print("\n=== PUT (replace/update; safe retries) ===")
    r = s.put(
        f"{base}/posts/1",
        json={"id": 1, "title": "foo2", "body": "bar2", "userId": 1},
        headers={"Idempotency-Key": str(uuid.uuid4())},
        allow_unsafe_method_retry=True,
        timeout=10,
    )
    p("PUT /posts/1", r)

    print("\n=== PATCH (partial update; safe retries) ===")
    r = s.patch(
        f"{base}/posts/1",
        json={"title": "foo3"},
        headers={"Idempotency-Key": str(uuid.uuid4())},
        allow_unsafe_method_retry=True,
        timeout=10,
    )
    p("PATCH /posts/1", r)

    print("\n=== DELETE (remove) ===")
    r = s.delete(f"{base}/posts/1", timeout=10, raise_on_status=False)
    p("DELETE /posts/1", r)

    print("\nAll integration calls completed.")


    print("\nAll tests attempted.")
