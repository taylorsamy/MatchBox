"""
Bitfocus Companion integration for MatchBox.

Turns FTC scoring-system websocket messages (SHOW_PREVIEW, SHOW_MATCH, START_MATCH, ...)
into Companion actions over Companion's HTTP remote-control API (Companion 3.0+):

* ``POST /api/location/<page>/<row>/<column>/press``   - press a button
* ``POST /api/custom-variable/<name>/value?value=...`` - set a custom variable

Requests are queued and sent from a background thread, so a slow or unreachable
Companion can never stall scene switching or match clipping.

Mappings are a small text format, one per line (``#`` starts a comment)::

    START_MATCH = 1/0/0          # press page 1, row 0, column 0
    SHOW_PREVIEW = 1/0/1, 1/0/2  # several buttons are fine

The left side is the ``type`` of the scoring-system message.  Any type can be mapped;
MatchBox logs each new message type it sees so you can discover their names.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from matchbox import MatchBoxConfig

logger = logging.getLogger("matchbox")

KNOWN_EVENTS: tuple[str, ...] = ("SHOW_PREVIEW", "SHOW_MATCH", "START_MATCH")

VARIABLE_EVENT = "matchbox_event"
VARIABLE_FIELD = "matchbox_field"
VARIABLE_MATCH = "matchbox_match"

REQUEST_TIMEOUT_SECONDS = 2.0
FAILURE_LOG_INTERVAL_SECONDS = 30.0
MAX_QUEUE = 200

_LINE_RE = re.compile(r"^([A-Za-z0-9_]+)\s*=\s*(.+)$")
_LOCATION_RE = re.compile(r"^(\d{1,3})\s*/\s*(\d{1,3})\s*/\s*(\d{1,3})$")


@dataclass(frozen=True)
class ButtonLocation:
    """A Companion button, addressed the way Companion's HTTP API does (page 1+, row/column 0+)."""

    page: int
    row: int
    column: int

    @override
    def __str__(self) -> str:
        return f"{self.page}/{self.row}/{self.column}"


Mappings = dict[str, list[ButtonLocation]]


def parse_mappings(text: str) -> tuple[Mappings, list[str]]:
    """Parse the mapping text.  Returns (mappings, errors); bad lines are skipped, not fatal."""
    mappings: Mappings = {}
    errors: list[str] = []
    for number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = _LINE_RE.match(line)
        if not match:
            errors.append(f"Line {number}: expected EVENT = page/row/column")
            continue
        event = match.group(1).upper()
        locations: list[ButtonLocation] = []
        for part in match.group(2).split(","):
            loc = _LOCATION_RE.match(part.strip())
            if not loc or int(loc.group(1)) < 1:
                errors.append(f"Line {number}: '{part.strip()}' is not a valid button location (use page/row/column, e.g. 1/0/0; pages start at 1)")
                continue
            locations.append(ButtonLocation(int(loc.group(1)), int(loc.group(2)), int(loc.group(3))))
        if locations:
            mappings.setdefault(event, []).extend(locations)
    return mappings, errors


def _clean(value: object) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return str(int(value)) if float(value).is_integer() else str(value)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


@dataclass(frozen=True)
class _Request:
    path: str
    query: str
    description: str


class CompanionClient:
    """Sends Companion actions for scoring-system messages.  Safe to call from any thread."""

    def __init__(self, config: MatchBoxConfig) -> None:
        self._config: MatchBoxConfig = config
        self._queue: queue.Queue[_Request | None] = queue.Queue(maxsize=MAX_QUEUE)
        self._thread: threading.Thread | None = None
        self._thread_lock: threading.Lock = threading.Lock()
        self._parsed_text: str | None = None
        self._parsed: Mappings = {}
        self._seen_events: set[str] = set()
        self._last_failure_log: float = 0.0
        self._failing: bool = False
        # Never route Companion traffic through an HTTP proxy configured for the internet.
        self._opener: urllib.request.OpenerDirector = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    # ---- public API

    def handle_message(self, message: dict[str, object]) -> None:
        """Called with every parsed scoring-system message.  Never raises, never blocks."""
        try:
            self._handle_message(message)
        except Exception as e:
            logger.error(f"Companion: unexpected error handling message: {e}")

    def validate(self) -> list[str]:
        """Problems with the current settings (for the UI)."""
        _, errors = parse_mappings(self._config.companion_mappings)
        if not self._config.companion_host.strip():
            errors.insert(0, "Companion host is empty")
        return errors

    def stop(self) -> None:
        with self._thread_lock:
            thread = self._thread
            self._thread = None
        if thread is not None:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass

    # ---- internals

    def _mappings(self) -> Mappings:
        text = self._config.companion_mappings
        if text != self._parsed_text:
            self._parsed, errors = parse_mappings(text)
            self._parsed_text = text
            for error in errors:
                logger.warning(f"Companion mapping: {error}")
        return self._parsed

    def _handle_message(self, message: dict[str, object]) -> None:
        if not self._config.companion_enabled:
            return
        raw_type = message.get("type")
        if not isinstance(raw_type, str) or not raw_type:
            return
        event = raw_type.upper()

        mappings = self._mappings()
        locations = mappings.get(event, [])
        if event not in KNOWN_EVENTS and not locations:
            if event not in self._seen_events:
                self._seen_events.add(event)
                logger.info(f"Companion: scoring system sent '{event}' (add '{event} = page/row/column' to react to it)")
            return

        params_obj = message.get("params")
        params: dict[str, object] = params_obj if isinstance(params_obj, dict) else {}  # pyright: ignore[reportUnknownVariableType]

        # Variables first, so a pressed button's actions already see the new values.
        if self._config.companion_set_variables:
            self._enqueue_variable(VARIABLE_EVENT, event)
            field = _clean(message.get("field")) or _clean(params.get("field"))
            if field is not None:
                self._enqueue_variable(VARIABLE_FIELD, field)
            match_name = _clean(params.get("matchName")) or _clean(message.get("matchName"))
            if match_name is not None:
                self._enqueue_variable(VARIABLE_MATCH, match_name)

        for loc in locations:
            self._enqueue(_Request(
                path=f"/api/location/{loc.page}/{loc.row}/{loc.column}/press",
                query="",
                description=f"{event} -> pressed button {loc}",
            ))

    def _enqueue_variable(self, name: str, value: str) -> None:
        self._enqueue(_Request(
            path=f"/api/custom-variable/{urllib.parse.quote(name, safe='')}/value",
            query=urllib.parse.urlencode({"value": value}),
            description=f"set custom variable {name} = {value}",
        ))

    def _enqueue(self, request: _Request) -> None:
        self._ensure_thread()
        try:
            self._queue.put_nowait(request)
        except queue.Full:
            logger.warning("Companion: request queue full, dropping action")

    def _ensure_thread(self) -> None:
        with self._thread_lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="companion", daemon=True)
                self._thread.start()

    def _base_url(self) -> str | None:
        host = self._config.companion_host.strip()
        if not host or any(c in host for c in "/?#@ \t"):
            return None
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"  # bare IPv6 address
        return f"http://{host}:{int(self._config.companion_port)}"

    def _run(self) -> None:
        while True:
            request = self._queue.get()
            if request is None:
                return
            try:
                self._send(request)
            except Exception as e:  # a bad setting must not kill the worker
                self._report_failure(f"error sending {request.description}: {e}")

    def _send(self, request: _Request) -> None:
        base = self._base_url()
        if base is None:
            self._report_failure("Companion host is invalid")
            return
        url = base + request.path + (f"?{request.query}" if request.query else "")
        http_request = urllib.request.Request(url, data=b"", method="POST")
        try:
            with self._opener.open(http_request, timeout=REQUEST_TIMEOUT_SECONDS):
                pass
        except urllib.error.HTTPError as e:
            if e.code == 404 and "/custom-variable/" in request.path:
                name = request.path.split("/")[3]
                self._report_failure(
                    f"custom variable '{name}' doesn't exist in Companion "
                    + "(create it under Variables > Custom Variables, or turn off 'Set custom variables')",
                    transient=False,
                )
            else:
                self._report_failure(f"Companion answered HTTP {e.code} for {request.description}", transient=False)
            return
        except Exception as e:
            reason = getattr(e, "reason", e)
            self._report_failure(f"can't reach Companion at {base} ({reason})")
            return

        if self._failing:
            self._failing = False
            logger.info(f"Companion: connection restored ({base})")
        logger.info(f"Companion: {request.description}")

    def _report_failure(self, message: str, transient: bool = True) -> None:
        """Log failures, but at most every few seconds so a dead Companion doesn't flood the log."""
        if transient:
            self._failing = True
        now = time.time()
        if now - self._last_failure_log >= FAILURE_LOG_INTERVAL_SECONDS:
            self._last_failure_log = now
            logger.warning(f"Companion: {message}")
