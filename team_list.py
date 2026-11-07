"""
Teams at the event, so robot photographers can pick a team instead of typing a number.

Two sources, merged:

* the ``robot_photo_teams`` setting - one team per line, ``number name`` (also accepts
  tab- or comma-separated text pasted from a spreadsheet or the FTC Events team list);
* best effort, the FTC scoring system's local REST API
  (``/api/v1/events/<code>/teams/`` and ``.../teams/<number>/`` for names).  That API is only
  documented inside the running scoring software (``/swagger-docs/``), so several response
  shapes are accepted and any failure just leaves the settings list in charge.

The scoring system is queried on a background thread and cached, so the photo page never
waits on it.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from matchbox import MatchBoxConfig

logger = logging.getLogger("matchbox")

REFRESH_SECONDS = 300.0  # re-read the scoring system's list this often
RETRY_SECONDS = 60.0  # after a failure
REQUEST_TIMEOUT = 3.0
NAME_FETCH_BUDGET = 15.0  # stop looking up team names after this long (numbers are still used)
MAX_TEAMS = 500

_LINE_RE = re.compile(r"^\s*0*([0-9]{1,6})(?![0-9])\s*[,;:|\t-]?\s*(.*)$")


def parse_team_list(text: str) -> dict[str, str]:
    """``number name`` per line -> {number: name}.  Lines that don't start with a number are ignored."""
    teams: dict[str, str] = {}
    for line in text.splitlines():
        match = _LINE_RE.match(line)
        if not match or int(match.group(1)) == 0:
            continue
        name = match.group(2)
        for sep in ("\t", ","):  # spreadsheet / CSV paste: keep only the column after the number
            if sep in name:
                name = name.split(sep, 1)[0]
        teams[str(int(match.group(1)))] = name.strip()
    return teams


def _team_number(value: object) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and 0 < value < 1_000_000:
        return str(value)
    if isinstance(value, str) and value.strip().isdigit() and 0 < int(value) < 1_000_000:
        return str(int(value))
    return None


def _name_from(entry: dict[str, object]) -> str:
    for key in ("name", "nameShort", "teamName", "shortName"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def parse_scoring_teams(data: object) -> dict[str, str]:
    """Accept the plausible shapes of a team-list response: a list of numbers or team objects,
    or an object holding one (``teamNumbers``, ``teams``, ...)."""
    items: object = data
    if isinstance(data, dict):
        d = cast("dict[str, object]", data)
        for key in ("teamNumbers", "teams", "teamList", "data"):
            if isinstance(d.get(key), list):
                items = d[key]
                break
    teams: dict[str, str] = {}
    if not isinstance(items, list):
        return teams
    for item in cast("list[object]", items)[:MAX_TEAMS]:
        if isinstance(item, dict):
            entry = cast("dict[str, object]", item)
            number = next((n for k in ("number", "teamNumber", "team") if (n := _team_number(entry.get(k)))), None)
            if number:
                teams[number] = _name_from(entry)
        elif number := _team_number(item):
            teams[number] = ""
    return teams


@dataclass
class TeamList:
    teams: dict[str, str] = field(default_factory=lambda: {})
    source: str = "none"  # "settings" | "scoring system" | "scoring system + settings" | "none"
    problem: str | None = None  # why the scoring system's list isn't available (if it isn't)
    loading: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "teams": [{"number": n, "name": self.teams[n]} for n in sorted(self.teams, key=int)],
            "source": self.source,
            "problem": self.problem,
            "loading": self.loading,
        }


class TeamDirectory:
    """Merged team list for the current event.  ``get()`` never blocks on the network."""

    def __init__(self, config: MatchBoxConfig) -> None:
        self._config: MatchBoxConfig = config
        self._lock: threading.Lock = threading.Lock()
        self._live: dict[str, str] = {}
        self._live_key: tuple[str, int, str] | None = None
        self._live_problem: str | None = None
        self._fetched_at: float = 0.0
        self._refreshing: bool = False
        self._opener: urllib.request.OpenerDirector = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _key(self) -> tuple[str, int, str] | None:
        host = self._config.scoring_host.strip()
        code = self._config.event_code.strip()
        if not host or not code or any(c in host for c in "/?#@ \t"):
            return None
        try:
            port = int(self._config.scoring_port)
        except (TypeError, ValueError):
            return None
        return (host, port, code)

    def get(self) -> TeamList:
        manual = parse_team_list(self._config.robot_photo_teams)
        key = self._key()
        with self._lock:
            if key != self._live_key:  # event or scoring system changed: forget the old list
                self._live, self._live_key, self._live_problem, self._fetched_at = {}, key, None, 0.0
            age = time.time() - self._fetched_at
            stale = age > (REFRESH_SECONDS if self._live else RETRY_SECONDS)
            if key is not None and stale and not self._refreshing:
                self._refreshing = True
                _ = threading.Thread(target=self._refresh, args=(key,), name="team-list", daemon=True).start()
            live = dict(self._live)
            problem = self._live_problem if key is not None else "Scoring system host or event code not set"
            loading = self._refreshing and not live

        merged = dict(live)
        for number, name in manual.items():
            merged[number] = name or merged.get(number, "")
        if live and manual:
            source = "scoring system + settings"
        elif live:
            source = "scoring system"
        elif manual:
            source = "settings"
        else:
            source = "none"
        return TeamList(teams=merged, source=source, problem=None if live else problem, loading=loading)

    # ---- background refresh

    def _refresh(self, key: tuple[str, int, str]) -> None:
        try:
            teams = self._fetch(key)
            problem = None if teams else "The scoring system returned no teams for this event"
        except Exception as e:
            teams, problem = {}, f"Couldn't read the team list from the scoring system ({e})"
        with self._lock:
            self._refreshing = False
            if key != self._live_key:
                return  # settings changed while we were fetching
            if teams or not self._live:
                self._live = teams
            self._live_problem = problem
            self._fetched_at = time.time()
        if teams:
            logger.info(f"📷 Loaded {len(teams)} teams from the scoring system for the robot photo page")

    def _get_json(self, url: str) -> object:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with self._opener.open(request, timeout=REQUEST_TIMEOUT) as response:  # pyright: ignore[reportAny]
            return cast(object, json.loads(cast(bytes, response.read(2 * 1024 * 1024))))  # pyright: ignore[reportAny]

    def _fetch(self, key: tuple[str, int, str]) -> dict[str, str]:
        host, port, code = key
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        base = f"http://{host}:{port}/api/v1/events/{urllib.parse.quote(code, safe='')}/teams/"
        try:
            teams = parse_scoring_teams(self._get_json(base))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}") from None
        except urllib.error.URLError as e:
            raise RuntimeError(str(e.reason)) from None

        # Names, if the list didn't include them: one request per team, within a time budget
        deadline = time.time() + NAME_FETCH_BUDGET
        for number in [n for n, name in teams.items() if not name]:
            if time.time() > deadline:
                break
            try:
                detail = self._get_json(base + f"{number}/")
            except urllib.error.HTTPError:
                break  # no per-team endpoint: don't try the rest
            except Exception:
                continue
            if isinstance(detail, dict):
                teams[number] = _name_from(cast("dict[str, object]", detail))
        return teams
