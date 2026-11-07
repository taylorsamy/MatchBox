# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MatchBox is a Python desktop application that automates OBS scene switching and match video clipping for FIRST Tech Challenge (FTC) robotics events. It connects to the FTC scoring system via WebSocket, controls OBS via WebSocket, and serves clipped match videos via HTTP with mDNS discovery.

## Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the GUI application:**
```bash
python matchbox.py
```

**Run in CLI mode:**
```bash
python matchbox-cli.py --ftc-host <host> --event-code <code>
```

**Build standalone executables:**
```bash
python build.py
```

Note: There is no test suite.

**Type checking:**
```bash
basedpyright matchbox.py matchbox-cli.py matchbox-sync.py build.py web_api/ local_video_processor.py companion.py robot_photos.py
```

The project uses **basedpyright** in strict mode. When writing or modifying code:
- Avoid `Any` types — use `object`, `dict[str, object]`, or `cast()` instead
- Annotate class attributes, function parameters, and return types
- Use `_ =` for intentionally unused call results
- Prefix unused variables with `_` (e.g., `_dirs`)
- Use `from __future__ import annotations` with `TYPE_CHECKING` imports to avoid circular imports at runtime

## Architecture

### Core Components

**matchbox.py** - Main application containing:
- `MatchBoxConfig` - Configuration dataclass for all settings
- `MatchBoxCore` - Application controller handling WebSocket connections, OBS control, and video processing orchestration
- `MatchBoxGUI` - Tkinter GUI with sv-ttk theming

**matchbox-cli.py** - Command-line interface alternative to the GUI

**matchbox-sync.py** - Standalone rsync daemon for syncing match clips to a remote server

**local_video_processor.py** - Video clipping engine that uses ffmpeg subprocess calls to extract match segments from OBS recordings

**companion.py** - Bitfocus Companion integration. `CompanionClient.handle_message()` is called for every scoring-system WebSocket message; it queues Companion HTTP API calls (button press by `page/row/column`, custom variables) on a background thread so Companion can never block the asyncio loop. Mappings are the text setting `companion_mappings` (`SIGNAL = page/row/column`)

**robot_photos.py** - Robot photo pipeline. `RobotPhotoProcessor` stores uploads under `<clips_dir>/robot_photos/` (`originals/`, `processed/`, `thumbs/`) and processes them one at a time on a worker thread: EXIF rotate, U²-Net background removal through `onnxruntime` (`BackgroundRemover`, model downloaded lazily to `~/.u2net`), crop to subject, fit onto a fixed transparent canvas. Pillow/numpy/onnxruntime are imported lazily; without the model/onnxruntime photos are still resized and flagged `bg_removed: false`

**web_api/** - Web server components:
- `handler.py` - HTTP request handler with REST API routes, admin UI static file serving, and session-cookie authentication
- `websocket_server.py` - WebSocket server for real-time log streaming, status updates, and OBS WebSocket proxy
- `ws_tunnel_client.py` - WebSocket tunnel client that connects to a relay server for remote access

**web_admin/** - Static HTML/CSS/JS for the browser-based admin dashboard (served by `handler.py` at `/admin/`)

**web_photos/** - Single-page phone/tablet UI for photographing robots (served by `handler.py` at `/photos/`). Uses `<input type="file" capture>` rather than `getUserMedia`, because the server is plain HTTP and browsers only allow live camera access on HTTPS

**pi-server/relay_server.py** - Relay server (designed to run on a VPS/Pi) that bridges remote browser clients to MatchBox instances via WebSocket tunneling

### Data Flow

```
FTC Scoring WebSocket → MatchBoxCore → OBS Scene Switch
                              ↓
                     Match Start Event
                              ↓
                  LocalVideoProcessor (delayed)
                              ↓
                     ffmpeg Clip Extraction
                              ↓
                   match_clips/<event_code>/
                              ↓
                   HTTP Server (mDNS: ftcvideo.local)
```

### Threading Model

- Main thread: Tkinter GUI event loop
- Background thread: asyncio event loop for WebSocket monitoring
- Web server: ThreadingHTTPServer with multi-threaded request handling
- WebSocket server: Separate daemon thread with its own asyncio event loop
- Companion: one daemon worker thread draining a queue of HTTP requests
- Robot photos: one daemon worker thread processing uploads sequentially (ONNX session limited to 2 threads so OBS isn't starved)
- mDNS: Separate daemon thread for Zeroconf registration

### Web Admin & Remote Access

- **Local access**: Admin UI served at `/admin/`, API at `/api/`, clip downloads at `/`
- **Remote access**: WS tunnel client connects to relay server; relay proxies HTTP/WS requests back through the tunnel
- **Authentication**: HMAC-signed session cookies (`mb_session`), 24h expiry. Two-tier: instance password (per-instance) and admin password (hardcoded hash via `ADMIN_SALT`/`ADMIN_HASH`). Localhost requests bypass auth (for tunnel-proxied requests).
- **Robot photos**: `/photos/` (page), `/photos/api/photos|status|upload` and `/photos/img/<processed|thumbs>/<team>.png` are open to the local network like clip downloads (uploads take the raw image as the body, with an optional `X-Photo-Pin` header when `robot_photo_pin` is set). Originals are never served, and `/robot_photos/` is blocked from the clip file server. `/api/photos`, `/api/photos/reprocess` and `/api/photos/download-model` are authenticated admin routes. The rsync sync excludes `robot_photos/`
- **WebSocket paths**: `/ws/logs` (log streaming), `/ws/status` (status updates), `/ws/obs` (OBS WebSocket proxy)

### Key Technical Details

- Uses `obs-websocket-py` for OBS control with fallback methods for older API versions
- Custom HTTP handler implements Range requests for progressive video playback
- Configuration stored as JSON (`matchbox_config.json`) with CLI args taking priority
- Version derived from git tags via setuptools-scm
- Admin password hash generated via `generate_admin_hash.py`

### Build System

PyInstaller builds configured in `matchbox.spec`:
- Downloads platform-specific ffmpeg binaries automatically
- Produces standalone executables for Windows, macOS, and Linux
- CI/CD via GitHub Actions (`.github/workflows/pyinstaller.yml`)
