/** MatchBox Web Admin UI */

// --- API Client ---
function getRelayPrefix() {
    // Detect relay-proxied access: URL ends with /{instance_id}/admin[/]
    // e.g. /FTC/MatchBox/uswacmp/admin -> "/FTC/MatchBox/uswacmp"
    const pathParts = window.location.pathname.split('/').filter(Boolean);
    const adminIdx = pathParts.lastIndexOf('admin');
    if (adminIdx >= 1) {
        return '/' + pathParts.slice(0, adminIdx).join('/');
    }
    return null;
}

function getApiBase() {
    const prefix = getRelayPrefix();
    if (prefix) return prefix + '/api/';
    return '/api/';
}

const API = {
    async get(path) {
        const res = await fetch(getApiBase() + path);
        return res.json();
    },
    async post(path, body) {
        const res = await fetch(getApiBase() + path, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: body ? JSON.stringify(body) : undefined,
        });
        return res.json();
    }
};

// --- State ---
let wsStatus = null;
let wsLogs = null;
let currentStatus = {};
let autoScroll = true;

// --- Tab switching ---
let obsFrameLoaded = false;

function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabId);
    });
    document.querySelectorAll('.tab-content').forEach(el => {
        el.classList.toggle('active', el.id === 'tab-' + tabId);
    });

    // Lazy-load obs-web on first visit to OBS tab
    if (tabId === 'obs' && !obsFrameLoaded) {
        obsFrameLoaded = true;
        const wsBase = getWsBase();
        const password = getVal('cfg-obs-password') || '';
        const prefix = getRelayPrefix();
        const obsWebPath = prefix ? `${prefix}/obs-web/` : '/obs-web/';
        const obsUrl = `${obsWebPath}#${wsBase}/ws/obs` + (password ? `#${password}` : '');

        const frame = document.getElementById('obs-frame');
        if (frame) frame.src = obsUrl;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    // Tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });

    // Load initial data
    loadConfig();
    loadClips();
    loadStatus();
    loadPhotos();

    // Connect WebSockets
    connectStatusWS();
    connectLogsWS();

    // Button handlers
    document.getElementById('btn-start').addEventListener('click', startMatchbox);
    document.getElementById('btn-stop').addEventListener('click', stopMatchbox);
    document.getElementById('btn-configure-obs').addEventListener('click', configureObs);
    document.getElementById('btn-save-config').addEventListener('click', saveConfig);
    document.getElementById('btn-start-sync').addEventListener('click', startSync);
    document.getElementById('btn-stop-sync').addEventListener('click', stopSync);
    document.getElementById('btn-start-tunnel').addEventListener('click', startTunnel);
    document.getElementById('btn-stop-tunnel').addEventListener('click', stopTunnel);
    document.getElementById('btn-photos-reprocess').addEventListener('click', reprocessPhotos);
    document.getElementById('btn-photos-download').addEventListener('click', downloadPhotoModel);

    // Auto-scroll toggle
    const logContainer = document.getElementById('log-container');
    logContainer.addEventListener('scroll', () => {
        const atBottom = logContainer.scrollHeight - logContainer.scrollTop - logContainer.clientHeight < 50;
        autoScroll = atBottom;
    });

    // Periodic refresh
    setInterval(loadClips, 30000);
    setInterval(loadStatus, 10000);
    setInterval(loadPhotos, 15000);
});

// --- WebSocket connections ---
function getWsBase() {
    const loc = window.location;
    const prefix = getRelayPrefix();

    if (prefix) {
        // Accessed via relay - WS routes through relay at same origin
        const wsProto = loc.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${wsProto}//${loc.host}${prefix}`;
    }

    // Direct access - WS server is on port+1
    const wsPort = parseInt(loc.port || (loc.protocol === 'https:' ? '443' : '80')) + 1;
    return `ws://${loc.hostname}:${wsPort}`;
}

function connectStatusWS() {
    if (wsStatus) { try { wsStatus.close(); } catch(e) {} }

    wsStatus = new WebSocket(getWsBase() + '/ws/status');
    wsStatus.onmessage = (e) => {
        try {
            currentStatus = JSON.parse(e.data);
            updateStatusUI(currentStatus);
        } catch(err) {}
    };
    wsStatus.onclose = () => { setTimeout(connectStatusWS, 3000); };
    wsStatus.onerror = () => {};
}

function connectLogsWS() {
    if (wsLogs) { try { wsLogs.close(); } catch(e) {} }

    wsLogs = new WebSocket(getWsBase() + '/ws/logs');
    wsLogs.onmessage = (e) => {
        try {
            const entry = JSON.parse(e.data);
            appendLog(entry);
        } catch(err) {}
    };
    wsLogs.onclose = () => { setTimeout(connectLogsWS, 3000); };
    wsLogs.onerror = () => {};
}

// --- Status UI ---
function updateStatusUI(status) {
    // Status dots
    setDot('dot-running', status.running);
    setDot('dot-obs', status.obs_connected);
    setDot('dot-ftc', status.ftc_connected);
    setDot('dot-tunnel', status.tunnel_connected);

    // Overview cards
    setText('status-running', status.running ? 'Running' : 'Stopped');
    setText('status-field', status.current_field != null ? 'Field ' + status.current_field : '--');
    setText('status-clips', status.clips_count != null ? status.clips_count : '--');
    setText('status-event', status.event_code || '--');

    // Start/Stop button states
    const startBtn = document.getElementById('btn-start');
    const stopBtn = document.getElementById('btn-stop');
    if (startBtn) startBtn.disabled = !!status.running;
    if (stopBtn) stopBtn.disabled = !status.running;

    // Sync button states
    const startSyncBtn = document.getElementById('btn-start-sync');
    const stopSyncBtn = document.getElementById('btn-stop-sync');
    if (startSyncBtn) startSyncBtn.disabled = !!status.sync_running;
    if (stopSyncBtn) stopSyncBtn.disabled = !status.sync_running;

    // Tunnel button states
    const startTunnelBtn = document.getElementById('btn-start-tunnel');
    const stopTunnelBtn = document.getElementById('btn-stop-tunnel');
    if (startTunnelBtn) startTunnelBtn.disabled = !!status.tunnel_connected;
    if (stopTunnelBtn) stopTunnelBtn.disabled = !status.tunnel_connected;
}

function setDot(id, on) {
    const el = document.getElementById(id);
    if (el) el.classList.toggle('on', !!on);
}

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

// --- Load data ---
async function loadStatus() {
    try {
        const status = await API.get('status');
        currentStatus = status;
        updateStatusUI(status);
    } catch(e) {}
}

async function loadConfig() {
    try {
        const cfg = await API.get('config');
        populateConfigForm(cfg);
    } catch(e) {}
}

async function loadClips() {
    try {
        const clips = await API.get('clips');
        const list = document.getElementById('clips-list');
        if (!list) return;

        if (clips.length === 0) {
            list.innerHTML = '<p style="color: var(--text-muted)">No match clips available yet...</p>';
            return;
        }

        list.innerHTML = clips.map(c => {
            const sizeMB = (c.size / (1024 * 1024)).toFixed(1);
            return `<div class="clip-item">
                <a href="/${c.name}" target="_blank">${c.name}</a>
                <span class="clip-size">${sizeMB} MB</span>
            </div>`;
        }).join('');
    } catch(e) {}
}

// --- Config form ---
function populateConfigForm(cfg) {
    setVal('cfg-event-code', cfg.event_code);
    setVal('cfg-scoring-host', cfg.scoring_host);
    setVal('cfg-scoring-port', cfg.scoring_port);
    setVal('cfg-obs-host', cfg.obs_host);
    setVal('cfg-obs-port', cfg.obs_port);
    setVal('cfg-obs-password', cfg.obs_password);
    setVal('cfg-output-dir', cfg.output_dir);
    setVal('cfg-mdns-name', cfg.mdns_name);
    setVal('cfg-web-port', cfg.web_port);
    setVal('cfg-pre-buffer', cfg.pre_match_buffer_seconds);
    setVal('cfg-post-buffer', cfg.post_match_buffer_seconds);
    setVal('cfg-match-duration', cfg.match_duration_seconds);

    // Scene mappings
    const fsm = cfg.field_scene_mapping || {};
    setVal('cfg-scene-1', fsm['1'] || fsm[1] || '');
    setVal('cfg-scene-2', fsm['2'] || fsm[2] || '');
    setVal('cfg-scene-3', fsm['3'] || fsm[3] || '');

    // Rsync
    setVal('cfg-rsync-host', cfg.rsync_host);
    setVal('cfg-rsync-module', cfg.rsync_module);
    setVal('cfg-rsync-username', cfg.rsync_username);
    setVal('cfg-rsync-password', cfg.rsync_password);
    setVal('cfg-rsync-interval', cfg.rsync_interval_seconds);

    // Tunnel
    setVal('cfg-tunnel-relay-url', cfg.tunnel_relay_url);
    setVal('cfg-tunnel-password', cfg.tunnel_password);
    const allowAdmin = document.getElementById('cfg-tunnel-allow-admin');
    if (allowAdmin) allowAdmin.checked = cfg.tunnel_allow_admin !== false;

    // Companion
    setChecked('cfg-companion-enabled', cfg.companion_enabled === true);
    setVal('cfg-companion-host', cfg.companion_host);
    setVal('cfg-companion-port', cfg.companion_port);
    setChecked('cfg-companion-variables', cfg.companion_set_variables !== false);
    setVal('cfg-companion-mappings', cfg.companion_mappings);

    // Robot photos
    setChecked('cfg-photos-enabled', cfg.robot_photos_enabled !== false);
    setVal('cfg-photo-width', cfg.robot_photo_width);
    setVal('cfg-photo-height', cfg.robot_photo_height);
    setVal('cfg-photo-model', cfg.robot_photo_model === 'u2net' ? 'u2net' : 'birefnet-512');
    setVal('cfg-photo-pin', cfg.robot_photo_pin);
    setVal('cfg-photo-teams', cfg.robot_photo_teams);
    updatePhotosUrl();
}

function getConfigFromForm() {
    return {
        event_code: getVal('cfg-event-code'),
        scoring_host: getVal('cfg-scoring-host'),
        scoring_port: parseInt(getVal('cfg-scoring-port')) || 80,
        obs_host: getVal('cfg-obs-host'),
        obs_port: parseInt(getVal('cfg-obs-port')) || 4455,
        obs_password: getVal('cfg-obs-password'),
        output_dir: getVal('cfg-output-dir'),
        mdns_name: getVal('cfg-mdns-name'),
        web_port: parseInt(getVal('cfg-web-port')) || 80,
        pre_match_buffer_seconds: parseInt(getVal('cfg-pre-buffer')) || 10,
        post_match_buffer_seconds: parseInt(getVal('cfg-post-buffer')) || 10,
        match_duration_seconds: parseInt(getVal('cfg-match-duration')) || 158,
        field_scene_mapping: {
            1: getVal('cfg-scene-1'),
            2: getVal('cfg-scene-2'),
            3: getVal('cfg-scene-3'),
        },
        rsync_host: getVal('cfg-rsync-host'),
        rsync_module: getVal('cfg-rsync-module'),
        rsync_username: getVal('cfg-rsync-username'),
        rsync_password: getVal('cfg-rsync-password'),
        rsync_interval_seconds: parseInt(getVal('cfg-rsync-interval')) || 60,
        tunnel_relay_url: getVal('cfg-tunnel-relay-url'),
        tunnel_password: getVal('cfg-tunnel-password'),
        tunnel_allow_admin: document.getElementById('cfg-tunnel-allow-admin')?.checked ?? true,
        companion_enabled: getChecked('cfg-companion-enabled', false),
        companion_host: getVal('cfg-companion-host').trim() || '127.0.0.1',
        companion_port: parseInt(getVal('cfg-companion-port')) || 8000,
        companion_set_variables: getChecked('cfg-companion-variables', true),
        companion_mappings: getVal('cfg-companion-mappings'),
        robot_photos_enabled: getChecked('cfg-photos-enabled', true),
        robot_photo_width: parseInt(getVal('cfg-photo-width')) || 1200,
        robot_photo_height: parseInt(getVal('cfg-photo-height')) || 900,
        robot_photo_model: getVal('cfg-photo-model') || 'birefnet-512',
        robot_photo_pin: getVal('cfg-photo-pin').trim(),
        robot_photo_teams: getVal('cfg-photo-teams'),
    };
}

function setVal(id, val) {
    const el = document.getElementById(id);
    if (el) el.value = val != null ? val : '';
}

function getVal(id) {
    const el = document.getElementById(id);
    return el ? el.value : '';
}

function setChecked(id, checked) {
    const el = document.getElementById(id);
    if (el) el.checked = !!checked;
}

function getChecked(id, fallback) {
    const el = document.getElementById(id);
    return el ? el.checked : fallback;
}

// --- Actions ---
async function saveConfig() {
    try {
        const cfg = getConfigFromForm();
        await API.post('config', cfg);
        const result = await API.post('save-config');
        alert(result.ok ? 'Configuration saved!' : 'Error: ' + (result.error || 'Unknown'));
    } catch(e) {
        alert('Error saving config: ' + e.message);
    }
}

async function configureObs() {
    try {
        const cfg = getConfigFromForm();
        await API.post('config', cfg);
        const result = await API.post('configure-obs');
        alert(result.ok ? 'OBS scenes configured!' : 'Failed to configure OBS scenes');
    } catch(e) {
        alert('Error: ' + e.message);
    }
}

async function startMatchbox() {
    try {
        const result = await API.post('start');
        if (!result.ok) {
            alert(result.message || result.error || 'Could not start');
        }
    } catch(e) {
        alert('Error: ' + e.message);
    }
}

async function stopMatchbox() {
    try {
        await API.post('stop');
    } catch(e) {
        alert('Error: ' + e.message);
    }
}

async function startSync() {
    try {
        const result = await API.post('sync/start');
        if (!result.ok) alert(result.error || 'Could not start sync');
    } catch(e) {
        alert('Error: ' + e.message);
    }
}

async function stopSync() {
    try {
        const result = await API.post('sync/stop');
        if (!result.ok) alert(result.error || 'Could not stop sync');
    } catch(e) {
        alert('Error: ' + e.message);
    }
}

async function startTunnel() {
    try {
        const cfg = getConfigFromForm();
        await API.post('config', cfg);
        const result = await API.post('tunnel/start');
        if (!result.ok) alert(result.error || 'Could not start tunnel');
    } catch(e) {
        alert('Error: ' + e.message);
    }
}

async function stopTunnel() {
    try {
        const result = await API.post('tunnel/stop');
        if (!result.ok) alert(result.error || 'Could not stop tunnel');
    } catch(e) {
        alert('Error: ' + e.message);
    }
}

// --- Robot photos ---
function updatePhotosUrl() {
    const el = document.getElementById('photos-url');
    if (!el) return;
    const prefix = getRelayPrefix() || '';
    const urls = [`${window.location.origin}${prefix}/photos/`];
    const name = getVal('cfg-mdns-name');
    const port = parseInt(getVal('cfg-web-port')) || 80;
    if (!prefix && name) {
        const mdns = `http://${name}${port === 80 ? '' : ':' + port}/photos/`;
        if (!urls.includes(mdns)) urls.push(mdns);
    }
    el.textContent = '';
    urls.forEach((url, i) => {
        if (i > 0) el.append(' \u00b7 ');
        const a = document.createElement('a');
        a.href = url; a.target = '_blank'; a.textContent = url;
        a.style.color = 'var(--accent)';
        el.append(a);
    });
}

async function loadPhotos() {
    try {
        renderPhotos(await API.get('photos'));
    } catch(e) {}
}

function renderPhotos(data) {
    const caps = data.capabilities || {};
    let status;
    if (caps.dependencies_ok === false) {
        status = 'Background removal unavailable: ' + (caps.problem || 'missing libraries');
    } else if (caps.problem) {
        status = 'Background removal problem: ' + caps.problem;
    } else if (caps.model_ready) {
        status = `Ready: ${caps.model_label || caps.model}, photos are ${caps.width} \u00d7 ${caps.height} px.`;
    } else {
        status = 'Model not downloaded yet. It downloads automatically with the first photo (internet needed once), or use "Download model now".';
    }
    setText('photos-model-status', status);

    const tl = data.team_list || {};
    let teamStatus;
    if (tl.count) {
        teamStatus = `${tl.count} teams available to photographers (from ${tl.source}).`;
        if (tl.source === 'settings' && tl.problem) teamStatus += ` Scoring system list not used: ${tl.problem}`;
    } else if (tl.loading) {
        teamStatus = 'Reading the team list from the scoring system...';
    } else {
        teamStatus = 'No team list: photographers type team numbers. ' + (tl.problem ? `(${tl.problem})` : '');
    }
    setText('photos-team-status', teamStatus);

    const photos = data.photos || [];
    const done = photos.filter(p => p.state === 'done').length;
    setText('photos-count', photos.length ? `(${done})` : '');

    const grid = document.getElementById('photos-grid');
    if (!grid) return;
    grid.textContent = '';
    if (photos.length === 0) {
        const p = document.createElement('p');
        p.style.color = 'var(--text-muted)';
        p.textContent = 'No robot photos yet...';
        grid.appendChild(p);
        return;
    }
    const prefix = getRelayPrefix() || '';
    photos.forEach(photo => {
        const tile = document.createElement('div');
        tile.className = 'photo-tile';

        const thumb = document.createElement('div');
        thumb.className = 'thumb';
        if (photo.has_image) {
            const img = document.createElement('img');
            img.loading = 'lazy';
            img.alt = 'Team ' + photo.team;
            img.src = `${prefix}/photos/img/thumbs/${photo.team}.png?v=${Math.round((photo.updated || 0) * 1000)}`;
            const link = document.createElement('a');
            link.href = `${prefix}/photos/img/processed/${photo.team}.png`;
            link.target = '_blank';
            link.style.display = 'contents';
            link.appendChild(img);
            thumb.appendChild(link);
        } else {
            thumb.classList.add('empty');
            thumb.textContent = photo.state === 'error' ? 'Failed' : 'Working...';
        }
        tile.appendChild(thumb);

        const name = document.createElement('div');
        name.className = 'name';
        name.textContent = 'Team ' + photo.team + (photo.name ? ' \u00b7 ' + photo.name : '');
        tile.appendChild(name);

        const badgeText = photo.state === 'error' ? 'failed'
            : (photo.state === 'queued' || photo.state === 'processing') ? 'updating'
            : (photo.state === 'done' && photo.bg_removed === false) ? 'no cut-out' : '';
        if (badgeText) {
            const badge = document.createElement('span');
            badge.className = 'badge' + (photo.state === 'error' ? ' err' : '');
            badge.textContent = badgeText;
            if (photo.message) badge.title = photo.message;
            tile.appendChild(badge);
        }
        grid.appendChild(tile);
    });
}

async function reprocessPhotos() {
    try {
        await API.post('config', getConfigFromForm());  // use the size/model currently shown
        const result = await API.post('photos/reprocess');
        alert(result.ok ? `Reprocessing ${result.count} photo(s)...` : 'Error: ' + (result.error || 'Unknown'));
        setTimeout(loadPhotos, 1500);
    } catch(e) {
        alert('Error: ' + e.message);
    }
}

async function downloadPhotoModel() {
    try {
        await API.post('config', getConfigFromForm());
        const result = await API.post('photos/download-model');
        alert(result.ok ? 'Downloading in the background. Progress is shown in the Logs tab.' : 'Error: ' + (result.error || 'Unknown'));
        // Check back a few times so the status line updates when the download finishes
        [5000, 15000, 30000, 60000].forEach(ms => setTimeout(loadPhotos, ms));
    } catch(e) {
        alert('Error: ' + e.message);
    }
}

// --- Log display ---
function appendLog(entry) {
    const container = document.getElementById('log-container');
    if (!container) return;

    const div = document.createElement('div');
    div.className = 'log-entry ' + (entry.level || 'INFO');
    div.textContent = `${entry.timestamp || ''} [${entry.level || 'INFO'}] ${entry.message || ''}`;
    container.appendChild(div);

    // Limit log entries to prevent memory issues
    while (container.children.length > 1000) {
        container.removeChild(container.firstChild);
    }

    if (autoScroll) {
        container.scrollTop = container.scrollHeight;
    }
}
