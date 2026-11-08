# MatchBox™ for *FIRST®* Tech Challenge

MatchBox is a standalone application that integrates with OBS and the *FIRST®* Tech Challenge scoring system to provide:
- Automatic setup of scenes in OBS, including the *FIRST®* Tech Challenge scoring overlay
- Automatic real-time scene switching based on field events
- Fully automatic match video clipping
- Local web interface for easy access to match clips
- Automatic video upload
  - Coming soon: Full integration with https://www.clipfarm.watch/

<img width="941" height="981" alt="image" src="https://github.com/user-attachments/assets/3fbc17b2-c916-47e9-88b6-03cd474c3b4e" />

> [!IMPORTANT]
> **MatchBox is (currently) an unofficial, community-ran project.  While we hope to someday have it become an official part of FTC, similar to FRC's Webcast Unit, currently MatchBox is entirely community-supported, so please do not bother *FIRST®* with any issues you may encounter.**
> 
> *FIRST®, FIRST® Robotics Competition, and FIRST® Tech Challenge, are registered trademarks of FIRST® (www.firstinspires.org) which is not overseeing, involved with, or responsible for this activity, product, or service.*

## Features

- **OBS Integration**: Automatically configures OBS scenes with the FTC scoring system overlay
  - No more messing around with broken chroma key setups, awkward scoring overlay configuration, etc.
- **Automatic Scene Switching**: Connects to FTC scoring system, and automatically switches OBS scenes based on active field
- **Match Video Processing**: Automatic match detection and splitting
  - Clips saved locally for immediate access
- **Local Web Interface**: Locally serves match clips for easy access by both teams and event staff, even when the events's external Wi-Fi may be slow, unreliable, or even non-existant
  - mDNS support for easy access (http://ftcvideo.local)
  - Perfect for anyone on the event network to review matches, even without a full internet connection
- **Full remote administration**: Set an instance password, and you can have your team connect to MatchBox from anywhere, whether across the room or across the whole world, as long as you have an internet connection
  - Optionally, enable the option to let the MatchBox team connect to your instance.  Let the experts help keep your stream running smoothly, no matter where you are in the world
- **Bitfocus Companion Integration**: Press Companion buttons and set Companion custom variables from the scoring system's match signals (preview, show match, match start, ...), so Stream Decks, tally lights, graphics, etc. can follow the match automatically
- **Robot Photos**: A phone-friendly page on the local web server for photographing robots. Photographers pick the team from the event's team list, backgrounds are removed automatically, and every photo is fitted onto the same size canvas and saved under its team number, ready to use in overlays and galleries
- **Clipping continues even if the internet doesn't**: Internet at the event dropped out?  No worries!  MatchBox handles match clipping locally, and will upload the clips whenever internet returns.  No more loosing match footage forever because the internet dropped out for a time; MatchBox only needs the local connection to the scoring system to keep clipping away.

<img width="941" height="997" alt="image" src="https://github.com/user-attachments/assets/4b4467ca-b73b-40d4-bd8b-269f168e7924" />

## Usage

### Setting up OBS
- Enable WebSocket server in OBS:
   - Tools > WebSocket Server Settings
   - Enable WebSocket server
   - Set a password
   - Note the port (usually 4455)

### Configuration
- Fill in the Connection Settings tab:
   - **Event Code**: Your FTC event code
   - **Scoring System Host**: Usually IP of scoring computer
   - **OBS WebSocket Password**: Set in OBS Tools > WebSocket Server Settings
- (Optional) Configure Scene Mapping tab to match your OBS scene names
- Set output directory in Video & Web Settings
- Click "Configure OBS Scenes" to auto-create:
   - Field scenes (Field 1, Field 2, etc.)
   - Shared FTC overlay browser source with chroma key filter
   - Overlay automatically added to all field scenes
- Fill in the fields in the Sync Settings tab to allow match upload
   - Drop me a message to get an access code
   - Match clips will be uploaded to https://jacobspctuneup.tk/FTC/MatchBox/
   - Coming soon: Full integration with <https://www.clipfarm.watch/>

### Operation
1. Start the FTC scoring system
2. Start OBS and begin streaming and recording
3. Launch MatchBox (GUI or CLI)
4. MatchBox will:
   - Connect to scoring system WebSocket
   - Switch OBS scenes based on active field
   - Start local web server for match clips
   - Process and split match videos

### Accessing Match Clips
- **Local network (mDNS)**: http://ftcvideo.local:80 (or your configured port)
- **Local network (manual)**: http://[computer-ip]:80 (or your configured port)
- **Same computer**: http://localhost:80 (or your configured port)

### Bitfocus Companion
MatchBox can react to the scoring system's signals by pressing Companion buttons (Companion 3.0 or newer, using its HTTP remote control).
1. In Companion, note the HTTP port (default `8000`).
2. In MatchBox open the **Companion** tab (or the same tab in the web admin), tick **Enable Companion integration**, and enter Companion's host and port.
3. Map scoring system signals to buttons, one per line as `SIGNAL = page/row/column`:
   ```
   SHOW_PREVIEW = 1/0/0
   SHOW_MATCH = 1/0/1
   START_MATCH = 1/0/2, 1/0/3   # several buttons are fine
   ```
   `SHOW_PREVIEW`, `SHOW_MATCH` and `START_MATCH` are the signals MatchBox already uses. Any other message type the scoring system sends is written to the log once (`Companion: scoring system sent 'XYZ'`), and can be mapped the same way.
4. Optionally create the custom variables `matchbox_event`, `matchbox_field` and `matchbox_match` in Companion (Variables → Custom Variables). MatchBox updates them before pressing any buttons, so button text and triggers can show e.g. `$(custom:matchbox_match)`.

Companion calls never delay scene switching or clipping: if Companion is unreachable MatchBox just logs a warning (at most every 30 seconds) and carries on. If you run `matchbox-cli.py` on the same computer as Companion, change `--web-port` (the CLI also defaults to 8000).

### Robot Photos
1. Optionally list the event's teams in the **Photos** tab (one per line: `number name`; text pasted from a spreadsheet works). MatchBox also tries to read the team list from the scoring system, and merges the two.
2. On a phone or tablet on the event network, open `http://ftcvideo.local/photos/` (or `http://[computer-ip]/photos/`).
3. Pick the team: type the number (matching teams and their names are suggested), or tap it under **Still to photograph**. The page shows the team's name to confirm, and warns if the number isn't on the event's list.
4. Tap **Take photo**, check the preview (it says which team it will be saved as), and tap **Upload**. You can photograph the next robot straight away: processing happens in the background, and the page tells you when each photo is ready.
5. Tap a photo under **Photographed teams** to see it full size, or to retake it.

MatchBox removes the background, drops stray objects that aren't part of the robot (like a loose ball on the floor), and fits the robot onto a fixed-size transparent canvas (1200 × 900 by default).

**Using the photos (e.g. in an overlay):** every photo is named after its team number.

- Files: `<output dir>/<event code>/robot_photos/processed/<team>.png`. Thumbnails are in `thumbs/`, and every original upload is kept in `originals/`.
- URL: `http://ftcvideo.local/photos/img/processed/<team>.png`, usable as an OBS Browser or Image source.
- List: `http://ftcvideo.local/photos/api/photos` is a JSON list of all photos, with team number, name and URL.
- A retake replaces the team's file, so an overlay pointing at a team always shows the latest photo.
- Photos are not uploaded by Sync.

**Background removal models** (Photos tab), chosen by comparing methods on real event photos (see `experiments/background_removal/`):

- **BiRefNet 512 px** (default): the best cut-outs, with clean open frames and solid robots. It is a 941 MB one-time download, uses about 3.3 GB of RAM while processing, and takes ~7 s per photo on 2 CPU threads. The model is released after 10 idle minutes.
- **U²-Net**: a 176 MB download, about 1 GB of RAM, and under a second per photo, but it tends to fill in open robot frames with floor. Use it on low-memory computers.

Notes:

- **Download the model before the event.** It is downloaded the first time it's needed and stored in `~/.u2net` (set `U2NET_HOME` to change this). Use **Download model now** in the Photos tab while you have good internet. Without the model, or without the `onnxruntime` package, photos are still saved and resized, just without the cut-out. They're marked "no cut-out" so you can **Reprocess all photos** later.
- Anyone on the network can open the page. Set an **Upload PIN** in the Photos tab to require one for uploads.
- Processing uses at most two CPU threads and handles one photo at a time, so it won't starve OBS on the same computer.
- Photo capture is meant for the local network. Uploads can't go through the Remote Access relay, which limits request sizes to about 1 MB.
- The scoring-system team list uses the scoring system's local API (`/api/v1/events/<code>/teams/`), which is only documented inside the scoring software. If it isn't available, the Photos tab says why, and the list from settings is used.

## Troubleshooting

### OBS Connection Issues
- Check OBS WebSocket server is enabled (Tools > WebSocket Server Settings)
- Verify password matches
- Ensure OBS is running before starting MatchBox
- Check Windows firewall isn't blocking connections

### FTC Scoring System Issues
- Verify event code is correct
- Check scoring system is running and accessible
- Test WebSocket URL in browser: `ws://<scoring system host>/stream/display/command/?code=YOUREVENTCODE`

### Web Interface Issues
- Check if port is already in use
- Try different port number
- Verify output directory permissions
- Check firewall settings for HTTP access

### Companion Buttons Not Pressing
- Check the Companion host/port, and that the Companion HTTP API is reachable from this computer
- Button locations are `page/row/column` as in Companion's HTTP API (page starts at 1, row and column at 0)
- Watch the log: every action is logged (`Companion: START_MATCH -> pressed button 1/0/2`), as are failures
- `custom variable ... doesn't exist` means the variable has to be created in Companion first

### Robot Photos
- No cut-out / "no cut-out" badge: see the Photos tab for why (usually the model has not been downloaded yet), then **Reprocess all photos**
- No team list on the photo page: check the scoring system host and event code, or list the teams in the Photos tab. The tab shows where the list came from, or why the scoring system's list couldn't be read
- Two robots in one photo: background removal keeps both. Retake with only the team's robot in frame
- Processing is slow or the computer runs out of memory: switch the model to U²-Net in the Photos tab
- Can't open the page from a phone: use the computer's IP address instead of `ftcvideo.local` (some phones don't resolve `.local` names)

### Scene Switching Not Working
- Verify field-to-scene mapping matches OBS scene names
- Check OBS scenes exist
- Monitor log for WebSocket messages
- Test manual scene switching in OBS

## Developer Information

### Software Dependencies
- Python 3.12+
- `ffmpeg` and `ffprobe` for video encoding
- `rsync` for rsync-based remote clip uploading
- OBS Studio with WebSocket server enabled
- Official *FIRST* Tech Challenge Live Scoring System accessible on the network

### Python setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For building binaries with pyinstaller:

```bash
pip install -r pyinstaller-build-reqs.txt
pyinstaller matchbox.spec
```
