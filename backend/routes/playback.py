"""
Playback endpoints. The browser opens the stream with EventSource (Server-Sent Events):
one-way, plain HTTP, reconnect-friendly, and works through most hosting proxies —
playback only ever sends data server -> browser, so WebSockets would add nothing.
"""
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from backend.core.playback import load_for_playback_cached, playback_info, stream_frames
from backend.utils.session_registry import get_session

router = APIRouter(prefix="/playback", tags=["playback"])


@router.get("/{session_id}/info")
def info(session_id: str):
    return playback_info(load_for_playback_cached(get_session(session_id)))


@router.get("/{session_id}/stream")
def stream(session_id: str, start: float = 0.0, speed: float = 1.0, channels: Optional[str] = None,
           prefill: float = 0.0, preview: bool = False):
    """
    `channels` is comma-separated. `prefill` sends the preceding seconds at once so a
    seek shows a full screen; `preview` stops after that (scrubbing while paused).
    A sync generator: Starlette iterates it in a worker thread, so the time.sleep pacing
    never blocks the event loop, and it's closed when the browser disconnects.
    """
    raw = load_for_playback_cached(get_session(session_id))
    chans = [c for c in channels.split(",") if c] if channels else None
    return StreamingResponse(
        stream_frames(raw, chans, start=start, speed=speed, prefill=min(max(prefill, 0.0), 60.0),
                      preview=preview),
        media_type="text/event-stream",
        # no proxy buffering, or frames arrive in bursts instead of smoothly
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
