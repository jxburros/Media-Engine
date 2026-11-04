"""Optional pinned PyAV H.264 encoder. Never invokes an executable from PATH."""
import importlib.util
from fractions import Fraction
from pathlib import Path

MIME_TYPE = "video/mp4"


def available():
    if importlib.util.find_spec("av") is None:
        return False
    try:
        import av
        av.codec.Codec("libx264", "w")
        return True
    except (ImportError, ValueError):
        return False


def write(path, pngs, fps, cancelled):
    import av
    from . import png
    from .backend import Cancelled
    import numpy as np

    path = Path(path)
    try:
        with av.open(str(path), "w", format="mp4") as container:
            stream = container.add_stream("libx264", rate=fps)
            w, h, _ = png.read_rgb(pngs[0])
            stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
            stream.options = {"crf": "18", "preset": "medium", "threads": "1"}
            for index, encoded in enumerate(pngs):
                if cancelled.is_set():
                    raise Cancelled()
                width, height, pixels = png.read_rgb(encoded)
                if (width, height) != (w, h):
                    raise ValueError("inconsistent frame dimensions")
                frame = av.VideoFrame.from_ndarray(np.frombuffer(pixels, dtype=np.uint8).reshape(h, w, 3), format="rgb24")
                frame.pts = index
                frame.time_base = Fraction(1, fps)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        if cancelled.is_set():
            raise Cancelled()
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def read_metadata(path):
    import av
    with av.open(str(path), "r") as container:
        if len(container.streams.video) != 1 or container.streams.audio:
            raise ValueError("expected one video stream")
        stream = container.streams.video[0]
        if stream.codec_context.name != "h264" or stream.pix_fmt != "yuv420p":
            raise ValueError("unsupported video encoding")
        frames = list(container.decode(stream))
        rate = stream.average_rate
        if not frames or rate is None or rate.denominator != 1:
            raise ValueError("invalid frame rate")
        fps = int(rate)
        times = [frame.pts * frame.time_base for frame in frames]
        if any(t != Fraction(i, fps) for i, t in enumerate(times)):
            raise ValueError("video is not constant frame rate")
        return {"width": stream.width, "height": stream.height, "frames": len(frames), "fps": fps,
                "duration_seconds": len(frames) / fps}
