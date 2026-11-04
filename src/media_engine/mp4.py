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
                frame = av.VideoFrame(w, h, format="rgb24")
                # Fill PyAV's RGB plane directly; the encoder extra does not depend on numpy.
                plane = frame.planes[0]
                stride = w * 3
                padding = bytes(plane.line_size - stride)
                plane.update(b"".join(pixels[row*stride:(row+1)*stride] + padding for row in range(h)))
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
        rate = stream.average_rate
        if rate is None or rate.denominator != 1:
            raise ValueError("invalid frame rate")
        fps = int(rate)
        count = 0
        for frame in container.decode(stream):
            if frame.pts is None or frame.pts * frame.time_base != Fraction(count, fps):
                raise ValueError("video is not constant frame rate")
            if (frame.width, frame.height) != (stream.width, stream.height):
                raise ValueError("inconsistent frame dimensions")
            count += 1
        if count == 0:
            raise ValueError("empty video")
        return {"width": stream.width, "height": stream.height, "frames": count, "fps": fps,
                "duration_seconds": count / fps}
