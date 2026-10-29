"""Frame archives: lossless PNG frames in a stored ZIP plus `frames.json` (standard library only).

An image-to-video execution's `frames` output is one archive holding `000001.png`, `000002.png` …
(one 8-bit RGB PNG per frame, numbered from one) and `frames.json` with `fps`, `width`, `height`
and `frames`. Members are stored, not deflated again (PNG is already compressed), and carry a
fixed timestamp and mode, so the same frames always give the same archive bytes. The poster is
frame 1's PNG, byte for byte. A caller can encode any container later from the archive without
re-running inference; the worker never re-times or drops frames.
"""

import json
import zipfile

from . import png

MANIFEST = "frames.json"
MANIFEST_FIELDS = ("fps", "width", "height", "frames")
MIME_TYPE = "application/zip"
# The DOS epoch: ZIP cannot store an earlier time, and a fixed one keeps archives reproducible.
TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def frame_name(index):
    """The archive member for frame `index` (1-based): 000001.png …"""
    return f"{index:06d}.png"


def duration_seconds(frames, fps):
    return frames / fps


def member(name):
    info = zipfile.ZipInfo(name, TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3  # Unix, so the stored mode below means the same on every host
    info.external_attr = 0o644 << 16
    return info


def write_archive(path, frames, fps, width, height):
    """Write PNG frame bytes and frames.json to `path`; returns the manifest written.

    Every frame must be a PNG of exactly width × height; a mismatch is a programming error in
    the backend and fails the execution rather than producing an archive that lies about itself.
    """
    frames = list(frames)
    if not frames:
        raise ValueError("an archive needs at least one frame")
    for data in frames:
        if png.header(data)[:2] != (width, height):
            raise ValueError("frame size does not match the archive")
    manifest = {"fps": fps, "width": width, "height": height, "frames": len(frames)}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        # The manifest comes first so a streaming reader learns the shape before the frames.
        archive.writestr(member(MANIFEST), json.dumps(manifest, sort_keys=True).encode())
        for index, data in enumerate(frames, 1):
            archive.writestr(member(frame_name(index)), data)
    return manifest


def read_manifest(path):
    """Validate an archive's layout and return its manifest; raises ValueError if it is not one.

    Checks the member names, the stored compression, the manifest's fields and that each frame's
    PNG header matches it, without decoding any pixels. The server reads output facts back from
    the file through this, as it reads image sizes back from PNG headers.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [i.filename for i in infos]
            if any(i.compress_type != zipfile.ZIP_STORED for i in infos):
                raise ValueError("archive members must be stored")
            if MANIFEST not in names:
                raise ValueError("archive has no frames.json")
            manifest = json.loads(archive.read(MANIFEST))
            if not isinstance(manifest, dict) or set(manifest) != set(MANIFEST_FIELDS):
                raise ValueError("frames.json has the wrong fields")
            if any(type(manifest[k]) is not int or manifest[k] < 1 for k in MANIFEST_FIELDS):
                raise ValueError("frames.json values must be positive integers")
            expected = [frame_name(k) for k in range(1, manifest["frames"] + 1)]
            if sorted(n for n in names if n != MANIFEST) != expected or len(names) != len(expected) + 1:
                raise ValueError("archive members do not match frames.json")
            size = (manifest["width"], manifest["height"])
            for name in expected:
                with archive.open(name) as stream:
                    if png.header(stream.read(33))[:2] != size:
                        raise ValueError("frame size does not match frames.json")
    except (zipfile.BadZipFile, KeyError, UnicodeDecodeError) as exc:
        raise ValueError("not a frame archive") from exc
    return manifest


def read_archive(path):
    """Return (manifest, [PNG bytes per frame, in order]); used by tests and callers' tools."""
    manifest = read_manifest(path)
    with zipfile.ZipFile(path) as archive:
        frames = [archive.read(frame_name(k)) for k in range(1, manifest["frames"] + 1)]
    return manifest, frames
