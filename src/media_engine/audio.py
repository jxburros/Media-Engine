"""16-bit PCM WAV files written and read with the standard library `wave` module.

A text-to-audio execution's `audio` output is one canonical WAV: a 44-byte RIFF header (`fmt `
chunk for PCM, 16 bits per sample) followed by one `data` chunk of interleaved little-endian
samples, nothing else. The writer takes samples already quantized to 16 bits; `pcm16` does that
for plain floats in [-1, 1] (clamped, rounded to nearest), the same rule the reference backend
applies to its tensors. The reader validates exactly that layout and returns the sample rate,
channel count and length, which the server reports in the output record; the worker never
resamples, trims or pads a clip to fit a limit.
"""

import struct
import wave
from pathlib import Path

MIME_TYPE = "audio/wav"
SAMPLE_WIDTH = 2  # bytes: 16-bit PCM
HEADER_BYTES = 44
FULL_SCALE = 32767


def pcm16(values):
    """Interleaved float samples in [-1, 1] → little-endian 16-bit PCM bytes (clamped, rounded)."""
    values = [max(-1.0, min(1.0, float(v))) for v in values]
    return struct.pack(f"<{len(values)}h", *(round(v * FULL_SCALE) for v in values))


def duration_seconds(samples, sample_rate):
    """Seconds of a clip of `samples` frames (samples per channel)."""
    return samples / sample_rate


def size(samples, channels):
    """The exact file size the writer produces for a clip, used to reserve storage in advance."""
    return HEADER_BYTES + samples * channels * SAMPLE_WIDTH


def write_wav(path, pcm, sample_rate, channels):
    """Write interleaved 16-bit PCM bytes as a canonical WAV; returns the facts written.

    A byte count that is not a whole number of frames, or an empty clip, is a programming error
    in the backend and fails the execution rather than producing a file that lies about itself.
    """
    frame_bytes = channels * SAMPLE_WIDTH
    if not pcm or len(pcm) % frame_bytes:
        raise ValueError("PCM data is not a whole, non-empty number of frames")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(SAMPLE_WIDTH)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm)
    samples = len(pcm) // frame_bytes
    return {"sample_rate": sample_rate, "channels": channels, "samples": samples}


def read_header(path):
    """Validate a canonical 16-bit PCM WAV and return its facts; raises ValueError otherwise.

    Checks the RIFF/WAVE/fmt/data layout, PCM encoding, 16-bit samples and that the file holds
    exactly the frames its header declares, without reading the samples. The server reads output
    facts back from the file through this, as it reads image sizes back from PNG headers.
    """
    path = Path(path)
    with path.open("rb") as stream:
        head = stream.read(HEADER_BYTES)
    if len(head) != HEADER_BYTES:
        raise ValueError("not a WAV file")
    riff, riff_size, wave_id, fmt_id, fmt_size = struct.unpack("<4sI4s4sI", head[:20])
    encoding, channels, rate, byte_rate, align, bits = struct.unpack("<HHIIHH", head[20:36])
    data_id, data_size = struct.unpack("<4sI", head[36:44])
    if (riff, wave_id, fmt_id, data_id) != (b"RIFF", b"WAVE", b"fmt ", b"data") or fmt_size != 16:
        raise ValueError("not a canonical WAV file")
    if encoding != 1 or bits != 8 * SAMPLE_WIDTH or channels < 1 or rate < 1:
        raise ValueError("WAV is not 16-bit PCM")
    if align != channels * SAMPLE_WIDTH or byte_rate != rate * align:
        raise ValueError("WAV header is inconsistent")
    actual = path.stat().st_size
    if data_size == 0 or data_size % align or riff_size != actual - 8 or actual != HEADER_BYTES + data_size:
        raise ValueError("WAV length does not match its header")
    try:
        with wave.open(str(path), "rb") as stream:
            frames = stream.getnframes()
            if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) != (channels, SAMPLE_WIDTH, rate):
                raise ValueError("WAV header is inconsistent")
    except (wave.Error, EOFError) as exc:
        raise ValueError("not a WAV file") from exc
    return {
        "sample_rate": rate,
        "channels": channels,
        "samples": frames,
        "duration_seconds": duration_seconds(frames, rate),
    }


def read_wav(path):
    """Return (facts, interleaved 16-bit PCM bytes); used by tests and callers' tools."""
    facts = read_header(path)
    with wave.open(str(Path(path)), "rb") as stream:
        return facts, stream.readframes(facts["samples"])
