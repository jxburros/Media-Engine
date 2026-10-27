"""Standard-library PNG: the signature/IHDR reader, an 8-bit decoder and an 8-bit RGB writer.

The owned pipeline and the upscale path read inputs and write outputs here, so neither needs an
image library. The writer uses filter type 0 on every row and one IDAT chunk; it is lossless, so
the decoded pixels equal the tensor's bytes exactly (the file bytes differ from Pillow's, which
chooses filters adaptively).
"""

import struct
import zlib

SIGNATURE = b"\x89PNG\r\n\x1a\n"
CHANNELS = {0: 1, 2: 3, 4: 2, 6: 4}


def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def header(data):
    """Return (width, height, depth, color, interlace) from the signature and IHDR, or raise ValueError."""
    if len(data) < 33 or data[:8] != SIGNATURE or data[12:16] != b"IHDR":
        raise ValueError("not a PNG")
    width, height, depth, color, _compression, _filter, interlace = struct.unpack(
        ">IIBBBBB", data[16:29]
    )
    return width, height, depth, color, interlace


def write(width, height, pixels, level=6):
    """Encode 8-bit RGB bytes (row-major, width × height × 3) as a PNG."""
    stride = width * 3
    if len(pixels) != stride * height:
        raise ValueError("pixel buffer does not match the dimensions")
    raw = b"".join(b"\0" + pixels[y * stride : (y + 1) * stride] for y in range(height))
    return (
        SIGNATURE
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level))
        + chunk(b"IEND", b"")
    )


def _unfilter(raw, width, height, bpp):
    stride = width * bpp
    out = bytearray(stride * height)
    previous = bytearray(stride)
    position = 0
    for y in range(height):
        kind = raw[position]
        line = bytearray(raw[position + 1 : position + 1 + stride])
        position += 1 + stride
        if kind == 1:
            for x in range(bpp, stride):
                line[x] = (line[x] + line[x - bpp]) & 255
        elif kind == 2:
            for x in range(stride):
                line[x] = (line[x] + previous[x]) & 255
        elif kind == 3:
            for x in range(stride):
                left = line[x - bpp] if x >= bpp else 0
                line[x] = (line[x] + ((left + previous[x]) >> 1)) & 255
        elif kind == 4:
            for x in range(stride):
                a = line[x - bpp] if x >= bpp else 0
                b = previous[x]
                c = previous[x - bpp] if x >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[x] = (line[x] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        elif kind != 0:
            raise ValueError("invalid PNG filter")
        out[y * stride : (y + 1) * stride] = line
        previous = line
    return bytes(out), position


ADAM7 = ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2))


def read_rgb(data):
    """Decode an 8-bit greyscale/RGB(A) PNG (any filter, Adam7 or not) to (width, height, RGB bytes).

    Alpha is dropped and greyscale is replicated, matching an RGB conversion of the same file.
    """
    width, height, depth, color, interlace = header(data)
    if depth != 8 or color not in CHANNELS or interlace not in (0, 1):
        raise ValueError("unsupported PNG")
    position, idat = 8, []
    while position + 8 <= len(data):
        (length,) = struct.unpack(">I", data[position : position + 4])
        kind = data[position + 4 : position + 8]
        if kind == b"IDAT":
            idat.append(data[position + 8 : position + 8 + length])
        if kind == b"IEND":
            break
        position += 12 + length
    raw = zlib.decompress(b"".join(idat))
    bpp = CHANNELS[color]
    if interlace == 0:
        pixels, _ = _unfilter(raw, width, height, bpp)
    else:
        image = bytearray(width * height * bpp)
        offset = 0
        for x0, y0, dx, dy in ADAM7:
            w = (width - x0 + dx - 1) // dx if width > x0 else 0
            h = (height - y0 + dy - 1) // dy if height > y0 else 0
            if not w or not h:
                continue
            sub, used = _unfilter(raw[offset:], w, h, bpp)
            offset += used
            for y in range(h):
                for x in range(w):
                    source = (y * w + x) * bpp
                    target = ((y0 + y * dy) * width + x0 + x * dx) * bpp
                    image[target : target + bpp] = sub[source : source + bpp]
        pixels = bytes(image)
    if color == 2:
        return width, height, pixels
    rgb = bytearray(width * height * 3)
    for index in range(width * height):
        source = pixels[index * bpp : index * bpp + bpp]
        rgb[index * 3 : index * 3 + 3] = source[:3] if color == 6 else bytes((source[0],)) * 3
    return width, height, bytes(rgb)


def tensor_to_png(image):
    """Write a CPU-or-device float tensor (3, H, W) in [0, 1] as PNG without numpy or Pillow."""
    import ctypes

    import torch

    pixels = (image.permute(1, 2, 0).float() * 255).round().clamp(0, 255).to(torch.uint8)
    pixels = pixels.cpu().contiguous()
    height, width = pixels.shape[:2]
    return write(width, height, ctypes.string_at(pixels.data_ptr(), pixels.numel()))


def png_to_tensor(data):
    """Decode a PNG to a float32 tensor (3, H, W) in [0, 1]."""
    import torch

    width, height, pixels = read_rgb(data)
    tensor = torch.frombuffer(bytearray(pixels), dtype=torch.uint8)
    return tensor.view(height, width, 3).permute(2, 0, 1).float() / 255
