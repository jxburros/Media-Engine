//! PNG input and output with the semantics of `media_engine.png` (8-bit greyscale/RGB(A) in,
//! alpha dropped and greyscale replicated; 8-bit RGB out).

use crate::Error;

/// Decode to (width, height, RGB bytes). Only 8-bit greyscale, grey+alpha, RGB and RGBA inputs
/// are accepted (interlaced or not), as the Python reader.
pub fn decode_rgb(data: &[u8]) -> Result<(usize, usize, Vec<u8>), Error> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(data));
    decoder.set_transformations(png::Transformations::IDENTITY);
    let mut reader = decoder.read_info().map_err(|_| Error::InvalidInput("input image is not a readable PNG"))?;
    let (color, depth) = {
        let info = reader.info();
        (info.color_type, info.bit_depth)
    };
    if depth != png::BitDepth::Eight {
        return Err(Error::InvalidInput("unsupported PNG"));
    }
    let channels = match color {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => return Err(Error::InvalidInput("unsupported PNG")),
    };
    let size = reader.output_buffer_size().ok_or(Error::InvalidInput("input image is too large"))?;
    let mut buffer = vec![0u8; size];
    let frame = reader.next_frame(&mut buffer).map_err(|_| Error::InvalidInput("input image is not a readable PNG"))?;
    let (width, height) = (frame.width as usize, frame.height as usize);
    let stride = frame.line_size;
    let mut rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        let row = &buffer[y * stride..y * stride + width * channels];
        for x in 0..width {
            let px = &row[x * channels..(x + 1) * channels];
            if channels >= 3 {
                rgb.extend_from_slice(&px[..3]);
            } else {
                rgb.extend_from_slice(&[px[0], px[0], px[0]]);
            }
        }
    }
    Ok((width, height, rgb))
}

pub fn encode_rgb(width: usize, height: usize, rgb: &[u8]) -> Result<Vec<u8>, Error> {
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width as u32, height as u32);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().map_err(|e| Error::Failed(format!("png: {e}")))?;
        writer.write_image_data(rgb).map_err(|e| Error::Failed(format!("png: {e}")))?;
        writer.finish().map_err(|e| Error::Failed(format!("png: {e}")))?;
    }
    Ok(out)
}

/// `(pixels.permute(1, 2, 0).float() * 255).round().clamp(0, 255).to(uint8)` for a (3, H, W)
/// float32 buffer already in [0, 1]; torch rounds half to even.
pub fn to_bytes(chw: &[f32], width: usize, height: usize) -> Vec<u8> {
    let plane = width * height;
    let mut out = Vec::with_capacity(plane * 3);
    for i in 0..plane {
        for c in 0..3 {
            let v = (chw[c * plane + i] * 255.0f32).round_ties_even().clamp(0.0, 255.0);
            out.push(v as u8);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_rounds_half_even() {
        let png = encode_rgb(2, 1, &[1, 2, 3, 250, 251, 252]).unwrap();
        assert_eq!(decode_rgb(&png).unwrap(), (2, 1, vec![1, 2, 3, 250, 251, 252]));
        assert_eq!(to_bytes(&[0.0, 1.0, 2.0], 1, 1), vec![0, 255, 255]);
    }
}
