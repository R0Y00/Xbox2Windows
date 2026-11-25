//! BC1 block reconstruction with predictor and selector tables supplied by the caller.
use crate::{entropy, zig};
use std::{fmt, slice};

const WIDTHS: [u8; 12] = [8, 8, 5, 6, 5, 5, 6, 5, 8, 8, 8, 8];

pub(crate) enum DecodeError {
    ReservedLane { block: usize },
    MissingPredictor { block: usize, slot: usize },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::ReservedLane { block } => {
                write!(f, "BC1 block {block} has a nonzero reserved lane")
            }
            Self::MissingPredictor { block, slot } => {
                write!(
                    f,
                    "BC1 block {block} requires missing predictor slot {slot}"
                )
            }
        }
    }
}

pub(crate) fn blocks(
    count: usize,
    symbol: impl Fn(usize, usize) -> u8,
    endpoints: &[[i32; 6]],
    selectors: &[[u32; 256]],
    output: &mut [u8],
    strict: bool,
) -> Result<(), DecodeError> {
    let mut history = [[0u8; 6]; 256];
    for (i, block) in output.chunks_exact_mut(8).take(count).enumerate() {
        if strict && symbol(i, 1) != 0 {
            return Err(DecodeError::ReservedLane { block: i });
        }
        let distance = zig(symbol(i, 0)) as u8 as usize;
        let mut color = if distance == 0 {
            [0; 6]
        } else if distance <= i {
            history[(i - distance) & 255]
        } else {
            let age = distance - i;
            // The strict table uses ring slots. Calibrated profiles index by age.
            let slot = if strict { 256 - age } else { age };
            let seed = endpoints[slot];
            if seed.iter().any(|&value| value < 0) {
                return Err(DecodeError::MissingPredictor { block: i, slot });
            }
            seed.map(|value| value as u8)
        };
        for (component, base) in color.iter_mut().enumerate() {
            let mask = if component % 3 == 1 { 63 } else { 31 };
            *base = base.wrapping_add(zig(symbol(i, 2 + component)) as u8) & mask;
        }
        history[i & 255] = color;
        let rgb = |start: usize| {
            (u16::from(color[start]) << 11)
                | (u16::from(color[start + 1]) << 5)
                | u16::from(color[start + 2])
        };
        let indices = (0..4).fold(0, |bits, quadrant| {
            bits | selectors[quadrant][usize::from(symbol(i, 8 + quadrant))]
        });
        block[..2].copy_from_slice(&rgb(0).to_le_bytes());
        block[2..4].copy_from_slice(&rgb(3).to_le_bytes());
        block[4..].copy_from_slice(&indices.to_le_bytes());
    }
    Ok(())
}

// Caller supplies 256 endpoint rows, four selector rows and count * 8 writable bytes.
#[no_mangle]
pub unsafe extern "C" fn decode_strict_bc1(
    input: *const u8,
    length: usize,
    count: usize,
    endpoints: *const [i32; 6],
    selectors: *const [u32; 256],
    output: *mut u8,
) -> bool {
    if count == 0 || count > isize::MAX as usize / WIDTHS.len() || length > isize::MAX as usize {
        return false;
    }
    let mut lanes = vec![0; count * WIDTHS.len()];
    if entropy::unpack(
        slice::from_raw_parts(input, length),
        &WIDTHS,
        count,
        &mut lanes,
    )
    .is_none()
    {
        return false;
    }
    blocks(
        count,
        |i, lane| lanes[lane * count + i],
        slice::from_raw_parts(endpoints, 256),
        slice::from_raw_parts(selectors, 4),
        slice::from_raw_parts_mut(output, count * 8),
        true,
    )
    .is_ok()
}

// The calibrated reader supplies count interleaved records of twelve symbols each.
#[no_mangle]
pub unsafe extern "C" fn rebuild_calibrated_bc1(
    input: *const u8,
    count: usize,
    endpoints: *const [i32; 6],
    selectors: *const [u32; 256],
    output: *mut u8,
) -> bool {
    if count > isize::MAX as usize / WIDTHS.len() {
        return false;
    }
    let symbols = slice::from_raw_parts(input, count * WIDTHS.len());
    blocks(
        count,
        |i, lane| symbols[i * WIDTHS.len() + lane],
        slice::from_raw_parts(endpoints, 256),
        slice::from_raw_parts(selectors, 4),
        slice::from_raw_parts_mut(output, count * 8),
        false,
    )
    .is_ok()
}
