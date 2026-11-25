//! Strict mode-6 batches that need no external predictor history.
use crate::{entropy, rebuild, spread, zig, Cell};
use std::slice;

const WIDTHS: [u8; 16] = [4, 4, 8, 8, 8, 8, 8, 7, 7, 5, 6, 8, 8, 8, 8, 5];

// The caller supplies the confirmed signature and count * 16 writable bytes.
#[no_mangle]
pub unsafe extern "C" fn decode_strict_mode6(
    input: *const u8,
    length: usize,
    count: usize,
    first: u8,
    second: u8,
    output: *mut u8,
) -> bool {
    if count == 0
        || count > isize::MAX as usize / 32
        || length > isize::MAX as usize
        || first >= 32
        || second >= 16
    {
        return false;
    }
    let stride = count * 2;
    let mut lanes = vec![0; stride * WIDTHS.len()];
    if entropy::unpack(
        slice::from_raw_parts(input, length),
        &WIDTHS,
        stride,
        &mut lanes,
    )
    .is_none()
    {
        return false;
    }
    let mut symbols = vec![0; stride * WIDTHS.len()];
    for (index, block) in symbols.chunks_exact_mut(32).enumerate() {
        for (lane, values) in lanes.chunks_exact(stride).enumerate() {
            block[lane] = values[2 * index];
            block[16 + lane] = values[2 * index + 1];
        }
        if block[15] != first || block[31] != second {
            return false;
        }
        let di = spread(block[1]) + 2 * spread(block[17]);
        let delta = zig((spread(block[0]) + 2 * spread(block[16])) as u8);
        let de = ((di as i32 + delta) & 255) as usize;
        if di > index || de > index {
            return false;
        }
    }

    let mut labels = [-1; 4096];
    let label = 4 * (32 * usize::from(first) + usize::from(second));
    labels[label..label + 4].copy_from_slice(&[6, 0, 0, 0]);
    let seeds = [[Cell::default(); 56]; 256];
    let anchors = [0; 128];
    let mut ready = vec![0; count];
    rebuild(
        symbols.as_ptr(),
        count,
        labels.as_ptr(),
        seeds.as_ptr().cast(),
        anchors.as_ptr(),
        output,
        ready.as_mut_ptr(),
    );
    ready.into_iter().all(|block| block == 1)
}
