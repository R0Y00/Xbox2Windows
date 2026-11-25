use std::sync::OnceLock;

use super::{interleave, reference, Profile, BC7_ANCHOR2};

struct Field {
    channel: usize,
    endpoint: usize,
    shift: u32,
    bits: u32,
    reverse: bool,
}

struct Mode {
    code: u8,
    base: u8,
    delta: [u8; 3],
    transformed: bool,
    regions: u8,
    fields: Vec<Field>,
}

type ModeSpec = (u8, u8, [u8; 3], bool, u8, &'static str);

// The BC6H bit layouts after the mode bits, ordered by BCPack mode symbol.
const MODES: [ModeSpec; 14] = [
    (0, 10, [5, 5, 5], true, 2,
     "gy4 by4 bz4 rw10 gw10 bw10 rx5 gz4 gy4_ gx5 bz0 gz4_ bx5 bz1 by4_ ry5 bz2 rz5 bz3 d"),
    (1, 7, [6, 6, 6], true, 2,
     "gy5 gz4 gz5 rw7 bz0 bz1 by4 gw7 by5 bz2 gy4 bw7 bz3 bz5 bz4 rx6 gy4_ gx6 gz4_ bx6 by4_ ry6 rz6 d"),
    (2, 11, [5, 4, 4], true, 2,
     "rw10 gw10 bw10 rx5 rw:10 gy4_ gx4 gw:10 bz0 gz4_ bx4 bw:10 bz1 by4_ ry5 bz2 rz5 bz3 d"),
    (6, 11, [4, 5, 4], true, 2,
     "rw10 gw10 bw10 rx4 rw:10 gz4 gy4_ gx5 gw:10 gz4_ bx4 bw:10 bz1 by4_ ry4 bz0 bz2 rz4 gy4 bz3 d"),
    (10, 11, [4, 4, 5], true, 2,
     "rw10 gw10 bw10 rx4 rw:10 by4 gy4_ gx4 gw:10 bz0 gz4_ bx5 bw:10 by4_ ry4 bz1 bz2 rz4 bz4 bz3 d"),
    (14, 9, [5, 5, 5], true, 2,
     "rw9 by4 gw9 gy4 bw9 bz4 rx5 gz4 gy4_ gx5 bz0 gz4_ bx5 bz1 by4_ ry5 bz2 rz5 bz3 d"),
    (18, 8, [6, 5, 5], true, 2,
     "rw8 gz4 by4 gw8 bz2 gy4 bw8 bz3 bz4 rx6 gy4_ gx5 bz0 gz4_ bx5 bz1 by4_ ry6 rz6 d"),
    (22, 8, [5, 6, 5], true, 2,
     "rw8 bz0 by4 gw8 gy5 gy4 bw8 gz5 bz4 rx5 gz4 gy4_ gx6 gz4_ bx5 bz1 by4_ ry5 bz2 rz5 bz3 d"),
    (26, 8, [5, 5, 6], true, 2,
     "rw8 bz1 by4 gw8 by5 gy4 bw8 bz5 bz4 rx5 gz4 gy4_ gx5 bz0 gz4_ bx6 by4_ ry5 bz2 rz5 bz3 d"),
    (30, 6, [6, 6, 6], false, 2,
     "rw6 gz4 bz0 bz1 by4 gw6 gy5 by5 bz2 gy4 bw6 gz5 bz3 bz5 bz4 rx6 gy4_ gx6 gz4_ bx6 by4_ ry6 rz6 d"),
    (3, 10, [10, 10, 10], false, 1,
     "rw10 gw10 bw10 rx10 gx10 bx10"),
    (7, 11, [9, 9, 9], true, 1,
     "rw10 gw10 bw10 rx9 rw:10 gx9 gw:10 bx9 bw:10"),
    (11, 12, [8, 8, 8], true, 1,
     "rw10 gw10 bw10 rx8 rw:11-10 gx8 gw:11-10 bx8 bw:11-10"),
    (15, 16, [4, 4, 4], true, 1,
     "rw10 gw10 bw10 rx4 rw:15-10 gx4 gw:15-10 bx4 bw:15-10"),
];

fn modes() -> &'static [Mode; 14] {
    static MODES_CACHE: OnceLock<[Mode; 14]> = OnceLock::new();
    MODES_CACHE.get_or_init(|| {
        MODES.map(|(code, base, delta, transformed, regions, layout)| {
            let fields = layout
                .split_whitespace()
                .map(|token| {
                    if token == "d" {
                        return Field {
                            channel: 3,
                            endpoint: 0,
                            shift: 0,
                            bits: 5,
                            reverse: false,
                        };
                    }
                    let name = token.as_bytes();
                    let channel = b"rgb".iter().position(|&c| c == name[0]).unwrap();
                    let endpoint = b"wxyz".iter().position(|&c| c == name[1]).unwrap();
                    let spec = &token[2..];
                    let (shift, bits, reverse) = if let Some(span) = spec.strip_prefix(':') {
                        let (hi, lo) = span.split_once('-').unwrap_or((span, span));
                        let hi: u32 = hi.parse().unwrap();
                        let lo: u32 = lo.parse().unwrap();
                        (lo, hi - lo + 1, true)
                    } else if let Some(width) = spec.strip_suffix('_') {
                        (0, width.parse().unwrap(), false)
                    } else {
                        let n = spec.parse().unwrap();
                        if endpoint < 2 || channel == 0 {
                            (0, n, false)
                        } else {
                            (n, 1, false)
                        }
                    };
                    Field {
                        channel,
                        endpoint,
                        shift,
                        bits,
                        reverse,
                    }
                })
                .collect();
            Mode {
                code,
                base,
                delta,
                transformed,
                regions,
                fields,
            }
        })
    })
}

fn delta(value: u16) -> u16 {
    (value >> 1) ^ 0u16.wrapping_sub(value & 1)
}

pub(super) fn rebuild(symbols: &[u8], out: &mut [u8], profile: &Profile) -> Result<(), String> {
    let lane_size = out.len() / 8;
    let mut endpoints = [[[0u16; 4]; 3]; 256];
    let mut indices = [[0u8; 16]; 256];
    for (index, dst) in out.chunks_exact_mut(16).enumerate() {
        let sym = |lane, half| symbols[lane * lane_size + index * 2 + half];
        let joined = |lane, bits| interleave(sym(lane, 0), sym(lane, 1), bits);
        let mode_index = sym(15, 0) as usize;
        let mode = modes()
            .get(mode_index)
            .ok_or_else(|| format!("BC6H block {index} has an invalid mode"))?;
        let partition = sym(15, 1).min(31);
        let di = joined(1, 4) as u8;
        let de = di.wrapping_add(delta(joined(0, 4)) as u8);
        let base_end = reference(
            &endpoints,
            &profile.bc6h_endpoints,
            index,
            de.into(),
            [[0x8000, 0x200, 0x20, 0x20]; 3],
        )?;
        let base_ix = reference(&indices, &profile.bc6h_indices, index, di.into(), [8; 16])?;

        let mut fields = [[0u16; 4]; 3];
        for (channel, saved) in endpoints[index & 255].iter_mut().enumerate() {
            let width = mode.delta[channel];
            let mask = (1 << width) - 1;
            let shift = 6u8.saturating_sub(width);
            let base_mask = u16::MAX >> (16 - mode.base);
            let [r0, r1, r2, r3] = base_end[channel];
            let w =
                (r0 >> (16 - mode.base)).wrapping_add(delta(joined(2 + channel, 8))) & base_mask;
            let x = (r1 >> (10 - width)).wrapping_add(delta(joined(5 + channel, 5))) & mask;
            let x10 = x << (10 - width);
            let y = (r2 >> shift).wrapping_add(delta(sym(8 + channel, 0).into())) & mask;
            let z = (r3 >> shift).wrapping_add(delta(sym(8 + channel, 1).into())) & mask;
            let wire_w = match mode_index {
                12 => (w & 1023) | ((w >> 10).reverse_bits() >> 4),
                13 => (w & 1023) | ((w >> 10).reverse_bits()),
                _ => w,
            };
            fields[channel] = [wire_w, x, y, z];
            *saved = if mode.regions == 2 {
                [w << (16 - mode.base), x10, y << shift, z << shift]
            } else {
                let center = if mode.transformed {
                    1 << (width - 1)
                } else {
                    0
                };
                let low = ((w & mask) ^ center) << shift >> width.saturating_sub(6);
                [w << (16 - mode.base), x10, low, x10 >> 4]
            };
        }

        let mut data = u128::from(mode.code);
        let mut at = if mode.code < 2 { 2 } else { 5 };
        for field in &mode.fields {
            let value = if field.channel == 3 {
                partition.into()
            } else {
                fields[field.channel][field.endpoint]
            };
            let mut bits = (value >> field.shift) & (u16::MAX >> (16 - field.bits));
            if field.reverse {
                bits = bits.reverse_bits() >> (16 - field.bits);
            }
            data |= u128::from(bits) << at;
            at += field.bits;
        }
        let selector_width = if mode.regions == 2 { 3 } else { 4 };
        let anchor = if mode.regions == 2 {
            BC7_ANCHOR2[partition as usize]
        } else {
            0
        };
        for (pixel, predictor) in base_ix.into_iter().enumerate() {
            let row = pixel / 4;
            let wire = sym(11 + row, (pixel / 2) % 2);
            let parity = if row == 3 { pixel % 2 } else { 1 - pixel % 2 };
            let packed_delta = (0..4)
                .map(|bit| u16::from((wire >> (2 * bit + parity)) & 1) << bit)
                .sum();
            let ix = (predictor >> (4 - selector_width)).wrapping_add(delta(packed_delta) as u8)
                & ((1 << selector_width) - 1);
            indices[index & 255][pixel] = ix << (4 - selector_width);
            let bits = selector_width - usize::from(pixel == 0 || pixel == anchor as usize);
            if ix >= (1 << bits) {
                return Err(format!("BC6H block {index} has an invalid anchor selector"));
            }
            data |= u128::from(ix) << at;
            at += bits as u32;
        }
        debug_assert_eq!(at, 128);
        dst.copy_from_slice(&data.to_le_bytes());
    }
    Ok(())
}
