//! Standalone block decoding with caller-supplied predictor state.
mod bc6h;
mod tables;

use std::collections::BTreeMap;

use crate::{bc1, entropy, rebuild as rebuild_interleaved, spread, zig};
use tables::*;

pub const MAX_CHUNK: usize = 16 << 20;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Codec {
    Bc1,
    Bc3,
    Bc4,
    Bc5,
    Bc6h,
    Bc7,
}

impl Codec {
    pub fn block_bytes(self) -> usize {
        match self {
            Self::Bc1 | Self::Bc4 => 8,
            _ => 16,
        }
    }

    pub fn widths(self) -> &'static [u8] {
        match self {
            Self::Bc1 => &[8, 8, 5, 6, 5, 5, 6, 5, 8, 8, 8, 8],
            Self::Bc3 => &[4, 4, 8, 6, 6, 6, 6, 5, 6, 5, 8, 8],
            Self::Bc4 => &[8, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6],
            Self::Bc5 => &[4, 4, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6],
            Self::Bc6h => &[4, 4, 8, 8, 8, 5, 5, 5, 6, 6, 6, 8, 8, 8, 8, 5],
            Self::Bc7 => &[4, 4, 8, 8, 8, 8, 8, 7, 7, 5, 6, 8, 8, 8, 8, 5],
        }
    }
}

pub type Seeds<T> = BTreeMap<u8, T>;

#[derive(Default)]
pub struct Profile {
    pub bc1: Seeds<[u8; 6]>,
    pub bc3: Seeds<[u8; 8]>,
    pub bc4: Seeds<[u8; 2]>,
    pub bc5: Seeds<[u8; 4]>,
    pub bc6h_endpoints: Seeds<[[u16; 4]; 3]>,
    pub bc6h_indices: Seeds<[u8; 16]>,
    pub bc7: Bc7Profile,
}

pub struct Bc7Profile {
    labels: [i16; 4096],
    seeds: [crate::History; 256],
    anchors: [u16; 128],
}

impl Default for Bc7Profile {
    fn default() -> Self {
        let mut anchors = [0; 128];
        for i in 0..64 {
            anchors[i] = 1 | (1 << BC7_ANCHOR2[i]);
            anchors[i + 64] = 1 | (1 << BC7_ANCHOR3A[i]) | (1 << BC7_ANCHOR3B[i]);
        }
        Self {
            labels: [-1; 4096],
            seeds: [[crate::Cell::default(); 56]; 256],
            anchors,
        }
    }
}

impl Bc7Profile {
    pub fn add_signature(
        &mut self,
        [a, b, mode, partition, rotation, selection]: [u8; 6],
    ) -> Result<(), String> {
        let valid_partition = match mode {
            0 => partition < 16,
            1 | 2 | 3 | 7 => partition < 64,
            _ => partition == 0,
        };
        if a >= 32
            || b >= 16
            || mode > 7
            || !valid_partition
            || rotation > 3
            || (rotation != 0 && !matches!(mode, 4 | 5))
            || selection > u8::from(mode == 4)
        {
            return Err("invalid BC7 signature".into());
        }
        let at = 4 * (32 * usize::from(a) + usize::from(b));
        if self.labels[at] >= 0 {
            return Err("duplicate BC7 signature".into());
        }
        self.labels[at..at + 4]
            .copy_from_slice(&[mode, partition, rotation, selection].map(i16::from));
        Ok(())
    }

    pub fn set_endpoints(&mut self, slot: u8, value: [u8; 24]) {
        let seed = &mut self.seeds[256usize.wrapping_sub(usize::from(slot)) & 255][..24];
        for (cell, value) in seed.iter_mut().zip(value) {
            *cell = crate::Cell {
                value,
                precision: 8,
            };
        }
    }

    pub fn set_indices(&mut self, slot: u8, value: [u8; 32]) -> Result<(), String> {
        if value.iter().any(|&v| v >= 16) {
            return Err("BC7 index seeds must be nibbles".into());
        }
        let seed = &mut self.seeds[256usize.wrapping_sub(usize::from(slot)) & 255][24..];
        // The wire order is primary then secondary. The block rebuilder stores secondary then primary.
        for (cell, value) in seed.iter_mut().zip(value[16..].iter().chain(&value[..16])) {
            *cell = crate::Cell {
                value: value << 4,
                precision: 8,
            };
        }
        Ok(())
    }
}

pub fn decode(
    input: &[u8],
    size: usize,
    codec: Codec,
    profile: &Profile,
) -> Result<Vec<u8>, String> {
    rebuild(&decode_symbols(input, size, codec)?, size, codec, profile)
}

fn decode_symbols(input: &[u8], size: usize, codec: Codec) -> Result<Vec<u8>, String> {
    if size == 0 || size > MAX_CHUNK || size % codec.block_bytes() != 0 {
        return Err(format!("invalid {:?} chunk length {size}", codec));
    }
    let widths = codec.widths();
    let mut symbols = vec![0; (size / 8) * widths.len()];
    entropy::unpack(input, widths, size / 8, &mut symbols)
        .ok_or("invalid BCPack entropy stream")?;
    Ok(symbols)
}

pub fn rebuild(
    symbols: &[u8],
    size: usize,
    codec: Codec,
    profile: &Profile,
) -> Result<Vec<u8>, String> {
    let count = size / codec.block_bytes();
    if size == 0
        || size > MAX_CHUNK
        || size % codec.block_bytes() != 0
        || symbols.len() != (size / 8) * codec.widths().len()
        || symbols
            .chunks_exact(size / 8)
            .zip(codec.widths())
            .any(|(lane, width)| *width < 8 && lane.iter().any(|&s| s >= (1 << width)))
    {
        return Err("invalid BCPack lane dimensions or symbols".into());
    }
    let mut out = vec![0; size];
    match codec {
        Codec::Bc1 => {
            let mut seeds = [[-1; 6]; 256];
            for (&slot, value) in &profile.bc1 {
                seeds[usize::from(slot)] = value.map(i32::from);
            }
            let selectors = [(0, 8), (4, 12), (16, 24), (20, 28)].map(|(low, high)| {
                BC1_QUADRANT.map(|q| (u32::from(q) & 15) << low | (u32::from(q) >> 4) << high)
            });
            bc1::blocks(
                count,
                |index, lane| symbols[lane * count + index],
                &seeds,
                &selectors,
                &mut out,
                true,
            )
            .map_err(|error| error.to_string())?;
        }
        Codec::Bc6h => bc6h::rebuild(symbols, &mut out, profile)?,
        Codec::Bc7 => {
            let p = &profile.bc7;
            let mut status = vec![0; count];
            let interleaved = interleave_lanes(symbols, size / 8, codec.widths().len());
            unsafe {
                rebuild_interleaved(
                    interleaved.as_ptr(),
                    count,
                    p.labels.as_ptr(),
                    p.seeds.as_ptr().cast(),
                    p.anchors.as_ptr(),
                    out.as_mut_ptr(),
                    status.as_mut_ptr(),
                )
            };
            if let Some(block) = status.iter().position(|&ready| ready == 0) {
                return Err(format!(
                    "BC7 block {block} has unresolved predictors or invalid selectors"
                ));
            }
        }
        _ => rebuild_legacy(symbols, &mut out, codec, profile, true)?,
    }
    Ok(out)
}

fn interleave_lanes(lanes: &[u8], count: usize, width: usize) -> Vec<u8> {
    let mut symbols = vec![0; count * width];
    for (lane, values) in lanes.chunks_exact(count).enumerate() {
        for (row, value) in symbols.chunks_exact_mut(width).zip(values) {
            row[lane] = *value;
        }
    }
    symbols
}

pub fn decode_fitted_bc1(
    input: &[u8],
    size: usize,
    endpoints: &[[i32; 6]; 256],
    selectors: &[[u32; 256]; 4],
    rotation: &[usize],
) -> Result<Vec<u8>, String> {
    if size == 0 || size > MAX_CHUNK || size % 8 != 0 {
        return Err("invalid fitted BC1 chunk length".into());
    }
    let count = size / 8;
    let width = Codec::Bc1.widths().len();
    let mut lanes = vec![0; count * width];
    entropy::unpack(input, Codec::Bc1.widths(), count, &mut lanes)
        .ok_or("invalid BCPack entropy stream")?;
    let mut out = vec![0; size];
    let success = if rotation.is_empty() {
        bc1::blocks(
            count,
            |i, lane| lanes[lane * count + i],
            endpoints,
            selectors,
            &mut out,
            false,
        )
    } else {
        let mut symbols = interleave_lanes(&lanes, count, width);
        let mut seen = std::collections::HashSet::new();
        if !rotation
            .iter()
            .all(|&i| i < symbols.len() && seen.insert(i))
        {
            return Err("Invalid symbol rotation".into());
        }
        let first = symbols[rotation[0]];
        for pair in rotation.windows(2) {
            symbols[pair[0]] = symbols[pair[1]];
        }
        symbols[*rotation.last().unwrap()] = first;
        bc1::blocks(
            count,
            |i, lane| symbols[i * width + lane],
            endpoints,
            selectors,
            &mut out,
            false,
        )
    };
    success
        .map(|()| out)
        .map_err(|_| "Missing BC1 endpoint history".into())
}

/// Symbols and reconstruction state for a fitted BC7 chunk.
pub struct FittedBc7 {
    pub symbols: Vec<u8>,
    pub blocks: Vec<u8>,
    pub ready: Vec<u8>,
}

impl FittedBc7 {
    pub fn unpack(input: &[u8], size: usize) -> Result<Self, String> {
        let lanes = decode_symbols(input, size, Codec::Bc7)?;
        Ok(Self {
            symbols: interleave_lanes(&lanes, size / 8, Codec::Bc7.widths().len()),
            blocks: vec![0; size],
            ready: vec![0; size / 16],
        })
    }

    pub fn apply(
        &mut self,
        labels: &[[i16; 4]; 1024],
        anchors: &[u16; 128],
        seed: &[[u8; 2]],
    ) -> Result<(), String> {
        if seed.len() != 256 * 56 || seed.iter().any(|pair| pair[1] > 8) {
            return Err("invalid fitted BC7 seed table".into());
        }
        let count = self.ready.len();
        let mut alternative = vec![0; self.blocks.len()];
        let mut known = vec![0; count];
        // The seed values and precisions are the two bytes of a repr(C) Cell.
        unsafe {
            rebuild_interleaved(
                self.symbols.as_ptr(),
                count,
                labels.as_ptr().cast(),
                seed.as_ptr().cast(),
                anchors.as_ptr(),
                alternative.as_mut_ptr(),
                known.as_mut_ptr(),
            );
        }
        for index in 0..count {
            if self.ready[index] == 0 && known[index] == 1 {
                let bytes = index * 16..(index + 1) * 16;
                self.blocks[bytes.clone()].copy_from_slice(&alternative[bytes]);
                self.ready[index] = 1;
            }
        }
        Ok(())
    }
}

pub fn decode_fitted_bc4(
    input: &[u8],
    size: usize,
    seeds: &Seeds<[u8; 2]>,
) -> Result<Vec<u8>, String> {
    let lanes = decode_symbols(input, size, Codec::Bc4)?;
    let mut out = vec![0; size];
    let mut history = [[0; 2]; 256];
    for (index, block) in out.chunks_exact_mut(8).enumerate() {
        let lane = |i| lanes[i * (size / 8) + index];
        let distance = usize::from(zig(lane(0)) as u8);
        let base = reference(&history, seeds, index, distance, [128; 2])?;
        let end = [0, 1].map(|i| base[i].wrapping_add(zig(lane(2 + i)) as u8));
        history[index & 255] = end;
        write_alpha(block, end, |row, half| {
            BC4_PAIR[lane(4 + row * 2 + half) as usize]
        });
    }
    Ok(out)
}

pub fn decode_fitted_bc3(
    input: &[u8],
    size: usize,
    endpoints: &BTreeMap<i32, [u8; 8]>,
    color: &[u8; 256],
    alpha: &[u8; 64],
) -> Result<Vec<u8>, String> {
    if alpha.iter().any(|&rank| rank >= 64) {
        return Err("invalid fitted BC3 selector rank".into());
    }
    let lanes = decode_symbols(input, size, Codec::Bc3)?;
    let mut out = vec![0; size];
    let stride = size / 8;
    let mut history = [[0; 8]; 256];
    for (index, block) in out.chunks_exact_mut(16).enumerate() {
        let pair = |lane, half| lanes[lane * stride + index * 2 + half];
        let distance = usize::from(zig((spread(pair(0, 0)) + 2 * spread(pair(0, 1))) as u8) as u8);
        let mut values = if distance == 0 || distance > index {
            let root = if distance == 0 {
                -512
            } else {
                index as i32 - distance as i32
            };
            *endpoints
                .get(&root)
                .ok_or_else(|| format!("missing fitted BC3 predictor {root}"))?
        } else {
            history[(index - distance) & 255]
        };
        for (i, value) in values.iter_mut().enumerate() {
            *value = value.wrapping_add(zig(pair([7, 8, 9, 2][i % 4], i / 4)) as u8)
                & [31, 63, 31, 255][i % 4];
        }
        history[index & 255] = values;
        write_alpha(&mut block[..8], [values[3], values[7]], |row, half| {
            alpha[pair(3 + row, half) as usize]
        });
        for (i, bytes) in block[8..12].chunks_exact_mut(2).enumerate() {
            let [r, g, b] = [0, 1, 2].map(|c| u16::from(values[i * 4 + c]));
            bytes.copy_from_slice(&((r << 11) | (g << 5) | b).to_le_bytes());
        }
        for row in 0..2 {
            let a = color[pair(10 + row, 0) as usize];
            let b = color[pair(10 + row, 1) as usize];
            block[12 + row * 2] = (a & 15) | (b << 4);
            block[13 + row * 2] = (a >> 4) | (b & 240);
        }
    }
    Ok(out)
}

pub fn decode_fitted_bc5(
    input: &[u8],
    size: usize,
    seeds: &Seeds<[u8; 4]>,
) -> Result<Vec<u8>, String> {
    let lanes = decode_symbols(input, size, Codec::Bc5)?;
    let mut out = vec![0; size];
    let profile = Profile {
        bc5: seeds.clone(),
        ..Profile::default()
    };
    rebuild_legacy(&lanes, &mut out, Codec::Bc5, &profile, false)?;
    Ok(out)
}

fn interleave(a: u8, b: u8, bits: usize) -> u16 {
    (0..bits)
        .map(|k| (u16::from((a >> k) & 1) << (2 * k)) | (u16::from((b >> k) & 1) << (2 * k + 1)))
        .sum()
}

fn reference<T: Copy>(
    history: &[T; 256],
    seeds: &Seeds<T>,
    index: usize,
    distance: usize,
    reset: T,
) -> Result<T, String> {
    if distance == 0 {
        return Ok(reset);
    }
    if distance <= index {
        return Ok(history[(index - distance) & 255]);
    }
    let slot = (256 + index - distance) as u8;
    seeds
        .get(&slot)
        .copied()
        .ok_or_else(|| format!("block {index} needs predictor slot {slot}"))
}

fn rebuild_legacy(
    symbols: &[u8],
    out: &mut [u8],
    codec: Codec,
    p: &Profile,
    strict: bool,
) -> Result<(), String> {
    let lane_count = out.len() / 8;
    let mut hist3 = [[0; 8]; 256];
    let mut hist4 = [[0; 2]; 256];
    let mut hist5 = [[0; 4]; 256];
    if strict && symbols[lane_count..lane_count * 2].iter().any(|&s| s != 0) {
        return Err(format!("{codec:?} has a nonzero reserved lane"));
    }
    for (i, block) in out.chunks_exact_mut(codec.block_bytes()).enumerate() {
        let single = |lane: usize| symbols[lane * lane_count + i];
        let pair = |lane: usize, half: usize| symbols[lane * lane_count + i * 2 + half];
        let distance = if codec == Codec::Bc4 {
            single(0)
        } else {
            (spread(pair(0, 0)) + 2 * spread(pair(0, 1))) as u8
        };
        let distance = zig(distance) as u8 as usize;
        match codec {
            Codec::Bc3 => {
                let mut values =
                    reference(&hist3, &p.bc3, i, distance, [128, 128, 0, 0, 0, 0, 0, 0])?;
                let deltas = [
                    pair(2, 0),
                    pair(2, 1),
                    pair(7, 0),
                    pair(8, 0),
                    pair(9, 0),
                    pair(7, 1),
                    pair(8, 1),
                    pair(9, 1),
                ];
                for ((value, delta), mask) in values
                    .iter_mut()
                    .zip(deltas)
                    .zip([255, 255, 31, 63, 31, 31, 63, 31])
                {
                    *value = value.wrapping_add(zig(delta) as u8) & mask;
                }
                hist3[i & 255] = values;
                write_alpha(&mut block[..8], [values[0], values[1]], |r, h| {
                    BC3_ALPHA_PAIR[pair(3 + r, h) as usize]
                });
                for (dst, rgb) in block[8..12]
                    .chunks_exact_mut(2)
                    .zip(values[2..].chunks_exact(3))
                {
                    let packed =
                        (u16::from(rgb[0]) << 11) | (u16::from(rgb[1]) << 5) | u16::from(rgb[2]);
                    dst.copy_from_slice(&packed.to_le_bytes());
                }
                for row in 0..2 {
                    let a = BC3_QUADRANT[pair(10 + row, 0) as usize];
                    let b = BC3_QUADRANT[pair(10 + row, 1) as usize];
                    block[12 + row * 2] = (a & 15) | (b << 4);
                    block[13 + row * 2] = (a >> 4) | (b & 240);
                }
            }
            Codec::Bc4 => {
                let bases = reference(&hist4, &p.bc4, i, distance, [128; 2])?;
                let end = [0, 1].map(|j| bases[j].wrapping_add(zig(single(j + 2)) as u8));
                hist4[i & 255] = end;
                write_alpha(block, end, |r, h| BC4_PAIR[single(4 + r * 2 + h) as usize]);
            }
            Codec::Bc5 => {
                let bases = reference(&hist5, &p.bc5, i, distance, [128; 4])?;
                let end =
                    [0, 1, 2, 3].map(|j| bases[j].wrapping_add(zig(pair(2 + j / 2, j % 2)) as u8));
                hist5[i & 255] = end;
                for (c, dst) in block.chunks_exact_mut(8).enumerate() {
                    write_alpha(dst, [end[c * 2], end[c * 2 + 1]], |r, h| {
                        BC5_PAIR[pair(4 + 4 * c + r, h) as usize]
                    });
                }
            }
            _ => unreachable!("only BC3 through BC5 use this reconstruction"),
        }
    }
    Ok(())
}

fn write_alpha(block: &mut [u8], end: [u8; 2], selectors: impl Fn(usize, usize) -> u8) {
    let mut bits = 0u64;
    for row in 0..4 {
        for half in 0..2 {
            bits |= u64::from(selectors(row, half)) << (12 * row + 6 * half);
        }
    }
    block[..2].copy_from_slice(&end);
    block[2..].copy_from_slice(&bits.to_le_bytes()[..6]);
}
