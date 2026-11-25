mod layout;
mod pool;

use bcpack_calibration::blocks::{self, Codec, Profile};
use flate2::{Decompress, FlushDecompress, Status};
use layout::Layout;
use pool::{Chunk, Pool, Result, Texture};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Instant;

const USAGE: &str =
    "bcpack_dds <pool.ttpf> <output> [--jobs N] [--selection FILE] [--calibration PROFILE.json] [--entry-fits FITS.json] [--manifest FILE]";

#[derive(Deserialize)]
struct Selection {
    indices: Option<Vec<usize>>,
    #[serde(default)]
    start: usize,
    count: Option<usize>,
    #[serde(default)]
    bcpack_only: bool,
    #[serde(default)]
    prefer_fitted: bool,
}

#[derive(Deserialize)]
struct Calibration {
    #[serde(skip)]
    fits: BTreeMap<u64, EntryFit>,
    version: u32,
    bc1: Option<Bc1Calibration>,
    bc3: Option<Bc3Calibration>,
    #[serde(default)]
    bc4: BTreeMap<String, blocks::Seeds<[u8; 2]>>,
    #[serde(default)]
    bc5: blocks::Seeds<[u8; 4]>,
    bc6h: Option<Bc6hCalibration>,
    #[serde(skip)]
    bc6h_profile: Option<Profile>,
    bc7: Option<Bc7Calibration>,
    #[serde(default)]
    compatibility: BTreeMap<String, Vec<usize>>,
    #[serde(default)]
    entries: BTreeMap<String, BTreeMap<usize, String>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Patch {
    packed_offset: Option<usize>,
    #[serde(default)]
    bc7_endpoints: Vec<[usize; 3]>,
    #[serde(default)]
    bc7_indices: Vec<[usize; 3]>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct EntryFit {
    #[serde(default)]
    bc1_endpoints: BTreeMap<u8, [u8; 6]>,
    #[serde(default)]
    bc4_endpoints: blocks::Seeds<[u8; 2]>,
    #[serde(default)]
    bc5_endpoints: blocks::Seeds<[u8; 4]>,
    #[serde(default)]
    bc6h_endpoints: blocks::Seeds<[[u16; 4]; 3]>,
    #[serde(default)]
    strict: bool,
    shared_base: Option<String>,
    #[serde(default)]
    chunks: Vec<Patch>,
    packed_offset: Option<usize>,
    #[serde(default)]
    bc7_endpoints: Vec<[usize; 3]>,
    #[serde(default)]
    bc7_indices: Vec<[usize; 3]>,
}

fn apply_fields(
    seed: &mut [[u8; 2]],
    endpoints: &[[usize; 3]],
    indices: &[[usize; 3]],
) -> Result<()> {
    for &[slot, field, value] in endpoints {
        if slot >= 256 || field >= 24 || value > 255 {
            return Err("invalid BC7 endpoint override".into());
        }
        seed[((256 - slot) % 256) * 56 + field] = [value as u8, 8];
    }
    for &[slot, field, value] in indices {
        if slot >= 256 || field >= 32 || value > 15 {
            return Err("invalid BC7 selector override".into());
        }
        seed[((256 - slot) % 256) * 56 + 24 + ((field + 16) % 32)] = [(value as u8) << 4, 8];
    }
    Ok(())
}

impl EntryFit {
    fn decode(
        &self,
        bc7: &Bc7Calibration,
        packed: &[u8],
        length: usize,
        offset: usize,
        visible: &[usize],
    ) -> Result<Vec<u8>> {
        let name = self.shared_base.as_deref().unwrap_or("retail");
        let mut seed = bc7.seed(name)?;
        if self.packed_offset.is_none() || self.packed_offset == Some(offset) {
            apply_fields(&mut seed, &self.bc7_endpoints, &self.bc7_indices)?;
        }
        for chunk in self
            .chunks
            .iter()
            .filter(|chunk| chunk.packed_offset == Some(offset))
        {
            apply_fields(&mut seed, &chunk.bc7_endpoints, &chunk.bc7_indices)?;
        }
        let mut state = blocks::FittedBc7::unpack(packed, length)?;
        state.apply(
            bc7.labels.as_slice().try_into()?,
            bc7.anchors.as_slice().try_into()?,
            &seed,
        )?;
        if visible.iter().any(|&i| state.ready[i] == 0) {
            return Err("unresolved visible BC7 blocks using entry recipe".into());
        }
        Ok(state.blocks)
    }
}

#[derive(Deserialize)]
struct Bc1Calibration {
    endpoints: Vec<[i32; 6]>,
    selectors: Vec<Vec<u32>>,
}

#[derive(Deserialize)]
struct Bc3Calibration {
    endpoints: BTreeMap<i32, [u8; 8]>,
    color: Vec<u8>,
    alpha: Vec<u8>,
}

#[derive(Deserialize)]
struct Bc7Calibration {
    labels: Vec<[i16; 4]>,
    anchors: Vec<u16>,
    seeds: BTreeMap<String, Vec<Vec<[u8; 2]>>>,
    #[serde(default)]
    adjustments: BTreeMap<String, Vec<[usize; 3]>>,
}

impl Bc7Calibration {
    fn decode(
        &self,
        input: &[u8],
        length: usize,
        profile: &str,
        visible: &[usize],
    ) -> Result<Vec<u8>> {
        let labels = self.labels.as_slice().try_into()?;
        let anchors = self.anchors.as_slice().try_into()?;
        let names: &[&str] = match profile {
            "early" => &["early"],
            "refined" => &["base", "secondary", "refined"],
            _ => &["base", "secondary"],
        };
        let mut state = blocks::FittedBc7::unpack(input, length)?;
        for name in names {
            let seed = self.seed(name)?;
            state.apply(labels, anchors, &seed)?;
            if state.ready.iter().all(|&ready| ready != 0) {
                return Ok(state.blocks);
            }
        }
        let missing: Vec<_> = visible
            .iter()
            .copied()
            .filter(|&i| state.ready[i] == 0)
            .collect();
        if missing.is_empty() {
            return Ok(state.blocks);
        }
        if profile != "refined" {
            return Err("unresolved visible fitted BC7 blocks".into());
        }
        let mut hash = Sha256::new();
        hash.update(&state.symbols);
        for &i in &missing {
            hash.update((i as i64).to_le_bytes());
        }
        let rows = self
            .adjustments
            .get(&format!("{:x}", hash.finalize()))
            .ok_or("missing adjustment for visible fitted BC7 blocks")?;
        let mut seed = self.seed("refined")?;
        for &[age, field, value] in rows {
            if age >= 256 || field >= 24 || value > 255 {
                return Err("invalid fitted BC7 endpoint adjustment".into());
            }
            seed[age * 56 + field] = [value as u8, 8];
        }
        state.apply(labels, anchors, &seed)?;
        if missing.iter().any(|&i| state.ready[i] == 0) {
            return Err("unresolved visible fitted BC7 blocks after adjustment".into());
        }
        Ok(state.blocks)
    }

    fn seed(&self, name: &str) -> Result<Vec<[u8; 2]>> {
        let seed = self
            .seeds
            .get(name)
            .ok_or_else(|| format!("missing fitted BC7 seed {name}"))?;
        if seed.len() != 256 || seed.iter().any(|row| row.len() != 56) {
            return Err("invalid fitted BC7 seed table".into());
        }
        Ok(seed.iter().flatten().copied().collect())
    }
}

#[derive(Deserialize, Default)]
#[serde(default, rename_all = "camelCase")]
struct Bc6hCalibration {
    endpoint_history: HistoryRows<[[u16; 4]; 3]>,
    index_history: HistoryRows<[u8; 16]>,
    endpoint_class: HistoryRows<u8>,
    index_class: HistoryRows<u8>,
    endpoint_confirmed_bits: HistoryRows<[[u8; 4]; 3]>,
    index_confirmed_bits: HistoryRows<[u8; 16]>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum HistoryRows<T> {
    Dense(Vec<T>),
    Sparse(BTreeMap<String, T>),
}

impl<T> Default for HistoryRows<T> {
    fn default() -> Self {
        Self::Sparse(BTreeMap::new())
    }
}

impl<T: Copy> HistoryRows<T> {
    fn rows(&self) -> Result<blocks::Seeds<T>> {
        match self {
            Self::Dense(rows) if rows.len() == 256 => {
                Ok((0..=255).zip(rows.iter().copied()).collect())
            }
            Self::Dense(_) => Err("dense BC6H tables must have 256 rows".into()),
            Self::Sparse(rows) => rows
                .iter()
                .map(|(key, &row)| Ok((key.parse()?, row)))
                .collect(),
        }
    }
}

impl Bc6hCalibration {
    fn profile(&self) -> Result<Profile> {
        let ep_classes = self.endpoint_class.rows()?;
        let ix_classes = self.index_class.rows()?;
        if ep_classes
            .values()
            .chain(ix_classes.values())
            .any(|&class| class > 3)
        {
            return Err("invalid fitted BC6H predictor class".into());
        }
        if self
            .endpoint_confirmed_bits
            .rows()?
            .values()
            .any(|channels| {
                channels.iter().any(|ch| {
                    ch.iter()
                        .zip([16, 10, 6, 6])
                        .any(|(&depth, max)| depth > max)
                })
            })
            || self
                .index_confirmed_bits
                .rows()?
                .values()
                .flatten()
                .any(|&depth| depth > 4)
        {
            return Err("invalid fitted BC6H predictor precision".into());
        }
        let known =
            |classes: &blocks::Seeds<u8>, slot: &u8| classes.get(slot).copied().unwrap_or(0) != 0;
        let bc6h_endpoints = self
            .endpoint_history
            .rows()?
            .into_iter()
            .filter(|(slot, _)| known(&ep_classes, slot))
            .collect();
        let indices = self.index_history.rows()?;
        if indices.values().flatten().any(|&index| index >= 16) {
            return Err("invalid fitted BC6H index predictor".into());
        }
        let bc6h_indices = indices
            .into_iter()
            .filter(|(slot, _)| known(&ix_classes, slot))
            .collect();
        Ok(Profile {
            bc6h_endpoints,
            bc6h_indices,
            ..Profile::default()
        })
    }
}

enum Mode<'a> {
    Strict(&'a Profile),
    Fitted(&'a Calibration, &'a str, Option<&'a EntryFit>),
}

impl Mode<'_> {
    fn decode(
        &self,
        format: u32,
        chunk: Chunk,
        packed: &[u8],
        visible: &[usize],
    ) -> Result<Vec<u8>> {
        let length = chunk.virtual_end - chunk.virtual_start;
        match chunk.codec {
            0 => return Ok(packed.to_vec()),
            8 => {
                let mut raw = vec![0; length];
                let mut decoder = Decompress::new(true);
                if decoder.decompress(packed, &mut raw, FlushDecompress::Finish)?
                    != Status::StreamEnd
                    || decoder.total_out() != length as u64
                {
                    return Err("invalid zlib chunk length".into());
                }
                return Ok(raw);
            }
            _ => {}
        }
        let codec = match (format, chunk.codec) {
            (70..=72, 1) => Codec::Bc1,
            (76..=78, 3) => Codec::Bc3,
            (79..=81, 4) => Codec::Bc4,
            (82..=84, 5) => Codec::Bc5,
            (94..=96, 6) => Codec::Bc6h,
            (97..=99, 7) => Codec::Bc7,
            _ => return Err("chunk codec does not match the texture's DXGI format".into()),
        };
        Ok(match self {
            Mode::Strict(profile) => blocks::decode(packed, length, codec, profile)?,
            Mode::Fitted(profile, _, recipe) if codec == Codec::Bc1 => {
                let bc1 = profile
                    .bc1
                    .as_ref()
                    .ok_or("profile has no fitted BC1 table")?;
                let mut endpoints: [[i32; 6]; 256] = bc1.endpoints.as_slice().try_into()?;
                if let Some(recipe) = recipe {
                    for (&slot, values) in &recipe.bc1_endpoints {
                        if values
                            .iter()
                            .zip([31, 63, 31, 31, 63, 31])
                            .any(|(&v, limit)| v > limit)
                        {
                            return Err("BC1 endpoint exceeds RGB565 component range".into());
                        }
                        endpoints[(256 - usize::from(slot)) % 256] = values.map(i32::from);
                    }
                }
                let selectors: Vec<[u32; 256]> = bc1
                    .selectors
                    .iter()
                    .map(|row| row.as_slice().try_into())
                    .collect::<std::result::Result<_, _>>()?;
                let rotation = if profile.compatibility.is_empty() {
                    None
                } else {
                    profile
                        .compatibility
                        .get(&format!("{:x}", Sha256::digest(packed)))
                };
                blocks::decode_fitted_bc1(
                    packed,
                    length,
                    &endpoints,
                    selectors.as_slice().try_into()?,
                    rotation.map_or(&[], Vec::as_slice),
                )?
            }
            Mode::Fitted(profile, _, _) if codec == Codec::Bc3 => {
                let bc3 = profile
                    .bc3
                    .as_ref()
                    .ok_or("profile has no fitted BC3 table")?;
                blocks::decode_fitted_bc3(
                    packed,
                    length,
                    &bc3.endpoints,
                    bc3.color.as_slice().try_into()?,
                    bc3.alpha.as_slice().try_into()?,
                )?
            }
            Mode::Fitted(profile, name, fit) if codec == Codec::Bc4 => {
                let seeds = profile
                    .bc4
                    .get(*name)
                    .ok_or("profile has no fitted BC4 table for this entry")?;
                if let Some(fit) = fit.filter(|fit| !fit.bc4_endpoints.is_empty()) {
                    let mut fitted = seeds.clone();
                    fitted.extend(fit.bc4_endpoints.iter().map(|(&slot, &end)| (slot, end)));
                    blocks::decode_fitted_bc4(packed, length, &fitted)?
                } else {
                    blocks::decode_fitted_bc4(packed, length, seeds)?
                }
            }
            Mode::Fitted(profile, _, entry) if codec == Codec::Bc5 => {
                let mut seeds = profile.bc5.clone();
                if let Some(fit) = entry {
                    seeds.extend(
                        fit.bc5_endpoints
                            .iter()
                            .map(|(&slot, &values)| (slot, values)),
                    );
                }
                blocks::decode_fitted_bc5(packed, length, &seeds)?
            }
            Mode::Fitted(profile, _, fit) if codec == Codec::Bc6h => {
                let profile = profile
                    .bc6h_profile
                    .as_ref()
                    .ok_or("profile has no fitted BC6H tables")?;
                if let Some(recipe) = fit.filter(|r| !r.bc6h_endpoints.is_empty()) {
                    let mut fitted = Profile {
                        bc6h_endpoints: profile.bc6h_endpoints.clone(),
                        bc6h_indices: profile.bc6h_indices.clone(),
                        ..Profile::default()
                    };
                    for (&slot, channels) in &recipe.bc6h_endpoints {
                        if channels
                            .iter()
                            .any(|p| p[1] >= 1024 || p[2] >= 64 || p[3] >= 64)
                        {
                            return Err("BC6H endpoint exceeds history precision".into());
                        }
                        fitted.bc6h_endpoints.insert(slot, *channels);
                    }
                    blocks::decode(packed, length, codec, &fitted)?
                } else {
                    blocks::decode(packed, length, codec, profile)?
                }
            }
            Mode::Fitted(profile, name, fit) if codec == Codec::Bc7 => {
                let bc7 = profile
                    .bc7
                    .as_ref()
                    .ok_or("profile has no fitted BC7 table")?;
                match fit {
                    Some(recipe) => {
                        recipe.decode(bc7, packed, length, chunk.stored_start, visible)?
                    }
                    None => bc7.decode(packed, length, name, visible)?,
                }
            }
            Mode::Fitted(..) => return Err("fitted codec requires the reference exporter".into()),
        })
    }
}

fn header(texture: Texture, layout: &Layout, element: usize) -> [u8; 148] {
    let mut data = [0; 148];
    let mut flags = if layout.volume { 0x80100f } else { 0x81007 };
    let mut caps = 0x1000;
    if texture.mips > 1 {
        flags |= 0x20000;
        caps |= 0x400008;
    }
    if layout.volume || layout.faces > 1 {
        caps |= 8;
    }
    for (offset, value) in [
        (4, 124),
        (8, flags),
        (12, texture.height as u32),
        (16, texture.width as u32),
        (
            20,
            if layout.volume {
                (texture.width * element) as u32
            } else {
                (texture.width.div_ceil(4) * texture.height.div_ceil(4) * element) as u32
            },
        ),
        (
            24,
            if layout.volume {
                texture.depth as u32
            } else {
                0
            },
        ),
        (28, texture.mips as u32),
        (76, 32),
        (80, 4),
        (108, caps),
        (
            112,
            if layout.cube {
                0xfe00
            } else if layout.volume {
                0x200000
            } else {
                0
            },
        ),
        (128, texture.format),
        (132, if layout.volume { 4 } else { 3 }),
        (136, if layout.cube { 4 } else { 0 }),
        (140, if layout.cube { 1 } else { layout.faces as u32 }),
    ] {
        data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    data[..4].copy_from_slice(b"DDS ");
    data[84..88].copy_from_slice(b"DX10");
    data
}

fn encode(
    pool: &Pool,
    source: &mut File,
    texture: Texture,
    layout: &Layout,
    element: usize,
    mode: Mode,
) -> Result<Vec<u8>> {
    let mut output = Vec::with_capacity(layout.output_size);
    output.extend(header(texture, layout, element));
    let mut cache = BTreeMap::new();
    let mut cache_bytes = 0;
    let mut packed = Vec::new();
    let mut region = Vec::new();
    let mut last_region = None;
    for mip in &layout.mips {
        let start = texture.base + mip.offset;
        let end = start + mip.region;

        // Fitted BC7 can resolve different visible blocks on each mip of a shared tail.
        let fitted_bc7 = matches!(mode, Mode::Fitted(..)) && matches!(texture.format, 97..=99);
        if fitted_bc7 || last_region != Some((start, end)) {
            last_region = Some((start, end));
            region.clear();
            let chunks = pool.chunks_for(start, mip.region)?;
            let mut visible = BTreeMap::<usize, Vec<usize>>::new();
            if fitted_bc7
                && pool.chunks[chunks.clone()]
                    .iter()
                    .any(|chunk| chunk.codec == 7)
            {
                for addr in mip
                    .y
                    .iter()
                    .flat_map(|y| mip.x.iter().map(move |x| start + x + y))
                {
                    let index = chunks.start
                        + pool.chunks[chunks.clone()]
                            .partition_point(|chunk| chunk.virtual_end <= addr);
                    let chunk = pool
                        .chunks
                        .get(index)
                        .filter(|_| index < chunks.end)
                        .ok_or("tile address exceeds the mip allocation")?;
                    if chunk.codec == 7 {
                        visible
                            .entry(index)
                            .or_default()
                            .push((addr - chunk.virtual_start) / element);
                    }
                }
            }
            for index in chunks {
                let chunk = pool.chunks[index];
                let lo = start.max(chunk.virtual_start) - chunk.virtual_start;
                let hi = end.min(chunk.virtual_end) - chunk.virtual_start;
                if fitted_bc7 && chunk.codec == 7 {
                    let Some(visible) = visible.get(&index) else {
                        region.resize(region.len() + hi - lo, 0);
                        continue;
                    };
                    source.seek(SeekFrom::Start(chunk.stored_start as u64))?;
                    packed.resize(chunk.stored_len, 0);
                    source.read_exact(&mut packed)?;
                    let decoded = mode.decode(texture.format, chunk, &packed, visible)?;
                    region.extend_from_slice(&decoded[lo..hi]);
                    continue;
                }
                if !cache.contains_key(&index) {
                    source.seek(SeekFrom::Start(chunk.stored_start as u64))?;
                    packed.resize(chunk.stored_len, 0);
                    source.read_exact(&mut packed)?;
                    let decoded = mode.decode(texture.format, chunk, &packed, &[])?;
                    if cache_bytes + decoded.len() > 32 << 20 {
                        cache.clear();
                        cache_bytes = 0;
                    }
                    cache_bytes += decoded.len();
                    cache.insert(index, decoded);
                }
                region.extend_from_slice(&cache[&index][lo..hi]);
            }
        }
        for &y in &mip.y {
            for &x in &mip.x {
                let block = region
                    .get(x + y..x + y + element)
                    .ok_or("tile address exceeds the mip allocation")?;
                output.extend_from_slice(block);
            }
        }
    }
    Ok(output)
}

fn publish(path: &Path, data: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut file = tempfile::NamedTempFile::new_in(parent)?;
    file.write_all(data)?;
    file.persist(path)?;
    Ok(())
}

fn entry(
    pool: &Pool,
    source: &mut File,
    basename: &str,
    index: usize,
    out: &Path,
    strict: &Profile,
    calibration: Option<&Calibration>,
    selection: &Selection,
) -> Value {
    let texture = pool.textures[index];
    let Some(span) = pool.span(index) else {
        return json!({"index": index, "status": "no_layout", "reason": "no mapped allocation"});
    };
    let (element, format_name) = match texture.format {
        10 if texture.dimension == 3 => (8, "Format 10"),
        54 if texture.dimension == 3 => (2, "Format 54"),
        70..=72 => (
            8,
            if texture.format == 72 {
                "BC1_SRGB"
            } else {
                "BC1"
            },
        ),
        73..=75 => (
            16,
            if texture.format == 75 {
                "BC2_SRGB"
            } else {
                "BC2"
            },
        ),
        76..=78 => (
            16,
            if texture.format == 78 {
                "BC3_SRGB"
            } else {
                "BC3"
            },
        ),
        79..=81 => (
            8,
            if texture.format == 81 {
                "BC4_SNORM"
            } else {
                "BC4"
            },
        ),
        82..=84 => (
            16,
            if texture.format == 84 {
                "BC5_SNORM"
            } else {
                "BC5"
            },
        ),
        94..=96 => (
            16,
            if texture.format == 96 {
                "BC6H_SF16"
            } else {
                "BC6H"
            },
        ),
        97..=99 => (
            16,
            if texture.format == 99 {
                "BC7_SRGB"
            } else {
                "BC7"
            },
        ),
        _ => {
            return json!({"index": index, "status": "uncompressed", "format": texture.format,
                            "width": texture.width, "height": texture.height})
        }
    };
    let failed = |reason: String| {
        json!({"index": index, "status": "failed", "reason": reason,
        "format": texture.format, "format_name": format_name, "width": texture.width, "height": texture.height})
    };
    let chunks = match pool.chunks_for(texture.base, span) {
        Ok(chunks) => chunks,
        Err(error) => return failed(error.to_string()),
    };
    if selection.bcpack_only
        && !pool.chunks[chunks.clone()]
            .iter()
            .any(|chunk| matches!(chunk.codec, 1 | 3..=7))
    {
        return json!({"index": index, "status": "skipped"});
    }
    let layout = match Layout::new(texture, element, &pool.chunks[chunks.clone()]) {
        Ok(layout) if layout.allocation == span => layout,
        Ok(_) => {
            return json!({"index": index, "status": "no_layout",
                               "reason": "layout does not match the allocation span"})
        }
        Err(error) => return failed(error.to_string()),
    };
    let supported = layout.mips.iter().all(|mip| {
        pool.chunks_for(texture.base + mip.offset, mip.region)
            .is_ok_and(|range| {
                pool.chunks[range]
                    .iter()
                    .all(|chunk| matches!(chunk.codec, 0 | 1 | 3..=8))
            })
    });
    if !supported {
        return json!({"index": index, "status": "unsupported_codec", "format": texture.format,
           "format_name": format_name, "width": texture.width, "height": texture.height});
    }
    let entry_fit = pool
        .asset_keys
        .get(index)
        .and_then(|key| calibration?.fits.get(key));
    let fitted = |source: &mut File, profile: &Calibration| {
        if texture.dimension != 1
            || (texture.depth == 1
                && texture.mips > 1
                && !(texture.width.is_power_of_two() && texture.height.is_power_of_two()))
        {
            return Err("fitted layout requires the reference exporter".into());
        }
        let name = profile
            .entries
            .get(basename)
            .and_then(|entries| entries.get(&index))
            .map_or("base", String::as_str);
        encode(
            pool,
            source,
            texture,
            &layout,
            element,
            Mode::Fitted(profile, name, entry_fit),
        )
        .map(|data| (data, "fitted"))
    };
    let encode_strict = |source: &mut File| {
        encode(
            pool,
            source,
            texture,
            &layout,
            element,
            Mode::Strict(strict),
        )
        .map(|data| (data, "strict"))
    };
    let result = match calibration {
        _ if entry_fit.is_some_and(|fit| fit.strict) => encode_strict(source),
        Some(profile) if entry_fit.is_some() => fitted(source, profile),
        Some(profile) if selection.prefer_fitted && texture.dimension == 1 => {
            fitted(source, profile)
        }
        Some(profile) => encode_strict(source).or_else(|_| fitted(source, profile)),
        None => encode_strict(source),
    };
    let (data, mode) = match result {
        Ok(success) => success,
        Err(error) => return failed(error.to_string()),
    };
    let name = format!("tex_{index:05}.dds");
    let hash = format!("{:x}", Sha256::digest(&data));
    if let Err(error) = publish(&out.join(&name), &data) {
        return failed(error.to_string());
    }
    let recipe = match (mode, entry_fit) {
        ("strict", Some(fit)) if fit.strict => "entry_strict",
        ("strict", _) => "strict",
        (_, Some(_)) => "entry_fit",
        _ => "shared_profile",
    };
    json!({"index": index, "status": "success", "mode": mode, "recipe": recipe, "format": texture.format,
           "format_name": format_name, "width": texture.width, "height": texture.height,
           "mips": texture.mips, "dds_size": data.len(), "sha256": hash, "file": name})
}

fn run() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let input = PathBuf::from(args.next().ok_or(USAGE)?);
    if input == Path::new("--help") || input == Path::new("-h") {
        println!("{USAGE}");
        return Ok(());
    }
    let out = PathBuf::from(args.next().ok_or(USAGE)?);
    let mut selection = Selection {
        indices: None,
        start: 0,
        count: None,
        bcpack_only: false,
        prefer_fitted: false,
    };
    let mut manifest = out.join("manifest.json");
    let mut calibration = None;
    let mut entry_file = None;
    let mut jobs = thread::available_parallelism().map_or(4, |n| n.get().min(16));
    while let Some(flag) = args.next() {
        let value = args.next().ok_or(USAGE)?;
        match flag.to_str() {
            Some("--jobs") => {
                jobs = value.to_str().ok_or(USAGE)?.parse::<usize>()?;
                if !(1..=256).contains(&jobs) {
                    return Err("--jobs needs a value from 1 to 256".into());
                }
            }
            Some("--selection") => selection = serde_json::from_slice(&fs::read(value)?)?,
            Some("--manifest") => manifest = value.into(),
            Some("--entry-fits") => entry_file = Some(value),
            Some("--calibration") => {
                let mut profile: Calibration = serde_json::from_slice(&fs::read(value)?)?;
                if profile.version != 1 {
                    return Err("Unsupported calibration version (expected version 1)".into());
                }
                if let Some(bc1) = &profile.bc1 {
                    if bc1.endpoints.len() != 256
                        || bc1.selectors.len() != 4
                        || bc1.selectors.iter().any(|row| row.len() != 256)
                        || bc1
                            .endpoints
                            .iter()
                            .flatten()
                            .any(|&n| !(-1..=63).contains(&n))
                    {
                        return Err("Invalid fitted BC1 table shape or range".into());
                    }
                }
                profile.bc6h_profile =
                    profile.bc6h.take().map(|bc6h| bc6h.profile()).transpose()?;
                calibration = Some(profile);
            }
            _ => return Err(USAGE.into()),
        }
    }
    if let Some(path) = entry_file {
        let calibration = calibration
            .as_mut()
            .ok_or("--entry-fits requires --calibration")?;
        let fits: BTreeMap<String, EntryFit> = serde_json::from_slice(&fs::read(path)?)?;
        for (key, fit) in fits {
            if key.len() != 16 {
                return Err("expected a 16-digit hexadecimal asset key".into());
            }
            if fit.chunks.iter().any(|p| p.packed_offset.is_none()) {
                return Err("each chunk override requires packedOffset".into());
            }
            if calibration
                .fits
                .insert(u64::from_str_radix(&key, 16)?, fit)
                .is_some()
            {
                return Err("duplicate numeric asset key".into());
            }
        }
    }
    let mut profile = Profile {
        bc1: serde_json::from_str(include_str!("../../bcpack_bc1_history.json"))?,
        ..Profile::default()
    };
    profile.bc7.add_signature([20, 0, 6, 0, 0, 0])?;
    let start_time = Instant::now();
    let pool = Pool::open(&input)?;
    let total = pool.textures.len();
    let basename = input
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    let indices = match selection.indices.as_ref() {
        Some(indices) => indices
            .iter()
            .copied()
            .filter(|&i| i < total)
            .collect::<BTreeSet<_>>(),
        None => (selection.start
            ..selection
                .count
                .map_or(total, |n| selection.start.saturating_add(n).min(total)))
            .collect(),
    };
    let indices: Vec<_> = indices.into_iter().collect();
    fs::create_dir_all(&out)?;
    let next = AtomicUsize::new(0);
    let mut rows: Vec<_> = thread::scope(|scope| -> Result<_> {
        let handles: Vec<_> = (0..jobs.min(indices.len()))
            .map(|_| {
                scope.spawn(|| -> Result<_> {
                    let mut source = File::open(&input)?;
                    let mut rows = Vec::new();
                    while let Some(&index) = indices.get(next.fetch_add(1, Ordering::Relaxed)) {
                        rows.push((
                            index,
                            entry(
                                &pool,
                                &mut source,
                                basename,
                                index,
                                &out,
                                &profile,
                                calibration.as_ref(),
                                &selection,
                            ),
                        ));
                    }
                    Ok(rows)
                })
            })
            .collect();
        let mut rows = Vec::with_capacity(indices.len());
        for handle in handles {
            rows.extend(handle.join().map_err(|_| "export worker panicked")??);
        }
        Ok(rows)
    })?;
    rows.sort_unstable_by_key(|(index, _)| *index);
    let rows: Vec<_> = rows.into_iter().map(|(_, row)| row).collect();
    publish(&manifest, &serde_json::to_vec_pretty(&rows)?)?;
    let mut stats = BTreeMap::from([
        ("strict", 0),
        ("provisional", 0),
        ("failed", 0),
        ("skipped", 0),
    ]);
    if calibration.is_some() {
        stats.insert("fitted", 0);
    }
    for row in &rows {
        let key = match row["status"].as_str() {
            Some("success") => row["mode"].as_str().unwrap(),
            Some("failed") => "failed",
            _ => "skipped",
        };
        *stats.entry(key).or_insert(0) += 1;
    }
    println!(
        "{}",
        json!({"pool": input, "total_textures": total, "processed": indices.len(),
        "manifest": manifest, "entries": rows.len(), "elapsed_seconds": start_time.elapsed().as_secs_f64(),
        "stats": stats})
    );
    Ok(())
}

fn main() -> std::process::ExitCode {
    if let Err(error) = run() {
        eprintln!("{error}");
        return std::process::ExitCode::FAILURE;
    }
    std::process::ExitCode::SUCCESS
}
