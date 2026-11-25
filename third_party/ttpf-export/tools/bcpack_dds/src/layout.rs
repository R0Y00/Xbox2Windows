use crate::pool::{Chunk, Result, Texture, MAX_BYTES, PAGE};

const PATTERNS_4K: [[i8; 12]; 2] = [
    [0, 0, 0, 1, -1, -2, 2, 3, -3, 4, -4, 5],
    [0, 0, 0, 0, -1, -2, 1, 2, -3, 3, -4, 4],
];
const PATTERNS_64K: [[i8; 16]; 2] = [
    [0, 0, 0, 1, -1, 2, 3, -2, -3, 4, -4, 5, -5, 6, -6, 7],
    [0, 0, 0, 0, 1, -1, 2, -2, -3, 3, -4, 4, -5, 5, -6, 6],
];

pub struct Mip {
    pub offset: usize,
    pub region: usize,
    pub x: Vec<usize>,
    pub y: Vec<usize>,
}

pub struct Layout {
    pub allocation: usize,
    pub output_size: usize,
    pub mips: Vec<Mip>,
    pub faces: usize,
    pub cube: bool,
    pub volume: bool,
}

impl Layout {
    pub fn new(texture: Texture, element: usize, chunks: &[Chunk]) -> Result<Self> {
        if texture.dimension == 3 && texture.mode == 2 && texture.mips == 1 {
            return Self::volume(texture, element);
        }
        let cube = texture.dimension == 2
            && texture.depth == 6
            && texture.non_tail <= 3
            && texture.mode == 18 + 2 * texture.non_tail as u32;
        if !cube && (texture.dimension != 1 || (texture.depth > 1 && texture.mode != 1)) {
            return Err("layout requires the reference exporter".into());
        }
        if texture.mips == 0
            || texture.mips > 1 + texture.width.max(texture.height).ilog2() as usize
        {
            return Err("invalid mip count".into());
        }
        let tile = if !cube && (texture.mode >> 1) & 15 == 0 {
            16 * PAGE
        } else {
            PAGE
        };
        let (half_bits, max_tail) = if tile == PAGE { (6, 8) } else { (8, 12) };
        let (pattern, origin_width): (&[i8], usize) = match (tile, element) {
            (PAGE, 8) => (&PATTERNS_4K[0], 8),
            (PAGE, 16) => (&PATTERNS_4K[1], 4),
            (_, 8) => (&PATTERNS_64K[0], 8),
            (_, 16) => (&PATTERNS_64K[1], 4),
            _ => return Err("unsupported block size".into()),
        };
        let log = element.ilog2() as usize;
        let block_width = 1 << (half_bits - log / 2);
        let block_height = 1 << (half_bits - log.div_ceil(2));
        let top_width = texture.width.div_ceil(4);
        let top_height = texture.height.div_ceil(4);
        let mut regions = Vec::with_capacity(texture.mips);
        let mut first_tail = texture.mips;
        for level in 0..texture.mips {
            let w = top_width.div_ceil(1 << level);
            let h = top_height.div_ceil(1 << level);
            if texture.mips > 1
                && w <= block_width / 2
                && h <= block_height
                && texture.mips - level <= max_tail
            {
                first_tail = level;
                break;
            }
            regions.push(w.div_ceil(block_width) * h.div_ceil(block_height) * tile);
        }
        let faces = texture.depth;
        let array_stride = if cube { 1 } else { faces };
        let mut allocation = if first_tail < texture.mips {
            tile * array_stride
        } else {
            0
        };
        let mut offsets = vec![0; texture.mips];
        for (level, region) in regions.iter().enumerate().rev() {
            offsets[level] = allocation;
            allocation += region * array_stride;
        }
        let face_size = allocation;
        let mut data_offset = 0;
        if cube {
            allocation *= 6;
            let pages = cube_chunk_sizes(texture, chunks)?;
            if texture.non_tail != 0 {
                let final_size = if allocation >= 36 * PAGE {
                    allocation.next_multiple_of(16 * PAGE)
                } else {
                    allocation
                };
                if pages.len() != texture.non_tail + 1 || pages.last() != Some(&final_size) {
                    return Err("cubemap final chunk size mismatch".into());
                }
                data_offset = pages[..pages.len() - 1].iter().sum();
                allocation = data_offset + final_size;
            } else if !pages.is_empty() && pages.as_slice() != [allocation] {
                return Err("cubemap chunk size mismatch".into());
            }
        }
        let mut mips = Vec::with_capacity(texture.mips);
        let mut output_size = 148usize;
        for (level, offset) in offsets.into_iter().enumerate() {
            let visible_w = (texture.width >> level).max(1).div_ceil(4);
            let visible_h = (texture.height >> level).max(1).div_ceil(4);
            output_size += visible_w * visible_h * element * faces;
            if output_size > MAX_BYTES {
                return Err("DDS exceeds the 256 MiB output limit".into());
            }
            let (tile_x, tile_y, tiles_w, region) = if level < first_tail {
                (
                    0,
                    0,
                    top_width.div_ceil(1 << level).div_ceil(block_width),
                    regions[level],
                )
            } else {
                let entry = max_tail - 1 - (level - first_tail);
                let packed = if entry > 6 { 16 << entry } else { entry << 8 };
                let spread = |start| -> usize {
                    (0..6)
                        .map(|bit| ((packed >> (start + 2 * bit)) & 1usize) << bit)
                        .sum()
                };
                (spread(9) * origin_width, spread(8) * 4, 1, tile)
            };
            let x = (0..visible_w)
                .map(|i| (i / block_width) * tile + address(pattern, i + tile_x, true))
                .collect();
            let y = (0..visible_h)
                .map(|i| (i / block_height) * tiles_w * tile + address(pattern, i + tile_y, false))
                .collect();
            mips.push(Mip {
                offset,
                region,
                x,
                y,
            });
        }
        let mips = (0..faces)
            .flat_map(|face| {
                mips.iter().map(move |mip| Mip {
                    offset: data_offset
                        + mip.offset
                        + face * if cube { face_size } else { mip.region },
                    region: mip.region,
                    x: mip.x.clone(),
                    y: mip.y.clone(),
                })
            })
            .collect();
        Ok(Self {
            allocation,
            output_size,
            mips,
            faces,
            cube,
            volume: false,
        })
    }

    fn volume(texture: Texture, element: usize) -> Result<Self> {
        let (bw, bh, bd, pattern) = match (texture.format, element) {
            (54, 2) => (8, 16, 16, [0, 1, 101, -1, 102, -2, 2, 103, -3, 3, 104, -4]),
            (10, 8) => (8, 8, 8, [0, 0, 0, 1, 101, -1, 2, 102, -2, 3, 103, -3]),
            _ => return Err("unsupported volume format".into()),
        };
        let output_size = 148 + texture.width * texture.height * texture.depth * element;
        let tiles_x = texture.width.div_ceil(bw);
        let z_stride = tiles_x * texture.height.div_ceil(bh) * PAGE;
        let allocation = texture.depth.div_ceil(bd) * z_stride;
        if output_size > MAX_BYTES || allocation > MAX_BYTES {
            return Err("volume exceeds the 256 MiB output limit".into());
        }
        let x: Vec<_> = (0..texture.width)
            .map(|i| i / bw * PAGE + address(&pattern, i, true))
            .collect();
        let mips = (0..texture.depth)
            .map(|z| {
                let within_tile: usize = pattern
                    .iter()
                    .enumerate()
                    .filter(|&(_, &axis)| axis >= 101)
                    .map(|(bit, axis)| ((z >> (axis - 101)) & 1) << bit)
                    .sum();
                let z_start = (z / bd) * z_stride;
                let y = (0..texture.height)
                    .map(|i| {
                        z_start
                            + (i / bh) * tiles_x * PAGE
                            + address(&pattern, i, false)
                            + within_tile
                    })
                    .collect();
                Mip {
                    offset: 0,
                    region: allocation,
                    x: x.clone(),
                    y,
                }
            })
            .collect();
        Ok(Self {
            allocation,
            output_size,
            mips,
            faces: 1,
            cube: false,
            volume: true,
        })
    }
}

fn cube_chunk_sizes(texture: Texture, chunks: &[Chunk]) -> Result<Vec<usize>> {
    if texture.extra == 0 && texture.meta == 0 {
        return Ok(Vec::new());
    }
    const PAGES: [usize; 32] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 12, 16, 24, 32, 48, 64, 128, 0, 0, 0, 0, 18, 80, 96, 144, 0,
        272, 384, 528, 0, 0, 0,
    ];
    if texture.extra >> 9 != 0 || chunks.first().map(|c| c.virtual_start) != Some(texture.base) {
        return Err("cubemap chunk metadata mismatch".into());
    }
    let mut fields = (texture.extra << 11) | texture.meta;
    let mut stream = Vec::new();
    while fields != 0 {
        let pages = PAGES[(fields & 31) as usize];
        if pages == 0 {
            return Err("invalid cubemap chunk-page code".into());
        }
        stream.push(pages * PAGE);
        fields >>= 5;
    }
    if stream.len() != chunks.len()
        || stream
            .iter()
            .zip(chunks)
            .any(|(bytes, chunk)| *bytes != chunk.virtual_end - chunk.virtual_start)
    {
        return Err("cubemap chunk sizes disagree with the index".into());
    }
    Ok(stream)
}

fn address(pattern: &[i8], coordinate: usize, x: bool) -> usize {
    pattern
        .iter()
        .enumerate()
        .filter(|&(_, &axis)| axis != 0 && axis < 100 && (axis > 0) == x)
        .map(|(bit, axis)| ((coordinate >> (axis.unsigned_abs() - 1)) & 1) << bit)
        .sum()
}
