use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub const PAGE: usize = 4096;
pub const MAX_BYTES: usize = 256 << 20;
const SECTOR: usize = 256;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Clone, Copy)]
pub struct Chunk {
    pub virtual_start: usize,
    pub virtual_end: usize,
    pub stored_start: usize,
    pub stored_len: usize,
    pub codec: u8,
}

#[derive(Clone, Copy)]
pub struct Texture {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub mips: usize,
    pub non_tail: usize,
    pub format: u32,
    pub dimension: u32,
    pub mode: u32,
    pub base: usize,
    pub meta: u32,
    pub extra: u32,
}

impl Texture {
    fn read(row: &[u8]) -> Self {
        let a = word32(row, 0);
        let b = word32(row, 4);
        let c = word32(row, 8);
        Self {
            width: ((a & 0x3fff) + 1) as usize,
            height: (((a >> 14) & 0x3fff) + 1) as usize,
            depth: (((b >> 10) & 0x7ff) + 1) as usize,
            mips: ((a >> 28) + (b & 7)) as usize,
            non_tail: (b & 7) as usize,
            format: (b >> 3) & 0x7f,
            dimension: (b >> 21) & 3,
            mode: b >> 23,
            base: (c & 0x1f_ffff) as usize * PAGE,
            meta: c >> 21,
            extra: word32(row, 12),
        }
    }

    fn mapped(self) -> bool {
        self.base != 0 && self.base != 0x1f_ffff * PAGE
    }
}

pub struct Pool {
    pub chunks: Vec<Chunk>,
    pub textures: Vec<Texture>,
    pub asset_keys: Vec<u64>,
    spans: Vec<Option<usize>>,
}

impl Pool {
    pub fn open(path: &Path) -> Result<Self> {
        let mut source = File::open(path)?;
        let length = usize::try_from(source.metadata()?.len())?;
        let mut header = [0; 24];
        source.read_exact(&mut header)?;
        let count = word32(&header, 4) as usize;
        let size = word64(&header, 8);
        let virtual_size = usize::try_from(size & ((1 << 56) - 1))?;
        let words = word32(&header, 16) as usize;
        if length < SECTOR
            || &header[..4] != b"RDS5"
            || !(3..=1_000_002).contains(&count)
            || size >> 56 != 22
            || virtual_size < PAGE
            || !(count.div_ceil(32)..=count.div_ceil(32) + 1).contains(&words)
        {
            return Err("unsupported RDS5 index layout".into());
        }
        let table_size = (count - 1) * 8;
        let etpf = (24 + table_size).next_multiple_of(SECTOR);
        if etpf + 16 > length {
            return Err("RDS5 index exceeds its source".into());
        }
        let mut table = vec![0; table_size];
        source.read_exact(&mut table)?;
        let file_pos = |v: u64| (v & 0x3ff_ffff) as usize * SECTOR;
        let virtual_pos = |v: u64| ((v >> 30) & 0x3f_ffff) as usize * PAGE;
        let sentinel = word64(&table, table_size - 8);
        if file_pos(sentinel) != length || virtual_pos(sentinel) != virtual_size {
            return Err("RDS5 sentinel does not match its lengths".into());
        }

        let mut chunks = Vec::with_capacity(count - 2);
        for index in 0..count - 2 {
            let entry = word64(&table, index * 8);
            let next = word64(&table, (index + 1) * 8);
            let start = file_pos(entry);
            let end = file_pos(next);
            let virtual_start = virtual_pos(entry);
            let virtual_end = virtual_pos(next);
            let padding = (entry >> 52) as usize;
            let codec = ((entry >> 26) & 15) as u8;
            let stored_len = end.saturating_sub(start).saturating_sub(padding);
            let decoded_len = virtual_end.saturating_sub(virtual_start);
            if start < etpf + 16
                || end > length
                || virtual_end > virtual_size
                || !(1..=MAX_BYTES).contains(&stored_len)
                || !(1..=MAX_BYTES).contains(&decoded_len)
                || (codec == 0 && stored_len != decoded_len)
            {
                return Err(format!("RDS5 chunk {index} range is invalid").into());
            }
            chunks.push(Chunk {
                virtual_start,
                virtual_end,
                stored_start: start,
                stored_len,
                codec,
            });
        }

        source.seek(SeekFrom::Start(etpf as u64))?;
        let mut embedded = [0; 16];
        source.read_exact(&mut embedded)?;
        let count = word32(&embedded, 8) as usize;
        let records = word32(&embedded, 12) as usize;
        let alignment = word32(&header, 20);
        let boundary = (16 + records + count * 32).next_multiple_of(PAGE);
        if &embedded[..4] != b"ETPF"
            || word32(&embedded, 4) != 634
            || !(1..=1_000_000).contains(&count)
            || records < count * 16
            || alignment & 0x0fff_ffff != 0
            || boundary != chunks[0].virtual_start
            || etpf + boundary + (alignment >> 28) as usize * SECTOR != chunks[0].stored_start
        {
            return Err("ETPF tables do not match the data boundary".into());
        }
        let mut descriptor_bytes = vec![0; count * 16];
        source.read_exact(&mut descriptor_bytes)?;
        let textures: Vec<_> = descriptor_bytes
            .chunks_exact(16)
            .map(Texture::read)
            .collect();
        let key_count = count;
        let mut key_bytes = vec![0; key_count * 8];
        source.read_exact(&mut key_bytes)?;
        let asset_keys = key_bytes
            .chunks_exact(8)
            .map(|key| word64(key, 0))
            .collect();
        let mut spans = vec![None; textures.len()];
        let mut next_base = virtual_size;
        let mut next_distinct = virtual_size;
        for (index, texture) in textures.iter().enumerate().rev() {
            if !texture.mapped() || texture.base >= virtual_size {
                continue;
            }
            if next_base != texture.base {
                next_distinct = next_base;
            }
            spans[index] = next_distinct
                .checked_sub(texture.base)
                .filter(|&span| span != 0);
            next_base = texture.base;
        }
        Ok(Self {
            chunks,
            textures,
            asset_keys,
            spans,
        })
    }

    pub fn span(&self, index: usize) -> Option<usize> {
        self.spans[index]
    }

    pub fn chunks_for(&self, start: usize, size: usize) -> Result<std::ops::Range<usize>> {
        let end = start.checked_add(size).ok_or("virtual range overflow")?;
        let first = self
            .chunks
            .partition_point(|chunk| chunk.virtual_end <= start);
        let last = self
            .chunks
            .partition_point(|chunk| chunk.virtual_start < end);
        if first >= last
            || self.chunks[first].virtual_start > start
            || self.chunks[last - 1].virtual_end < end
        {
            return Err("texture crosses an unmapped virtual range".into());
        }
        Ok(first..last)
    }
}

fn word32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn word64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}
