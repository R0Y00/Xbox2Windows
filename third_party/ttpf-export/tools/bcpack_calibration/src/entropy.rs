use bcpack_rans::rans::{model, Model, FEED_WINDOW, L, MAX_LANES};
use std::slice;

#[derive(Clone, Copy)]
enum Phase {
    Header,
    Length,
    LengthHigh,
    Symbols,
}

struct Lane {
    phase: Phase,
    model: &'static Model,
    remaining: usize,
    unread: usize,
    state: u32,
    byte: Option<u8>,
    held: Option<u16>,
    zeros: u8,
    flush: bool,
    long: bool,
    done: bool,
    written: usize,
}

impl Lane {
    fn new(count: usize) -> Self {
        Self {
            phase: Phase::Header,
            model: model(0, 0),
            remaining: count,
            unread: 0,
            state: 0,
            byte: None,
            held: None,
            zeros: 0,
            flush: false,
            long: false,
            done: false,
            written: 0,
        }
    }

    fn pad(&mut self) {
        while self.state != 0 && self.state < L && self.zeros < 8 {
            self.state <<= 8;
            self.zeros += 1;
        }
    }

    fn emit(&mut self, symbol: u16, output: &mut [u8]) {
        output[self.written] = symbol as u8;
        self.written += 1;
    }

    fn step(&mut self, default_width: u8) -> Option<()> {
        match self.phase {
            Phase::Header => {
                let header = self.byte.take()?;
                let width = default_width.checked_sub(header >> 6)?;
                if width == 0 {
                    return None;
                }
                self.model = model(width - 1, header & 15);
                self.flush = header & 16 != 0;
                self.long = header & 32 != 0;
                self.zeros = 0;
                self.phase = Phase::Length;
            }
            Phase::Length => {
                self.unread = usize::from(self.byte.take()?);
                self.phase = if self.long {
                    Phase::LengthHigh
                } else {
                    self.unread += 1;
                    Phase::Symbols
                };
            }
            Phase::LengthHigh => {
                self.unread += usize::from(self.byte.take()?) * 256 + 257;
                self.phase = Phase::Symbols;
            }
            Phase::Symbols => {
                if self.state < L {
                    if self.unread > 0 {
                        if let Some(byte) = self.byte.take() {
                            self.state = (self.state << 8) | u32::from(byte);
                            self.unread -= 1;
                            if self.unread == 0 && self.flush {
                                self.pad();
                            }
                        }
                    } else if self.flush {
                        if self.state == 0 || self.zeros >= 8 {
                            return None;
                        }
                        self.pad();
                    } else {
                        self.phase = Phase::Header;
                        self.done = self.remaining == 0;
                        return Some(());
                    }
                }
                if self.state > L && self.remaining != 0 {
                    let symbol = self.model.inverse(self.state as u16);
                    if symbol >= 1 << (self.model.eab + 1) {
                        return None;
                    }
                    self.held = Some(symbol);
                    self.remaining -= 1;
                }
                if self.unread == 0 {
                    if self.flush {
                        if self.state == L {
                            self.done = self.remaining == 0;
                            if !self.done {
                                self.phase = Phase::Header;
                                self.state = 0;
                            }
                        } else if self.state > L && self.held.is_none() {
                            return None;
                        }
                    } else if self.remaining == 0 && self.held.is_none() {
                        self.done = true;
                    }
                } else if self.state == L
                    || (self.remaining == 0 && self.held.is_none() && self.state > L)
                {
                    return None;
                }
            }
        }
        Some(())
    }
}

/// Decode lane-major bytes with the Python reader's feeder and flush rules.
pub(crate) fn unpack(
    input: &[u8],
    widths: &[u8],
    count: usize,
    output: &mut [u8],
) -> Option<usize> {
    let mut lanes: Vec<Lane> = widths.iter().map(|_| Lane::new(count)).collect();
    let all = (1u32 << lanes.len()) - 1;
    let mut pos = 0;
    let mut slots = 0;
    let mut drained = 0;
    while drained != all || slots != 0 {
        let mut budget = FEED_WINDOW;
        for (i, (lane, target)) in lanes
            .iter_mut()
            .zip(output.chunks_exact_mut(count))
            .enumerate()
        {
            let bit = 1 << i;
            if lane.done || (lane.held.is_some() && slots & bit != 0) {
                continue;
            }
            if lane.byte.is_none() && (!matches!(lane.phase, Phase::Symbols) || lane.unread != 0) {
                if budget == 0 {
                    continue;
                }
                lane.byte = Some(*input.get(pos)?);
                pos += 1;
                budget -= 1;
            }
            if let Some(symbol) = lane.held.take() {
                if matches!(lane.phase, Phase::Symbols) {
                    lane.state = lane.model.advance(lane.state, symbol);
                }
                lane.emit(symbol, target);
                slots |= bit;
            }
            lane.step(widths[i])?;
            if lane.done && lane.held.is_none() {
                drained |= bit;
            }
        }
        if slots | drained == all {
            slots = 0;
        }
        for (i, (lane, target)) in lanes
            .iter_mut()
            .zip(output.chunks_exact_mut(count))
            .enumerate()
        {
            let bit = 1 << i;
            if lane.done && lane.held.is_some() && slots & bit == 0 {
                let symbol = lane.held.take().unwrap();
                lane.emit(symbol, target);
                slots |= bit;
                drained |= bit;
            }
        }
    }
    if input[pos..].iter().any(|&b| b != 0) {
        return None;
    }
    lanes
        .iter()
        .all(|lane| lane.written == count)
        .then_some(pos)
}

// The caller supplies valid, disjoint buffers, including count * lanes bytes of output.
#[no_mangle]
pub unsafe extern "C" fn unpack_lanes(
    input: *const u8,
    length: usize,
    widths: *const u8,
    lanes: usize,
    count: usize,
    output: *mut u8,
) -> isize {
    if !(1..=MAX_LANES).contains(&lanes) || count == 0 || length > isize::MAX as usize {
        return -1;
    }
    let Some(output_len) = count
        .checked_mul(lanes)
        .filter(|&n| n <= isize::MAX as usize)
    else {
        return -1;
    };
    let widths = slice::from_raw_parts(widths, lanes);
    if widths.iter().any(|&w| !(1..=8).contains(&w)) {
        return -1;
    }
    unpack(
        slice::from_raw_parts(input, length),
        widths,
        count,
        slice::from_raw_parts_mut(output, output_len),
    )
    .map_or(-1, |used| used as isize)
}
