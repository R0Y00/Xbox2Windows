//! Generate the reference bytes for the C++ emitter tests in a temporary
//! checkout of the upstream Rust emitter. The adjacent README.txt identifies
//! the crate, test invocation, and optimizer settings.

use super::bitcode::{Attr, Bin, Cast, Rmw, Term, pred};
use super::container::{self, Binding, Psv, SigElement};
use super::shader::{Mesh, Shader};

fn out_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("X2W_REF_OUT").expect("X2W_REF_OUT"))
}

fn put(name: &str, bytes: &[u8]) {
    let dir = out_dir();
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(format!("{name}.bin")), bytes).unwrap();
}

fn el(name: &str, index: u32, kind: u8, comp: u8, row: Option<u32>, col: u8, cols: u8, interp: u8, usage: u8) -> SigElement {
    SigElement { name: name.into(), index, kind, comp, interp, row, col, cols, usage, half: false }
}

/// Signature, PSV0, build and hash of fixed inputs.
fn containers() {
    let ins = vec![
        el("SV_Position", 0, 3, 9, Some(0), 0, 4, 4, 0xf),
        el("TEXCOORD", 0, 0, 9, Some(1), 0, 2, 2, 3),
        el("TEXCOORD", 1, 0, 5, Some(1), 2, 2, 1, 1),
        el("COLOR", 0, 0, 9, Some(2), 0, 4, 2, 0),
        el("SV_IsFrontFace", 0, 13, 5, Some(3), 0, 1, 1, 1),
        el("SV_SampleIndex", 0, 12, 5, None, 0, 1, 0, 1),
    ];
    let outs = vec![
        el("SV_Target", 0, 16, 9, Some(0), 0, 4, 0, 0xf),
        el("SV_Target", 1, 16, 5, Some(1), 0, 2, 0, 3),
        el("SV_Depth", 0, 17, 9, None, 0, 1, 0, 1),
    ];
    put("sig_in", &container::signature(&ins, true));
    put("sig_out", &container::signature(&outs, false));
    let bindings = [
        Binding { kind: 2, space: 0, register: 0, dxil_kind: 13 },
        Binding { kind: 1, space: 0, register: 3, dxil_kind: 14 },
        Binding { kind: 4, space: 1, register: 7, dxil_kind: 11 },
        Binding { kind: 7, space: 1, register: 9, dxil_kind: 11 },
    ];
    put(
        "psv_ps",
        &container::psv(&Psv {
            stage: 0,
            flag: true,
            inputs: &ins,
            outputs: &outs,
            prims: &[],
            mesh: None,
            geometry: None,
            bindings: &bindings,
            threads: [0; 3],
            wave_size: None,
        }),
    );
    let prims = vec![el("PRIMDATA", 0, 0, 5, Some(1), 0, 3, 1, 7)];
    put(
        "psv_ms",
        &container::psv(&Psv {
            stage: 13,
            flag: false,
            inputs: &[],
            outputs: &ins[..4],
            prims: &prims,
            mesh: Some((Mesh { max_vertices: 64, max_primitives: 126, topology: 2 }, 4096)),
            geometry: None,
            bindings: &[],
            threads: [128, 1, 1],
            wave_size: None,
        }),
    );
    put(
        "psv_cs",
        &container::psv(&Psv {
            stage: 5,
            flag: false,
            inputs: &[],
            outputs: &[],
            prims: &[],
            mesh: None,
            geometry: None,
            bindings: &bindings[2..],
            threads: [8, 8, 1],
            wave_size: Some(64),
        }),
    );
    put(
        "psv_gs",
        &container::psv(&Psv {
            stage: 2,
            flag: true,
            inputs: &ins[..3],
            outputs: &ins[..3],
            prims: &[],
            mesh: None,
            geometry: Some(6),
            bindings: &[],
            threads: [0; 3],
            wave_size: None,
        }),
    );
    // The container hash of every length 0..=200 (pattern bytes), concatenated.
    let mut hashes = Vec::new();
    for n in 0..=200usize {
        let data: Vec<u8> = (0..n).map(|i| (i * 7 + 3) as u8).collect();
        hashes.extend(container::hash(&data));
    }
    put("hashes", &hashes);
    let a = [1u8, 2, 3];
    let b = [9u8; 37];
    put("build", &container::build(&[(b"SFI0", &a), (b"XXXX", &b), (b"EMPT", &[])]));
}

/// A compute shader through the Shader API: user data, root views, LDS, a loop, a branch, wave
/// operations, a heap handle, atomics.
fn cs_basic() -> Vec<u8> {
    let mut s = Shader::new(false);
    s.threads = Some([64, 1, 1]);
    s.wave_size = Some(32);
    s.lds_dwords = 256;
    s.user_data = 6;
    let b = s.entry;
    let i32t = s.t.i32;
    let ud = s.user_data_handle();
    let zero = s.m.i32(0);
    let row = s.op(b, "cbufferLoadLegacy.i32", Attr::ReadOnly, s.t.cbuf_ret, 59, &[ud, zero]);
    let x = s.m.extract(b, row, 1);
    let tid = s.op(b, "threadId.i32", Attr::ReadNone, i32t, 93, &[zero]);
    let four = s.m.i32(4);
    let off = s.m.bin(b, Bin::Mul, tid, four);
    let srv = s.root_srv_handle(3);
    let undef = s.m.undef(i32t);
    let one8 = s.m.i8(1);
    let ld = s.op(b, "rawBufferLoad.f32", Attr::ReadOnly, s.t.res_ret, 139, &[srv, off, undef, one8, four]);
    let vf = s.m.extract(b, ld, 0);
    let v = s.m.cast(b, Cast::Bitcast, vf, i32t);
    let sum = s.m.bin(b, Bin::Add, v, x);
    // A dead value and a constant fold for the clean-up.
    let k = s.m.bin(b, Bin::Add, zero, four);
    let _dead = s.m.bin(b, Bin::Mul, sum, k);
    let lds = s.lds();
    let k255 = s.m.i32(255);
    let idx = s.m.bin(b, Bin::And, tid, k255);
    let p = s.m.element(b, lds, idx);
    s.m.store(b, p, sum);
    let p2 = s.m.element(b, lds, idx);
    let back = s.m.load(b, p2);
    let old = s.m.atomic(b, Rmw::Add, p2, back);
    let exch = s.m.cmpxchg(b, p2, old, back);
    // A loop: i from 0 while i < x.
    let (head, body, exit) = (s.m.new_block(), s.m.new_block(), s.m.new_block());
    s.m.terminate(b, Term::Br(head));
    let i = s.m.phi(head, i32t);
    let acc = s.m.phi(head, i32t);
    let c = s.m.cmp(head, pred::IULT, i, x);
    s.m.terminate(head, Term::CondBr(c, body, exit));
    let one = s.m.i32(1);
    let i2 = s.m.bin(body, Bin::Add, i, one);
    let zero8 = s.m.i8(0);
    let wsum = s.op(body, "waveActiveOp.i32", Attr::NoUnwind, i32t, 119, &[acc, zero8, one8]);
    let acc2 = s.m.bin(body, Bin::Xor, wsum, exch);
    s.m.terminate(body, Term::Br(head));
    s.m.set_phi(i, vec![(zero, b), (i2, body)]);
    s.m.set_phi(acc, vec![(sum, b), (acc2, body)]);
    s.flags |= super::shader::flags::WAVE_OPS;
    let two = s.m.i32(2);
    let heap_index = s.m.bin(exit, Bin::LShr, acc, two);
    let h = s.heap_handle(exit, heap_index, false, true, (11 | 0x1000, 0));
    let f = s.m.cast(exit, Cast::Bitcast, acc, s.t.f32);
    let fi = s.m.cast(exit, Cast::FpToUi, f, i32t);
    s.op(exit, "rawBufferStore.i32", Attr::NoUnwind, s.t.void, 140, &[h, off, undef, fi, undef, undef, undef, one8, four]);
    let uav = s.root_uav_handle(5);
    let sel = s.m.select(exit, c, fi, acc);
    s.op(exit, "rawBufferStore.i32", Attr::NoUnwind, s.t.void, 140, &[uav, off, undef, sel, undef, undef, undef, one8, four]);
    let cbv = s.root_cbv_handle(2);
    let crow = s.op(exit, "cbufferLoadLegacy.i32", Attr::ReadOnly, s.t.cbuf_ret, 59, &[cbv, zero]);
    let cx = s.m.extract(exit, crow, 2);
    s.op(exit, "rawBufferStore.i32", Attr::NoUnwind, s.t.void, 140, &[uav, cx, undef, cx, undef, undef, undef, one8, four]);
    s.m.terminate(exit, Term::Ret);
    s.finish()
}

/// A pixel shader: inputs read, outputs written, an immediate sampler and a heap texture.
fn ps_basic() -> Vec<u8> {
    let mut s = Shader::new(true);
    s.inputs = vec![
        el("SV_Position", 0, 3, 9, Some(0), 0, 4, 4, 0x3),
        el("TEXCOORD", 0, 0, 9, Some(1), 0, 2, 2, 3),
        el("TEXCOORD", 1, 0, 5, Some(3), 0, 1, 1, 0),
        el("COLOR", 0, 0, 9, Some(2), 0, 4, 2, 0),
    ];
    s.outputs = vec![el("SV_Target", 0, 16, 9, Some(0), 0, 4, 0, 0xf)];
    s.samplers = 2;
    s.cmp_samplers = 2;
    s.user_data = 1;
    let b = s.entry;
    let f32t = s.t.f32;
    let mut uv = Vec::new();
    for col in 0..2u8 {
        let ty = s.t.i32;
        let args = [s.m.i32(1), s.m.i32(0), s.m.i8(col), s.m.undef(ty)];
        uv.push(s.op(b, "loadInput.f32", Attr::ReadNone, f32t, 4, &args));
    }
    let ud = s.user_data_handle();
    let zero = s.m.i32(0);
    let row = s.op(b, "cbufferLoadLegacy.i32", Attr::ReadOnly, s.t.cbuf_ret, 59, &[ud, zero]);
    let slot = s.m.extract(b, row, 0);
    let tex = s.heap_handle(b, slot, false, false, (2 | (9 << 8), 0x0409));
    let smp = s.sampler_handle(b, 0);
    let undef = s.m.undef(f32t);
    let i32t = s.t.i32;
    let ui = s.m.undef(i32t);
    let r = s.op(
        b,
        "sample.f32",
        Attr::ReadOnly,
        s.t.res_ret,
        60,
        &[tex, smp, uv[0], uv[1], undef, undef, ui, ui, ui, undef],
    );
    let half = s.m.f32(0x3f00_0000);
    for col in 0..4u32 {
        let v = s.m.extract(b, r, col);
        let m = s.m.bin(b, Bin::Mul, v, half);
        let args = [s.m.i32(0), s.m.i32(0), s.m.i8(col as u8), m];
        s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
    }
    s.m.terminate(b, Term::Ret);
    s.finish()
}

fn rect_outputs() -> Vec<SigElement> {
    vec![
        el("SV_Position", 0, 3, 9, Some(0), 0, 4, 0, 0),
        el("TEXCOORD", 0, 0, 9, Some(1), 0, 4, 0, 0),
        el("TEXCOORD", 1, 0, 5, Some(2), 0, 2, 0, 0),
    ]
}

#[test]
fn ref_dump() {
    containers();
    put("cs_basic", &cs_basic());
    put("ps_basic", &ps_basic());
    let wanted = [el("SV_PrimitiveID", 0, 10, 5, Some(3), 0, 1, 0, 0)];
    put("rect_prim", &super::rect::build(rect_outputs(), &wanted).unwrap());
    put("rect_plain", &super::rect::build(rect_outputs(), &[]).unwrap());
    println!("ref_dump wrote {}", out_dir().display());
}
