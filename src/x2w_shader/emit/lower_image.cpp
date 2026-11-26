// Image instructions. See lower_impl.h and GMR/vs_port/lower/PORTING.md for the conventions.
#include <algorithm>
#include <bit>
#include <cstdlib>

#include "x2w_shader/emit/lower_impl.h"

namespace x2w::vs::lower {

void Lower::image(const Inst& inst, std::size_t b, std::size_t i) {
    const auto& img = std::get<gfx10::Image>(inst.fields);
    const Operand& data = img.data;
    const std::vector<Operand>& addr = img.addr;
    const bool unorm = img.unorm;
    const bool a16 = img.a16;
    const bool d16 = img.d16;
    if (img.tfe || img.lwe) {
        fail("an image access with tfe or lwe");
    }
    const std::string_view name = inst.name();
    // A load or sample with no channel enabled returns the first.
    const std::uint8_t dmask = img.dmask == 0 ? std::uint8_t{1} : img.dmask;
    const Site site{b, i};
    const std::optional<V> guard = this->guard(b, i);
    const std::size_t c = this->cur;
    std::uint32_t kind = 0;
    std::size_t coords = 0;
    switch (img.dim) {
        case 0: kind = 1; coords = 1; break;
        case 1: kind = 2; coords = 2; break;
        case 2: kind = 4; coords = 3; break;
        case 3:
            // Sampled (face coordinates and face, see `cube_direction`) or sized only.
            if (name.starts_with("image_sample") || name.starts_with("image_gather") ||
                name == "image_get_resinfo") {
                kind = 5;
                coords = 3;
                break;
            }
            fail("an image dimension (cube loads or MSAA)");
        case 4: kind = 6; coords = 2; break;
        case 5: kind = 7; coords = 3; break;
        default: fail("an image dimension (cube loads or MSAA)");
    }
    // Address VGPRs in order.
    std::vector<std::uint8_t> regs;
    for (const Operand& a : addr) {
        if (a.kind == Operand::Kind::Vector) {
            for (std::uint8_t k = 0; k < a.dwords; ++k) {
                regs.push_back(static_cast<std::uint8_t>(a.reg + k));
            }
        }
    }
    // Full 32-bit address VGPR at `reg_idx` (offset, bias, z-compare).
    auto addr_u32 = [&](Lower& l, std::size_t reg_idx) -> V {
        if (reg_idx >= regs.size()) fail("an image address");
        const std::uint8_t r = regs[reg_idx];
        const Value v = ok_or(inst.read(Reg::V(r)), "an image address");
        return l.i32_of(v);
    };
    // With a16, two 16-bit components per VGPR starting at `base_reg`, low half first.
    auto addr_packed = [&](Lower& l, std::size_t base_reg, std::size_t k) -> V {
        const V x = addr_u32(l, base_reg + (a16 ? k / 2 : k));
        if (!a16) {
            return x;
        }
        const std::size_t c = l.cur;
        const V sh = l.s.m.i32(16 * (static_cast<std::uint32_t>(k) % 2));
        const V mask = l.s.m.i32(0xFFFF);
        const V y = l.s.m.bin(c, Bin::LShr, x, sh);
        return l.s.m.bin(c, Bin::And, y, mask);
    };
    auto addr_val = [&](Lower& l, std::size_t k) -> V { return addr_packed(l, 0, k); };
    const Source* src = nullptr;
    {
        auto it = this->descriptors.find(std::pair<Site, bool>{site, false});
        if (it == this->descriptors.end()) fail("an untracked image");
        src = it->second;
    }
    const bool store = name.starts_with("image_store");
    bool uav = store;
    if (!uav) {
        const UsedResource* u = this->used_kind(*src);
        uav = u != nullptr && u->kind == ResourceType::Uav();
    }
    // A UAV inside an ordered section is a rasterizer ordered view.
    const bool ordered = uav && this->ordered.contains(site);
    if (ordered) {
        this->s.flags |= flags::ROVS;
    }
    const std::uint32_t rov = ordered ? 0x2000 : 0;
    const std::pair<std::uint32_t, std::uint32_t> props{kind | (uav ? 0x1000u : 0u) | rov, 0x409};
    if (name.starts_with("image_atomic_")) {
        const std::string_view op = name.substr(std::string_view("image_atomic_").size());
        // GLC returns the old value. DXIL implements subtraction by adding the two's complement
        // of the operand at the same width.
        std::uint32_t code = 0;
        if (op == "add" || op == "sub") code = 0;
        else if (op == "and") code = 1;
        else if (op == "or") code = 2;
        else if (op == "xor") code = 3;
        else if (op == "smin") code = 4;
        else if (op == "smax") code = 5;
        else if (op == "umin") code = 6;
        else if (op == "umax") code = 7;
        else if (op == "swap") code = 8;
        else if (op == "cmpswap") code = std::numeric_limits<std::uint32_t>::max();
        else fail(name.data());
        const gfx10::Cache& cache = img.cache;
        // 16-bit coordinates (a16) are read as `addr_val` reads them.
        if (d16) {
            fail("a 16-bit image atomic");
        }
        // Compare-exchange consumes a replacement followed by a comparand.
        const bool compare = code == std::numeric_limits<std::uint32_t>::max();
        bool wide = false;
        if ((!compare && dmask == 1) || (compare && dmask == 3)) {
            wide = false;
        } else if ((!compare && dmask == 3) || (compare && dmask == 15)) {
            wide = true;
        } else {
            fail("an image atomic data mask");
        }
        const V iu = this->s.m.undef(this->s.t.i32);
        std::vector<V> cs(3, iu);
        for (std::size_t k = 0; k < cs.size() && k < coords; ++k) {
            cs[k] = addr_val(*this, k);
        }
        if (data.kind != Operand::Kind::Vector) {
            fail("image atomic data");
        }
        const std::uint8_t reg = data.reg;
        const std::uint8_t width = wide ? 2 : 1;
        const std::vector<V> vals =
            this->vgprs(inst, data, static_cast<std::uint8_t>(compare ? 2 * width : width));
        Ty ty{};
        std::uint32_t component = 0;
        if (wide) {
            this->s.flags |= flags::INT64 | flags::ATOMIC_TYPED64 | flags::ATOMIC_HEAP64;
            ty = this->s.m.int_(64);  // one uint64_t in an R32G32_UINT view
            component = 0x107;
        } else {
            ty = this->s.t.i32;
            component = 0x105;
        }
        const V original = wide ? this->join64(vals.at(0), vals.at(1)) : vals.at(0);
        V value = original;
        if (op == "sub") {
            const V zero = this->s.m.int_const(ty, 0);
            value = this->s.m.bin(c, Bin::Sub, zero, original);
        }
        const V h = this->handle(site, *src, false, {kind | 0x1000 | rov, component});
        std::vector<V> args{h};
        const char* intrinsic = nullptr;
        std::uint32_t opcode = 0;
        if (compare) {
            const V comparand = wide ? this->join64(vals.at(2), vals.at(3)) : vals.at(1);
            args.insert(args.end(), cs.begin(), cs.end());
            args.push_back(comparand);
            args.push_back(value);
            intrinsic = wide ? "atomicCompareExchange.i64" : "atomicCompareExchange.i32";
            opcode = 79;
        } else {
            const V k = this->s.m.i32(code);
            args.push_back(k);
            args.insert(args.end(), cs.begin(), cs.end());
            args.push_back(value);
            intrinsic = wide ? "atomicBinOp.i64" : "atomicBinOp.i32";
            opcode = 78;
        }
        const auto at = this->begin_guarded(guard);
        const V r = this->s.op(this->cur, intrinsic, Attr::NoUnwind, ty, opcode, args);
        if (cache.glc) {
            const std::vector<std::pair<V, V>> pairs{{r, original}};
            const V old = this->end_guarded(at, pairs).at(0);
            if (wide) {
                const auto [lo, hi] = this->split64(old);
                this->write(inst, Reg::V(reg), lo, std::nullopt);
                this->write(inst, Reg::V(static_cast<std::uint8_t>(reg + 1)), hi, std::nullopt);
            } else {
                this->write(inst, Reg::V(reg), old, std::nullopt);
            }
        } else {
            this->end_guarded(at, {});
        }
        return;
    }
    if (store) {
        if (name != "image_store") {
            fail("an image store to a mip level");
        }
        // shortcut: stores are taken to write all four channels of a float format (the console
        // converts by the view's format). One channel goes into a view the program also counts in.
        const bool counter = dmask == 1 && this->is_counter(*src);
        if (dmask != 0xF && !counter) {
            fail("an image store of some channels");
        }
        const V iu = this->s.m.undef(this->s.t.i32);
        std::vector<V> cs(3, iu);
        for (std::size_t k = 0; k < cs.size() && k < coords; ++k) {
            cs[k] = addr_val(*this, k);
        }
        if (data.kind != Operand::Kind::Vector) {
            fail("image store data");
        }
        const std::uint8_t reg = data.reg;
        if (counter) {
            const Value v = ok_or(inst.read(Reg::V(reg)), "image store data");
            const V x = this->i32_of(v);
            // A 16-bit integer, extended by the view's sign: the same number either way only if it
            // is known to be small.
            if (d16) {
                const auto kx = this->s.m.int_k(x);
                if (!(kx && kx->second < 0x8000)) {
                    fail("a 16-bit image store of one channel");
                }
            }
            const V h = this->handle(site, *src, false, {kind | 0x1000 | rov, 0x105});
            const auto at = this->begin_guarded(guard);
            // All four channels are written (the validator wants them all): the view has the first
            // alone.
            std::vector<V> args{h};
            args.insert(args.end(), cs.begin(), cs.end());
            for (int n = 0; n < 4; ++n) args.push_back(x);
            const V mask = this->s.m.i8(0xF);
            args.push_back(mask);
            const Ty void_ = this->s.t.void_;
            this->s.op(this->cur, "textureStore.i32", Attr::NoUnwind, void_, 67, args);
            this->end_guarded(at, {});
            return;
        }
        std::vector<V> vals;
        for (std::uint8_t k = 0; k < 4; ++k) {
            // d16: two f16 channels per VGPR, low half first.
            const std::uint8_t r = d16 ? k / 2 : k;
            const Value v =
                ok_or(inst.read(Reg::V(static_cast<std::uint8_t>(reg + r))), "image store data");
            if (d16) {
                V x = this->i32_of(v);
                if (k % 2 == 1) {
                    const V k16 = this->s.m.i32(16);
                    x = this->s.m.bin(this->cur, Bin::LShr, x, k16);
                }
                const Ty f32t = this->s.t.f32;
                vals.push_back(this->s.op(this->cur, "legacyF16ToF32", Attr::ReadNone, f32t, 131,
                                          std::vector<V>{x}));
            } else {
                vals.push_back(this->float_of(v));
            }
        }
        const V h = this->handle(site, *src, false, props);
        const auto at = this->begin_guarded(guard);
        std::vector<V> args{h};
        args.insert(args.end(), cs.begin(), cs.end());
        args.insert(args.end(), vals.begin(), vals.end());
        const V mask = this->s.m.i8(0xF);
        args.push_back(mask);
        const Ty void_ = this->s.t.void_;
        this->s.op(this->cur, "textureStore.f32", Attr::NoUnwind, void_, 67, args);
        this->end_guarded(at, {});
        return;
    }
    const bool has_cl = name.ends_with("_cl");
    std::uint32_t op = 0;
    bool sample = false;
    if (name == "image_sample" || name == "image_sample_o" || name == "image_sample_cl") {
        op = 60; sample = true;
    } else if (name == "image_sample_lz" || name == "image_sample_l" || name == "image_sample_lz_o") {
        op = 62; sample = true;
    } else if (name == "image_sample_b" || name == "image_sample_b_cl") {
        op = 61; sample = true;
    } else if (name == "image_sample_c_lz") {
        op = 65; sample = true;
    } else if (name == "image_sample_d" || name == "image_sample_d_cl") {
        op = 63; sample = true;
    } else if (name == "image_get_lod" && this->s.pixel) {
        op = 81; sample = true;
    } else if (name == "image_sample_c" && this->s.pixel) {
        // Implicit derivatives: pixel shaders only.
        op = 64; sample = true;
    } else if (name == "image_sample_c_cl" && this->s.pixel) {
        op = 64; sample = true;
    } else if (name == "image_gather4" || name == "image_gather4_lz" || name == "image_gather4_lz_o" ||
               name == "image_gather4_l" || name == "image_gather4_l_o") {
        op = 73; sample = true;
    } else if (name == "image_load" || name == "image_load_mip") {
        op = 66; sample = false;
    } else if (name == "image_get_resinfo") {
        op = 72; sample = false;
    } else {
        fail(name.data());
    }
    if (op == 72) {
        // Width, height, depth or layers, levels (per dmask) of the level in the first address:
        // DXIL's dimensions; a channel DXIL leaves out reads 1 (UAVs: one level).
        const V h = this->handle(site, *src, false, props);
        const V mip = uav ? this->s.m.undef(this->s.t.i32) : addr_val(*this, 0);
        const Ty i32t = this->s.t.i32;
        const Ty dims = this->s.m.named_struct("dx.types.Dimensions", std::vector<Ty>(4, i32t));
        const V d = this->s.op(c, "getDimensions", Attr::ReadOnly, dims, 72, std::vector<V>{h, mip});
        const V one = this->s.m.i32(1);
        std::array<bool, 4> present{true, true, true, true};
        if (kind == 1) present = {true, false, false, true};
        else if (kind == 2 || kind == 5 || kind == 6) present = {true, true, false, true};
        if (data.kind != Operand::Kind::Vector) {
            fail("an image result");
        }
        const std::uint8_t reg = data.reg;
        std::size_t k = 0;
        for (std::uint32_t ch = 0; ch < 4; ++ch) {
            if (((dmask >> ch) & 1) == 0) continue;
            const V x = (ch == 3 && uav) || !present[ch] ? one : this->s.m.extract(c, d, ch);
            this->write(inst, Reg::V(static_cast<std::uint8_t>(reg + k)), x, guard);
            ++k;
        }
        return;
    }
    if (uav && op != 66) {
        fail("sampling a UAV");
    }
    const V h = this->handle(site, *src, false, props);
    const V fu = this->s.m.undef(this->s.t.f32);
    const V iu = this->s.m.undef(this->s.t.i32);
    // A gather at a named level, done as four texel reads (below).
    std::optional<std::array<V, 4>> gathered;
    V r{};
    if (sample) {
        const Source* ss = nullptr;
        {
            auto it = this->descriptors.find(std::pair<Site, bool>{site, true});
            if (it == this->descriptors.end()) fail("an untracked sampler");
            ss = it->second;
        }
        // Comparison: the reference value is the first address after the bias, and the sampler is
        // a comparison sampler.
        const bool cmp = op == 64 || op == 65;
        if (cmp && (a16 || std::popcount(dmask) != 1)) {
            fail("a comparison sample of 16-bit coordinates or several channels");
        }
        std::optional<std::pair<V, std::vector<std::uint32_t>>> sampler_choice;
        V sh{};
        if (const Source* phi = ss->kind == Source::Kind::Phi ? ss : nullptr) {
            std::optional<std::pair<std::optional<V>, std::vector<std::uint32_t>>> choice;
            try {
                choice = this->sampler_phi_choice(phi->block, phi->args, cmp, 8);
            } catch (const WhyFail&) {
                choice.reset();
            }
            if (choice && !choice->first) {
                sh = this->s.sampler_handle(this->cur, choice->second.at(0));
            } else if (choice) {
                const V first = this->s.sampler_handle(this->cur, choice->second.at(0));
                sampler_choice = std::pair<V, std::vector<std::uint32_t>>{*choice->first,
                                                                          std::move(choice->second)};
                sh = first;
            } else {
                sh = this->handle(site, *ss, true, {14u | (cmp ? 0x8000u : 0u), 0});
            }
        } else {
            sh = this->handle(site, *ss, true, {14u | (cmp ? 0x8000u : 0u), 0});
        }
        // Texel offsets (`_o`) come first: x in bits 0-5, y in 8-13, signed (programmable offsets:
        // gathers only).
        const bool offset_word = name.ends_with("_o");
        const std::size_t first = std::size_t(offset_word) +
                                  std::size_t(name.starts_with("image_sample_b")) + std::size_t(cmp);
        // Coordinates the size divides (not the array slice); explicit gradients (x then y) come
        // before the coordinates.
        const std::size_t spatial = coords - std::size_t(kind == 6 || kind == 7);
        // A cube's gradients are the two of its face coordinates.
        const std::size_t grads = kind == 5 ? 2 : spatial;
        const std::size_t deriv_regs = op == 63 ? (a16 ? div_ceil<std::size_t>(grads, 2) : grads) : 0;
        const std::size_t coord_base = first + 2 * deriv_regs;
        std::vector<V> cs(4, fu);
        // With a16, sampling coordinates, gradients, LOD and clamp are f16.
        auto to_float = [&](Lower& l, V x) -> V {
            if (a16) {
                const Ty f = l.s.t.f32;
                return l.s.op(l.cur, "legacyF16ToF32", Attr::ReadNone, f, 131, std::vector<V>{x});
            }
            return l.as_float(x);
        };
        if (kind == 5 && (unorm || op == 81)) {
            fail("a cube sample with texel coordinates or a level query");
        }
        for (std::size_t k = 0; k < cs.size() && k < coords; ++k) {
            const V x = addr_packed(*this, coord_base, k);
            cs[k] = to_float(*this, x);
        }
        if (kind == 5) {
            const std::array<V, 3> dir = this->cube_direction(cs[0], cs[1], cs[2]);
            cs[0] = dir[0];
            cs[1] = dir[1];
            cs[2] = dir[2];
        }
        if (unorm) {
            // Texel coordinates: divided by the size of level 0.
            const Ty i32t = this->s.t.i32;
            const Ty dims = this->s.m.named_struct("dx.types.Dimensions", std::vector<Ty>(4, i32t));
            const V z = this->s.m.i32(0);
            const V d = this->s.op(c, "getDimensions", Attr::ReadOnly, dims, 72, std::vector<V>{h, z});
            for (std::size_t k = 0; k < cs.size() && k < spatial; ++k) {
                const V n0 = this->s.m.extract(c, d, static_cast<std::uint32_t>(k));
                const V n = this->s.m.cast(c, Cast::UiToFp, n0, this->s.t.f32);
                cs[k] = this->s.m.bin(c, Bin::SDiv, cs[k], n);  // fdiv: same opcode as sdiv
            }
        }
        if (op == 81) {
            // The clamped then the unclamped level of detail.
            if (data.kind != Operand::Kind::Vector) {
                fail("an image result");
            }
            const std::uint8_t reg = data.reg;
            const Ty f32t = this->s.t.f32;
            std::size_t k = 0;
            for (std::uint32_t ch = 0; ch < 2; ++ch) {
                if (((dmask >> ch) & 1) == 0) continue;
                const V clamped = this->s.m.bool_(ch == 0);
                const V lod = this->s.op(c, "calculateLOD.f32", Attr::ReadOnly, f32t, 81,
                                         std::vector<V>{h, sh, cs[0], cs[1], cs[2], clamped});
                const V x = this->as_int(lod);
                this->write(inst, Reg::V(static_cast<std::uint8_t>(reg + k)), x, guard);
                ++k;
            }
            return;
        }
        const std::size_t offsets =
            kind == 5 ? 0 : std::min<std::size_t>(coords - std::size_t(kind == 6 || kind == 7), 3);
        const V zero = this->s.m.i32(0);
        std::vector<V> os(3, iu);
        for (std::size_t k = 0; k < offsets; ++k) {
            os[k] = zero;
        }
        if (offset_word && offsets > 0) {
            const V w = addr_val(*this, 0);
            const std::pair<std::size_t, std::uint32_t> fields[2] = {{0, 26}, {1, 18}};
            for (std::size_t n = 0; n < std::min<std::size_t>(offsets, 2); ++n) {
                const auto [k, at] = fields[n];
                // Bits at..at+5 sign-extended: shift up, then down.
                const V left = this->s.m.i32(at);
                const V t = this->s.m.bin(c, Bin::Shl, w, left);
                const V k26 = this->s.m.i32(26);
                os[k] = this->s.m.bin(c, Bin::AShr, t, k26);
            }
        }
        std::vector<V> args{h, sh};
        args.insert(args.end(), cs.begin(), cs.end());
        V clamp = fu;
        if (has_cl) {
            const V x = addr_packed(*this, coord_base, coords);
            clamp = to_float(*this, x);
        }
        auto call_sample = [&](Lower& l, const char* intrinsic, std::uint32_t opcode,
                               const std::vector<V>& args,
                               std::optional<std::array<V, 4>>& gathered) -> V {
            if (!sampler_choice) {
                return l.s.op(c, intrinsic, Attr::ReadOnly, l.s.t.res_ret, opcode, args);
            }
            const V idx = sampler_choice->first;
            const std::vector<std::uint32_t>& regs = sampler_choice->second;
            const std::size_t merge = l.s.m.new_block();
            std::size_t cur_test = c;
            std::array<std::vector<std::pair<V, std::size_t>>, 4> per_ch;
            V last_r = args.at(0);
            for (std::size_t i = 0; i < regs.size(); ++i) {
                const std::uint32_t reg = regs[i];
                std::size_t arm = 0;
                if (i + 1 == regs.size()) {
                    arm = cur_test;
                } else {
                    arm = l.s.m.new_block();
                    const std::size_t next = l.s.m.new_block();
                    const V k = l.s.m.i32(static_cast<std::uint32_t>(i));
                    const V cond = l.s.m.cmp(cur_test, pred::EQ, idx, k);
                    l.s.m.terminate(cur_test, Term::cond_br(cond, arm, next));
                    cur_test = next;
                }
                std::vector<V> arm_args = args;
                arm_args.at(1) = l.s.sampler_handle(arm, reg);
                const V r_i = l.s.op(arm, intrinsic, Attr::ReadOnly, l.s.t.res_ret, opcode, arm_args);
                for (std::size_t ch = 0; ch < 4; ++ch) {
                    const V v = l.s.m.extract(arm, r_i, static_cast<std::uint32_t>(ch));
                    per_ch[ch].push_back({v, arm});
                }
                l.s.m.terminate(arm, Term::br(merge));
                last_r = r_i;
            }
            std::array<V, 4> phis{fu, fu, fu, fu};
            for (std::size_t ch = 0; ch < 4; ++ch) {
                const V phi = l.s.m.phi(merge, l.s.t.f32);
                l.s.m.set_phi(phi, std::move(per_ch[ch]));
                phis[ch] = phi;
            }
            gathered = phis;
            l.cur = merge;
            return last_r;
        };
        if (op == 60 && !this->s.pixel) {
            // Outside a pixel shader there are no screen derivatives: the validator refuses
            // `sample` there ("features incompatible with the shader stage": 730 compute programs,
            // among them the G-buffer resolves and the sky), and what a driver does with it is its
            // own affair.
            // shortcut: such a sample reads level 0 (a bias: that level). The console takes
            // derivatives from neighbouring lanes there; compute them from the lanes if a resolve
            // shows the wrong mip.
            args.insert(args.end(), os.begin(), os.end());
            const V z = this->s.m.f32(0);
            args.push_back(z);
            r = call_sample(*this, "sampleLevel.f32", 62, args, gathered);
        } else if (op == 60) {
            if (offset_word) {
                // DXIL immediate offsets must be in [-8, 7]. Console offsets are six bits and may be computed. Emit
                // immediates for in-range constants and use the computed offsets otherwise.
                const V w0 = addr_val(*this, 0);
                if (const auto wv = this->s.m.int_value(w0)) {
                    const std::uint32_t w = static_cast<std::uint32_t>(*wv);
                    auto field = [&](std::size_t k) -> std::int32_t {
                        const std::int32_t x = static_cast<std::int32_t>((w >> (8 * k)) & 0x3F);
                        return x >= 32 ? x - 64 : x;
                    };
                    const std::size_t n = std::min<std::size_t>(offsets, 2);
                    bool all = true;
                    for (std::size_t k = 0; k < n; ++k) {
                        const std::int32_t f = field(k);
                        if (!(f >= -8 && f <= 7)) {
                            all = false;
                            break;
                        }
                    }
                    if (all) {
                        for (std::size_t k = 0; k < os.size() && k < n; ++k) {
                            os[k] = this->s.m.i32(static_cast<std::uint32_t>(field(k)));
                        }
                    }
                }
            }
            args.insert(args.end(), os.begin(), os.end());
            args.push_back(clamp);
            r = call_sample(*this, "sample.f32", 60, args, gathered);
        } else if (op == 61) {
            // With a16 the bias is an f16 in the low half of its word (the word is whole, the value
            // is not).
            const V bias0 = addr_u32(*this, std::size_t(offset_word));
            V bias{};
            if (a16) {
                const V low = this->s.m.i32(0xFFFF);
                const V b16 = this->s.m.bin(c, Bin::And, bias0, low);
                bias = to_float(*this, b16);
            } else {
                bias = this->as_float(bias0);
            }
            args.insert(args.end(), os.begin(), os.end());
            if (this->s.pixel) {
                args.push_back(bias);
                args.push_back(clamp);
                r = call_sample(*this, "sampleBias.f32", 61, args, gathered);
            } else {
                // As above: the bias is the level.
                args.push_back(bias);
                r = call_sample(*this, "sampleLevel.f32", 62, args, gathered);
            }
        } else if (op == 63 && kind == 5) {
            // A cube with explicit gradients: the host wants them as changes of the direction, the
            // console gives the changes of the face coordinates (the face is one unit wide). The
            // level they select is taken from the longer of the two, in texels, and sampled
            // directly.
            // shortcut: no anisotropic filtering for these samples (the longer axis alone decides
            // the level).
            std::array<V, 2> len{fu, fu};
            for (std::size_t axis = 0; axis < 2; ++axis) {
                V sum = this->s.m.f32(0);
                for (std::size_t k = 0; k < 2; ++k) {
                    const V x0 = addr_packed(*this, first + axis * deriv_regs, k);
                    const V x = to_float(*this, x0);
                    const V sq = this->s.m.bin(c, Bin::Mul, x, x);
                    sum = this->s.m.bin(c, Bin::Add, sum, sq);
                }
                len[axis] = this->unary(24, sum);  // square root
            }
            const V longer = this->binary(35, len[0], len[1]);  // maximum
            const Ty i32t = this->s.t.i32;
            const Ty dims = this->s.m.named_struct("dx.types.Dimensions", std::vector<Ty>(4, i32t));
            const V z = this->s.m.i32(0);
            const V d = this->s.op(c, "getDimensions", Attr::ReadOnly, dims, 72, std::vector<V>{h, z});
            const V width0 = this->s.m.extract(c, d, 0);
            const V width = this->s.m.cast(c, Cast::UiToFp, width0, this->s.t.f32);
            const V texels = this->s.m.bin(c, Bin::Mul, longer, width);
            // DXIL uses a base-2 logarithm. A length of zero gives minus infinity, which the sampler clamps to level
            // zero. Apply the instruction's minimum level here because SampleLevel takes no clamp.
            V lod = this->unary(23, texels);
            if (has_cl) {
                lod = this->binary(35, lod, clamp);
            }
            args.insert(args.end(), os.begin(), os.end());
            args.push_back(lod);
            r = call_sample(*this, "sampleLevel.f32", 62, args, gathered);
        } else if (op == 63) {
            std::vector<V> g(6, fu);
            for (std::size_t k = 0; k < spatial; ++k) {
                const V x = addr_packed(*this, first, k);
                g.at(k) = to_float(*this, x);
                const V y = addr_packed(*this, first + deriv_regs, k);
                g.at(3 + k) = to_float(*this, y);
            }
            args.insert(args.end(), os.begin(), os.end());
            args.insert(args.end(), g.begin(), g.end());
            args.push_back(clamp);
            r = call_sample(*this, "sampleGrad.f32", 63, args, gathered);
        } else if (op == 64 || op == 65) {
            const V reference0 = addr_u32(*this, first - 1);
            const V reference = this->as_float(reference0);
            args.insert(args.end(), os.begin(), os.end());
            args.push_back(reference);
            const char* intrinsic = nullptr;
            if (op == 64) {
                args.push_back(clamp);
                intrinsic = "sampleCmp.f32";
            } else {
                intrinsic = "sampleCmpLevelZero.f32";
            }
            r = call_sample(*this, intrinsic, op, args, gathered);
        } else if (op == 62) {
            V lod{};
            if (name == "image_sample_l") {
                const V x = addr_packed(*this, coord_base, coords);
                lod = to_float(*this, x);
            } else {
                lod = this->s.m.f32(0);
            }
            args.insert(args.end(), os.begin(), os.end());
            args.push_back(lod);
            r = call_sample(*this, "sampleLevel.f32", 62, args, gathered);
        } else {
            if (name == "image_gather4") {
                fail("a gather with implicit level");
            }
            if (std::popcount(dmask) != 1) {
                fail("a gather of several channels");
            }
            if ((name == "image_gather4_l" || name == "image_gather4_l_o") && coords == 2 &&
                spatial == 2) {
                // The host's gather has no level: it read level 0 where the title names one (the
                // depth pyramid its occlusion tests read: most of the world was culled against the
                // wrong depths). The four texels are read at the level instead, in the gather's
                // order: (u0,v1) (u1,v1) (u1,v0) (u0,v0).
                // With an offset too (`image_gather4_l_o`): the offset is in texels of that level.
                // The title's meshlet culling takes a 4x4 block of the pyramid as one gather and
                // three offset ones ((2,0), (0,2), (2,2)); the three read level 0, two texels
                // beside the first one's place, so a meshlet three or four texels wide was tested
                // against a quarter of what it covers.
                // shortcut: the level is rounded to a whole one and coordinates past the edge are
                // clamped, whatever the sampler says.
                const V lod0 = addr_packed(*this, coord_base, coords);
                V lod = to_float(*this, lod0);
                const Ty i32t = this->s.t.i32;
                const Ty f32t = this->s.t.f32;
                const V fzero = this->s.m.f32(0);
                lod = this->binary(35, lod, fzero);  // maximum
                lod = this->unary(26, lod);          // round to nearest
                const V mip = this->s.m.cast(c, Cast::FpToUi, lod, i32t);
                const Ty dims = this->s.m.named_struct("dx.types.Dimensions", std::vector<Ty>(4, i32t));
                const V d = this->s.op(c, "getDimensions", Attr::ReadOnly, dims, 72,
                                       std::vector<V>{h, mip});
                const V half = this->s.m.f32(f32_bits(0.5f));
                const V one = this->s.m.f32(f32_bits(1.0f));
                std::array<V, 2> lo{iu, iu};
                std::array<V, 2> hi{iu, iu};
                for (std::size_t k = 0; k < 2; ++k) {
                    const V n0 = this->s.m.extract(c, d, static_cast<std::uint32_t>(k));
                    const V n = this->s.m.cast(c, Cast::UiToFp, n0, f32t);
                    const V last = this->s.m.bin(c, Bin::Sub, n, one);
                    const V t0 = this->s.m.bin(c, Bin::Mul, cs[k], n);
                    const V t = this->s.m.bin(c, Bin::Sub, t0, half);
                    const V first0 = this->unary(27, t);  // round down
                    // The offset word's field (0 without one).
                    const V by = this->s.m.cast(c, Cast::SiToFp, os[k], f32t);
                    const V firstk = this->s.m.bin(c, Bin::Add, first0, by);
                    const V next = this->s.m.bin(c, Bin::Add, firstk, one);
                    const std::pair<V*, V> pairs[2] = {{&lo[k], firstk}, {&hi[k], next}};
                    for (const auto& [slot, x0] : pairs) {
                        const V x1 = this->binary(35, x0, fzero);  // maximum
                        const V x = this->binary(36, x1, last);    // minimum
                        *slot = this->s.m.cast(c, Cast::FpToSi, x, i32t);
                    }
                }
                const std::uint32_t ch = static_cast<std::uint32_t>(std::countr_zero(dmask));
                std::array<V, 4> g{fu, fu, fu, fu};
                std::optional<V> read;
                const std::array<std::pair<V, V>, 4> texels{
                    std::pair<V, V>{lo[0], hi[1]}, {hi[0], hi[1]}, {hi[0], lo[1]}, {lo[0], lo[1]}};
                for (std::size_t n = 0; n < 4; ++n) {
                    const auto [x, y] = texels[n];
                    const V rr = this->s.op(c, "textureLoad.f32", Attr::ReadOnly, this->s.t.res_ret, 66,
                                            std::vector<V>{h, mip, x, y, iu, iu, iu, iu});
                    g[n] = this->s.m.extract(c, rr, ch);
                    read = rr;
                }
                gathered = g;
                r = *read;  // "four reads"
            } else {
                args.push_back(os[0]);
                args.push_back(os[1]);
                const V chv = this->s.m.i32(static_cast<std::uint32_t>(std::countr_zero(dmask)));
                args.push_back(chv);
                r = call_sample(*this, "textureGather.f32", 73, args, gathered);
            }
        }
    } else {
        std::vector<V> cs(3, iu);
        for (std::size_t k = 0; k < cs.size() && k < coords; ++k) {
            cs[k] = addr_val(*this, k);
        }
        V mip{};
        if (name == "image_load_mip") {
            mip = addr_val(*this, coords);
        } else if (uav) {
            // Our declarations have four channels.
            this->s.flags |= flags::TYPED_UAV_LOADS;
            mip = iu;
        } else {
            mip = this->s.m.i32(0);
        }
        std::vector<V> args{h, mip};
        args.insert(args.end(), cs.begin(), cs.end());
        args.push_back(iu);
        args.push_back(iu);
        args.push_back(iu);
        r = this->s.op(c, "textureLoad.f32", Attr::ReadOnly, this->s.t.res_ret, 66, args);
    }
    if (data.kind != Operand::Kind::Vector) {
        fail("an image result");
    }
    const std::uint8_t reg = data.reg;
    std::vector<std::uint32_t> channels;
    if (op == 73) {
        channels = {0, 1, 2, 3};
    } else {
        for (std::uint32_t ch = 0; ch < 4; ++ch) {
            if (((dmask >> ch) & 1) != 0) channels.push_back(ch);
        }
    }
    std::vector<V> xs;
    for (const std::uint32_t ch : channels) {
        const V x = gathered ? (*gathered)[ch] : this->s.m.extract(c, r, ch);
        if (d16) {
            // shortcut: d16 results are taken as floats (rounded to nearest even); integer formats
            // would keep 16 bits.
            const Ty i32t = this->s.t.i32;
            xs.push_back(this->s.op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, std::vector<V>{x}));
        } else {
            xs.push_back(this->as_int(x));
        }
    }
    if (d16) {
        // Two channels per VGPR, low half first (an odd last high half 0).
        const V k16 = this->s.m.i32(16);
        std::vector<V> packed;
        for (std::size_t n = 0; n < xs.size(); n += 2) {
            if (n + 1 < xs.size()) {
                const V hi = this->s.m.bin(c, Bin::Shl, xs[n + 1], k16);
                packed.push_back(this->s.m.bin(c, Bin::Or, xs[n], hi));
            } else {
                packed.push_back(xs[n]);
            }
        }
        xs = std::move(packed);
    }
    for (std::size_t k = 0; k < xs.size(); ++k) {
        this->write(inst, Reg::V(static_cast<std::uint8_t>(reg + k)), xs[k], guard);
    }
}

// The direction a cube sample's face coordinates stand for: the console takes s and t in [1, 2]
// (`v_cubesc`/`v_cubetc` over `|v_cubema|`, plus 1.5) and the face id (`v_cubeid`), which the
// host's cube sampling turns back into the same texel.
// shortcut: the view is taken as a cube, not a cube array (whose face operand adds 8 x the slice).
std::array<V, 3> Lower::cube_direction(V s, V t, V face) {
    const std::size_t c = this->cur;
    auto k = [](Lower& l, float v) { return l.s.m.f32(f32_bits(v)); };
    const V two = k(*this, 2.0f);
    const V three = k(*this, 3.0f);
    const V u0 = this->s.m.bin(c, Bin::Mul, s, two);
    const V u = this->s.m.bin(c, Bin::Sub, u0, three);
    const V v0 = this->s.m.bin(c, Bin::Mul, t, two);
    const V v = this->s.m.bin(c, Bin::Sub, v0, three);
    const V z = this->s.m.f32(0x8000'0000u);
    const V nu = this->s.m.bin(c, Bin::Sub, z, u);
    const V nv = this->s.m.bin(c, Bin::Sub, z, v);
    const V one = k(*this, 1.0f);
    const V mone = k(*this, -1.0f);
    const V f = this->s.m.cast(c, Cast::FpToUi, face, this->s.t.i32);
    // Per face +x, -x, +y, -y, +z, -z.
    const std::array<std::array<V, 6>, 3> table{{
        {one, mone, u, u, u, nu},
        {nv, nv, one, mone, nv, nv},
        {nu, u, v, nv, one, mone},
    }};
    std::array<V, 3> out{};
    for (std::size_t row = 0; row < 3; ++row) {
        V r = table[row][0];
        for (std::size_t n = 1; n < 6; ++n) {
            const V kn = this->s.m.i32(static_cast<std::uint32_t>(n));
            const V is = this->s.m.cmp(c, pred::EQ, f, kn);
            r = this->s.m.select(c, is, table[row][n], r);
        }
        out[row] = r;
    }
    return out;
}

}  // namespace x2w::vs::lower
