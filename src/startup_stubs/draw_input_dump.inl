// LOCAL (lt-lead probe, opt-in; nothing changes while X2W_DRAW_DUMP_INPUTS is unset). Included by d3d12_draw.inl just
// before draw_translated. Not related to X2W_DRAW_DUMP_FRAME / X2W_DRAW_DUMP_DIR (the per-frame target-0 bitmaps).
//
// X2W_DRAW_DUMP_INPUTS=<key hex>[:<occurrence>][,<key>[:<occurrence>]...] (up to 8 entries; "," ";" or " " between
// them; "*" for a key matches every key, for tests): the occurrence-th translated draw (default 1) of the key. Draws
// are counted per key (entries of the same key share one count, so k:1,k:2 are the first and second draw of k) over the
// run from present X2W_DRAW_DUMP_INPUTS_FROM_FRAME (default 0), in any frame (no traced frame needed). Each entry is
// dumped once. A draw that is counted but then not issued (a refusal after the count, or skipped index data) is
// uncounted again, so the next draw of the key takes its place. Indirect and mesh draws are never selected or counted;
// the first indirect one of a listed key logs a line. A listed key whose draws are refused before the count (no
// desktop pipeline, heap or texture refusals ...) logs its first refusal reason once ("x2w-dddump <key>: draw not
// issued (first refusal of this key): <why>"); a pipeline with no translation record at all is logged by the runtime's
// own "no translation for pipeline <key>" line.
// For the selected draw the runtime writes:
//   - the draw arguments, the vertex window the draw copies (vertex_first, vertex_count), the entry values with their
//     kinds, the list's root constants and root descriptor addresses, viewport 0, the bound targets and depth view;
//   - the index range the draw reads (CPU bytes at record time, the same bytes the draw copies);
//   - each srv buffer (vertex streams, root srvs, table buffers): the range as bound, CPU bytes at record time (at most
//     X2W_DRAW_DUMP_INPUTS_MAX_MIB MiB, default 16), with how it was bound (copied, in place, null, ...). An in-place
//     binding is read by the GPU later, when the list runs: the dump is what the CPU saw at record time;
//   - each srv texture: a GPU copy of mip 0 / slice 0 just BEFORE the draw. The whole subresource when that is at most
//     X2W_DRAW_DUMP_INPUTS_FULL_MIB MiB (default 4), else the box X2W_DRAW_DUMP_INPUTS_BOX=x,y,w,h (default 64x64 at
//     the centre, aligned to 4). Depth resources only whole (up to X2W_DRAW_DUMP_INPUTS_DEPTH_MIB, default 32);
//   - uav textures: copies before and after the draw;
//   - every bound render target: a copy BEFORE and AFTER the draw (same size rules), and the depth target after it;
//   - heap draws (`heap 1`): the title view heap's descriptor bytes (CPU, at most 4 MiB) and one line per view-token
//     slot (view id, kind, resource, dimensions) for the first 16384 slots. The descriptors a heap shader indexes are
//     computed on the GPU, so their textures and buffers are not followed by default: X2W_DRAW_DUMP_INPUTS_HEAP=<i>,<j>..
//     (decimal slot numbers, up to 32) follows those slots: a texture slot is copied like an srv texture, a buffer slot
//     (view token or hardware descriptor) is written as CPU bytes. The per-view info buffer (vinfo) is GPU-built and is
//     not dumped.
// Files go to X2W_DRAW_DUMP_INPUTS_DIR, else the folder of X2W_TRAP_LOG, else the current folder:
//   dddump_<key>_o<occurrence>_f<frame>_p<process id>.txt      index, one line per item (see lt_work/dddump_read.py)
//   dddump_<key>_o<occurrence>_f<frame>_p<process id>_<name>.bin
// A texture is copied only when its state is known: moved by this list, or untouched by it (COMMON) after an upload or
// GPU write filled it. It is restored after the copy. Otherwise the index says "skipped: not moved by this list and
// not uploaded or written yet" (for example a followed heap texture that was never uploaded).
// Texture files hold tightly packed rows (row pitch removed). GPU copies are read back at a later present once a marker
// copied at the end of the same command list has arrived and two more presents have passed (or at once through the test
// export x2w_draw_dump_inputs_test(1), after the caller waited for the list).
namespace dd_dump {

inline unsigned env_number(const char* name, unsigned fallback)
{
    char value[32]{};
    return GetEnvironmentVariableA(name, value, sizeof(value)) ? static_cast<unsigned>(std::strtoul(value, nullptr, 10)) : fallback;
}

constexpr u64 any_key = ~0ull;
struct Entry {
    u64 key = 0;  // any_key: "*", every key
    LONG occurrence = 1;
    unsigned counter = 0;  // index into Selection::counters (one per distinct key)
    bool done = false;
};
struct Counter {
    u64 key = 0;
    LONG seen = 0;  // draws of this key counted so far (issued, or in flight)
    bool announced = false, indirect_noted = false, refusal_noted = false;
};
struct Selection {
    Entry entries[8];
    unsigned count = 0;
    Counter counters[8];
    unsigned counter_count = 0;
    LONG from_frame = 0;
    std::uint32_t heap_slots[32]{};
    unsigned heap_count = 0;
};
inline std::mutex& lock()
{
    static std::mutex m;
    return m;
}
inline Selection& selection()
{
    static Selection s = [] {
        Selection out;
        char value[8 * 24 + 16]{};
        const DWORD n = GetEnvironmentVariableA("X2W_DRAW_DUMP_INPUTS", value, sizeof(value));
        if (!n || n >= sizeof(value)) return out;
        for (char* p = value; *p && out.count < 8;) {
            while (*p == ',' || *p == ';' || *p == ' ') ++p;
            char* end = nullptr;
            u64 key = 0;
            if (*p == '*') {  // any key (for tests: the first translated draw of whatever key comes)
                key = any_key;
                end = p + 1;
            } else {
                key = _strtoui64(p, &end, 16);
            }
            if (end == p) break;
            p = end;
            LONG occurrence = 1;
            if (*p == ':') {
                occurrence = std::strtol(p + 1, &end, 10);
                p = end;
            }
            if (!key) continue;
            unsigned c = 0;
            while (c < out.counter_count && out.counters[c].key != key) ++c;
            if (c == out.counter_count) out.counters[out.counter_count++].key = key;
            out.entries[out.count++] = Entry{key, occurrence < 1 ? 1 : occurrence, c};
        }
        out.from_frame = static_cast<LONG>(env_number("X2W_DRAW_DUMP_INPUTS_FROM_FRAME", 0));
        char slots[32 * 12]{};
        if (GetEnvironmentVariableA("X2W_DRAW_DUMP_INPUTS_HEAP", slots, sizeof(slots)))
            for (char* p = slots; *p && out.heap_count < 32;) {
                while (*p == ',' || *p == ';' || *p == ' ') ++p;
                char* end = nullptr;
                const unsigned long slot = std::strtoul(p, &end, 10);
                if (end == p) break;
                out.heap_slots[out.heap_count++] = static_cast<std::uint32_t>(slot);
                p = end;
            }
        for (unsigned i = 0; i < out.count; ++i)
            log("x2w-dddump X2W_DRAW_DUMP_INPUTS: key %016llx occurrence %ld (from frame %ld, %u heap slots followed)",
                static_cast<unsigned long long>(out.entries[i].key), out.entries[i].occurrence, out.from_frame, out.heap_count);
        return out;
    }();
    return s;
}
inline bool on() { return selection().count != 0; }

inline std::string out_dir()
{
    char path[MAX_PATH]{};
    DWORD n = GetEnvironmentVariableA("X2W_DRAW_DUMP_INPUTS_DIR", path, MAX_PATH);
    if (n && n < MAX_PATH) return std::string(path, n);
    n = GetEnvironmentVariableA("X2W_TRAP_LOG", path, MAX_PATH);
    if (n && n < MAX_PATH) {
        std::string s(path, n);
        const auto cut = s.find_last_of("\\/");
        return cut == std::string::npos ? std::string(".") : s.substr(0, cut);
    }
    return ".";
}

struct Copy {
    std::string name, line;
    Microsoft::WRL::ComPtr<ID3D12Resource> readback;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 row_bytes = 0;
};
struct Pending {
    u64 key = 0;
    std::string stem;
    std::vector<Copy> copies;
    std::vector<std::string> lines;  // record-time lines (CPU files already written)
    Microsoft::WRL::ComPtr<ID3D12Resource> marker_upload, marker_readback;
    std::uint32_t magic[4]{};
    int seen = -1;
    int waited = 0;
    bool active = false;
};
inline std::vector<Pending>& pending()
{
    static std::vector<Pending> p;
    return p;
}

// The title view heap a heap draw indexes (the same choice as draw_title_heap, d3d12_draw.inl).
inline bool title_heap(const Pass& pass, u64* address, u64* bytes)
{
    *address = *bytes = 0;
    if (pass.source_view_heap) {
        AcquireSRWLockShared(&note_lock);
        for (const HeapInfo& h : heap_infos())
            if (h.wrapper == pass.source_view_heap && h.title_cpu) {
                *address = h.title_cpu;
                *bytes = u64{h.logical_count} * h.increment;
            }
        ReleaseSRWLockShared(&note_lock);
    }
    return *address || title_view_heap(pass, address, bytes);
}

// A view record's resource, only while it is a live resource wrapper (the same test upload_heap_textures applies).
inline Wrapper* live(u64 title_resource)
{
    Wrapper* w = title_resource ? as_wrapper(reinterpret_cast<const void*>(title_resource)) : nullptr;
    if (!w || w != reinterpret_cast<Wrapper*>(title_resource) || w->own().kind != resource_kind || !w->own().real) return nullptr;
    return InterlockedCompareExchange(&w->own().refs, 0, 0) > 0 ? w : nullptr;
}

inline bool view_record(std::uint32_t id, ViewRecord& out)
{
    AcquireSRWLockShared(&note_lock);
    const bool known = id < view_records().size();
    if (known) out = view_records()[id];
    ReleaseSRWLockShared(&note_lock);
    return known;
}

// One selected draw, from the binding loop to the draw. Inactive (and free) for every other draw.
class Draw {
public:
    Draw(Wrapper* self, u64 key, bool indirect, bool indexed, UINT count, UINT instances, UINT start, INT base_vertex, UINT start_instance)
        : self_(self), key_(key)
    {
        if (!key || !on() || frame_number < selection().from_frame) return;
        std::lock_guard<std::mutex> guard(lock());
        Selection& s = selection();
        for (unsigned c = 0; c < s.counter_count; ++c) {
            Counter& k = s.counters[c];
            if (k.key != key && k.key != any_key) continue;
            if (indirect) {
                if (!k.indirect_noted) log("x2w-dddump %016llx: an indirect draw of this key is not dumped (plain draws only)", static_cast<unsigned long long>(key));
                k.indirect_noted = true;
                continue;
            }
            if (!k.announced) log("x2w-dddump %016llx: first draw of this key seen (frame %ld)", static_cast<unsigned long long>(key), frame_number);
            k.announced = true;
            ++k.seen;
            counted_ |= 1u << c;
        }
        for (unsigned i = 0; i < s.count && counted_; ++i) {
            Entry& e = s.entries[i];
            if (!e.done && (counted_ & (1u << e.counter)) && s.counters[e.counter].seen == e.occurrence) {
                entry_ = &e;
                break;
            }
        }
        if (!entry_) return;
        active_ = true;
        frame_ = frame_number + 1;  // the frame being recorded (presented next)
        char stem[96];
        std::snprintf(stem, sizeof(stem), "dddump_%016llx_o%ld_f%ld_p%lu", static_cast<unsigned long long>(key), entry_->occurrence, frame_,
                      static_cast<unsigned long>(GetCurrentProcessId()));
        stem_ = stem;
        dir_ = out_dir();
        line("draw %s count %u instances %u start %u base_vertex %d start_instance %u", indexed ? "indexed" : "plain", count, instances, start,
             base_vertex, start_instance);
        log("x2w-dddump %016llx: occurrence %ld selected (frame %ld)", static_cast<unsigned long long>(key), entry_->occurrence, frame_);
    }
    ~Draw()
    {
        if (!counted_ || issued_) return;
        std::lock_guard<std::mutex> guard(lock());
        Selection& s = selection();
        for (unsigned c = 0; c < s.counter_count; ++c)
            if (counted_ & (1u << c)) s.counters[c].seen -= 1;  // not issued: the next draw of this key takes its place
        if (active_)
            log("x2w-dddump %016llx: selected draw was not issued (%s); re-armed", static_cast<unsigned long long>(key_), why_ ? why_ : "refused");
    }
    explicit operator bool() const { return active_; }
    void not_issued(const char* why) { why_ = why; }

    void geometry(INT vertex_first, UINT vertex_count, bool indexed, u64 index_address, std::size_t index_bytes, UINT index_size, bool indices_ok,
                  UINT topology, const Pass& pass)
    {
        if (!active_) return;
        line("vertices first %d count %u topology 0x%x", vertex_first, vertex_count, topology);
        const D3D12_VIEWPORT& v = pass.viewport0;
        line("viewport0 %.9g %.9g %.9g %.9g depth %.9g %.9g (title set: viewport %d scissor %d)", v.TopLeftX, v.TopLeftY, v.Width, v.Height,
             v.MinDepth, v.MaxDepth, pass.viewport_set ? 1 : 0, pass.scissor_set ? 1 : 0);
        for (UINT k = 0; k < 8; ++k)
            if (pass.vb[k].BufferLocation)
                line("vb %u title 0x%llx bytes %u stride %u", k, static_cast<unsigned long long>(pass.vb[k].BufferLocation), pass.vb[k].SizeInBytes,
                     pass.vb[k].StrideInBytes);
        if (!indexed) return;
        line("ib title 0x%llx bytes %u format %d", static_cast<unsigned long long>(pass.ib.BufferLocation), pass.ib.SizeInBytes, static_cast<int>(pass.ib.Format));
        write_cpu("indices", indices_ok ? index_address : 0, index_bytes, "indices read by the draw (index size %u)", index_size);
    }

    // A texture srv as bound (texture null: bound null). how: "bound", "null", ...
    void srv_texture(UINT i, int kind, u64 entry, const std::uint32_t (&words)[8], ID3D12Resource* texture, DXGI_FORMAT view_format, const char* how)
    {
        if (!active_) return;
        char label[48];
        std::snprintf(label, sizeof(label), "srv%u", i);
        textures_.push_back(Tex{label, texture, false});
        line("srv %u texture kind %d entry 0x%llx words %08x %08x %08x %08x %08x %08x %08x %08x view_format %d desktop %p: %s", i, kind,
             static_cast<unsigned long long>(entry), words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7],
             static_cast<int>(view_format), static_cast<void*>(texture), how);
    }
    // A buffer srv: the range as computed, before the binding chooses how (then bound()).
    void srv_buffer(UINT i, int source, int a, int b, u64 entry, const std::uint32_t (&words)[8], u64 address, std::size_t bytes, UINT stride)
    {
        if (!active_) return;
        pending_srv_ = static_cast<int>(i);
        std::snprintf(srv_text_, sizeof(srv_text_),
                      "srv %u buffer source %d(%d,%d) entry 0x%llx words %08x %08x %08x %08x title 0x%llx bytes %zu stride %u", i, source, a, b,
                      static_cast<unsigned long long>(entry), words[0], words[1], words[2], words[3], static_cast<unsigned long long>(address), bytes,
                      stride);
        srv_address_ = address;
        srv_bytes_ = bytes;
    }
    void bound(UINT i, const char* how, std::size_t bound_bytes)
    {
        if (!active_ || pending_srv_ != static_cast<int>(i)) return;
        pending_srv_ = -1;
        char name[32];
        std::snprintf(name, sizeof(name), "srv%u", i);
        write_cpu(name, srv_address_, srv_bytes_, "%s: %s, %zu bytes bound", srv_text_, how, bound_bytes);
    }
    void uav_texture(UINT j, ID3D12Resource* texture)
    {
        if (!active_) return;
        char label[48];
        std::snprintf(label, sizeof(label), "uav%u", j);
        textures_.push_back(Tex{label, texture, true});
        line("uav %u texture desktop %p", j, static_cast<void*>(texture));
    }
    void uav_buffer(UINT j, std::uint32_t base, std::uint32_t stride, std::uint32_t word3)
    {
        if (!active_) return;
        line("uav %u buffer base-entry 0x%08x stride %u word3 0x%08x (contents not dumped)", j, base, stride, word3);
    }

    void values(const Translation& tr, const std::uint32_t* values, const Pass& pass)
    {
        if (!active_) return;
        std::string kinds;
        char part[32];
        for (std::size_t i = 0; i < tr.uds.size(); ++i) {
            std::snprintf(part, sizeof(part), " %d(%d,%d)=%08x", static_cast<int>(tr.uds[i].kind), tr.uds[i].a, tr.uds[i].b, values[i]);
            kinds += part;
        }
        line("entry %zu values, kind(a,b)=value:%s", tr.uds.size(), kinds.c_str());
        write_mem("entry", values, tr.uds.size() * sizeof(values[0]), "entry values (root parameter 0), u32 each");
        write_mem("root_constants", pass.constants, sizeof(pass.constants), "list root constants [parameter 0-15][word 0-7] u32");
        for (UINT p = 0; p < 16; ++p)
            if (pass.root_address[p]) line("root descriptor %u title 0x%llx", p, static_cast<unsigned long long>(pass.root_address[p]));
    }

    // Heap draws: the descriptor bytes, the token slots, and the followed slots.
    void heap(const Pass& pass)
    {
        if (!active_) return;
        u64 address = 0, bytes = 0;
        if (!title_heap(pass, &address, &bytes)) return line("heap: title view heap not reachable");
        const std::size_t want = static_cast<std::size_t>(bytes < (4u << 20) ? bytes : (4u << 20));
        write_cpu("heap", address, want, "title view heap descriptors (32 bytes each), heap 0x%llx bytes %llu",
                  static_cast<unsigned long long>(address), static_cast<unsigned long long>(bytes));
        const std::size_t readable = address && want ? static_cast<std::size_t>(readable_window(address, want)) : 0;
        const std::uint32_t slots = static_cast<std::uint32_t>(readable / 32 < 16384 ? readable / 32 : 16384);
        unsigned tokens = 0;
        for (std::uint32_t k = 0; k < slots; ++k) {
            const auto* d = reinterpret_cast<const std::uint32_t*>(address + u64{k} * 32);
            if (d[0] != view_token_magic) continue;
            ++tokens;
            line("heapslot %u %s", k, describe_token(d[1]).c_str());
        }
        line("heap slots scanned %u, view tokens %u", slots, tokens);
        const Selection& s = selection();
        for (unsigned f = 0; f < s.heap_count; ++f) follow(address, readable, s.heap_slots[f]);
    }

    // GPU copies before the draw (srv / uav textures, every render target). Call just before the draw is recorded.
    void before(ID3D12GraphicsCommandList* list, const Pass& pass)
    {
        if (!active_) return;
        for (const Tex& t : textures_)
            if (t.texture) copy_texture(list, t.texture, t.label + "_pre", t.label.c_str(), "pre");
        for (UINT k = 0; k < pass.target_count && k < 8; ++k) {
            char label[16];
            std::snprintf(label, sizeof(label), "rtv%u", k);
            line("rtv %u desktop %p format %d", k, static_cast<void*>(pass.targets[k]), static_cast<int>(pass.target_formats[k]));
            if (pass.targets[k]) copy_texture(list, pass.targets[k], std::string(label) + "_pre", label, "pre");
        }
        line("dsv desktop %p format %d read_only %d", static_cast<void*>(pass.dsv_resource), static_cast<int>(pass.dsv_format), pass.dsv_read_only ? 1 : 0);
    }
    // After the draw: render targets, depth target, uav textures; then the marker. Only when the draw was issued.
    void after(ID3D12GraphicsCommandList* list, const Pass& pass)
    {
        issued_ = true;
        if (!active_) return;
        for (UINT k = 0; k < pass.target_count && k < 8; ++k) {
            char label[16];
            std::snprintf(label, sizeof(label), "rtv%u", k);
            if (pass.targets[k]) copy_texture(list, pass.targets[k], std::string(label) + "_post", label, "post");
        }
        if (pass.dsv_resource) copy_texture(list, pass.dsv_resource, "dsv_post", "dsv", "post");
        for (const Tex& t : textures_)
            if (t.texture && t.uav) copy_texture(list, t.texture, t.label + "_post", t.label.c_str(), "post");
        finish(list);
    }

private:
    struct Tex {
        std::string label;
        ID3D12Resource* texture;
        bool uav;
    };

    void line(const char* format, ...)
    {
        char text[1024];
        va_list args;
        va_start(args, format);
        std::vsnprintf(text, sizeof(text), format, args);
        va_end(args);
        lines_.emplace_back(text);
    }

    void write_mem(const char* name, const void* data, std::size_t bytes, const char* note)
    {
        const std::string file = stem_ + "_" + name + ".bin";
        bool ok = false;
        if (std::FILE* f = std::fopen((dir_ + "\\" + file).c_str(), "wb")) {
            ok = !bytes || std::fwrite(data, 1, bytes, f) == bytes;
            ok = std::fclose(f) == 0 && ok;
        }
        line("file %s bytes %zu%s: %s", file.c_str(), bytes, ok ? "" : " (WRITE FAILED)", note);
    }
    // CPU bytes of a title range now, cut to the readable part and to X2W_DRAW_DUMP_INPUTS_MAX_MIB.
    void write_cpu(const char* name, u64 address, std::size_t bytes, const char* format, ...)
    {
        char note[1024];
        va_list args;
        va_start(args, format);
        std::vsnprintf(note, sizeof(note), format, args);
        va_end(args);
        static const std::size_t cap = std::size_t{env_number("X2W_DRAW_DUMP_INPUTS_MAX_MIB", 16)} << 20;
        const std::size_t wanted = bytes < cap ? bytes : cap;
        const std::size_t have = address && wanted ? static_cast<std::size_t>(readable_window(address, wanted)) : 0;
        if (!have) return line("file - bytes 0 of %zu (%s): %s", bytes, address ? "not readable" : "no address", note);
        const std::string file = stem_ + "_" + name + ".bin";
        bool ok = false;
        if (std::FILE* f = std::fopen((dir_ + "\\" + file).c_str(), "wb")) {
            ok = std::fwrite(reinterpret_cast<const void*>(address), 1, have, f) == have;
            ok = std::fclose(f) == 0 && ok;
        }
        line("file %s bytes %zu of %zu%s%s: %s", file.c_str(), have, bytes, have < bytes ? " (cut)" : "", ok ? "" : " (WRITE FAILED)", note);
    }

    std::string describe_token(std::uint32_t id)
    {
        ViewRecord r{};
        char text[256];
        if (!view_record(id, r)) {
            std::snprintf(text, sizeof(text), "view %u unknown", id);
            return text;
        }
        Wrapper* w = live(r.resource);
        const bool res = w != nullptr;
        ID3D12Resource* real = res ? static_cast<ID3D12Resource*>(w->own().real) : nullptr;
        const D3D12_RESOURCE_DESC d = real ? real->GetDesc() : D3D12_RESOURCE_DESC{};
        std::snprintf(text, sizeof(text), "view %u slot %u resource 0x%llx title 0x%llx desktop %p dim %d %llux%ux%u fmt %d desc_bytes %u", id, r.slot,
                      static_cast<unsigned long long>(r.resource), res ? static_cast<unsigned long long>(reinterpret_cast<u64>(w->own().title_memory)) : 0ull,
                      static_cast<void*>(real), static_cast<int>(d.Dimension), static_cast<unsigned long long>(d.Width), d.Height, d.DepthOrArraySize,
                      static_cast<int>(d.Format), r.desc_bytes);
        return text;
    }

    void follow(u64 heap, std::size_t readable, std::uint32_t slot)
    {
        char name[32];
        std::snprintf(name, sizeof(name), "heap%u", slot);
        if (u64{slot} * 32 + 32 > readable) return line("follow heap %u: outside the readable heap", slot);
        std::uint32_t d[8];
        std::memcpy(d, reinterpret_cast<const void*>(heap + u64{slot} * 32), 32);
        if (d[0] != view_token_magic) {
            const u64 at = d[0] | (static_cast<u64>(d[1] & 0xffff) << 32);
            const UINT stride = (d[1] >> 16) & 0x3fff;
            return write_cpu(name, at, std::size_t{stride ? stride : 1u} * d[2], "follow heap %u hardware words %08x %08x %08x %08x", slot, d[0], d[1],
                             d[2], d[3]);
        }
        ViewRecord r{};
        if (!view_record(d[1], r)) return line("follow heap %u: view %u unknown", slot, d[1]);
        Wrapper* w = live(r.resource);
        if (r.resource && !w) return line("follow heap %u: view %u names a released or unknown resource", slot, d[1]);
        if (r.slot == 17 && r.desc_bytes >= 12) {
            u64 at = 0;
            std::uint32_t size = 0;
            std::memcpy(&at, r.desc, 8);
            std::memcpy(&size, r.desc + 8, 4);
            return write_cpu(name, at, size, "follow heap %u constant buffer view %u", slot, d[1]);
        }
        if (r.slot == 18 && r.desc_bytes >= 40) {
            D3D12_SHADER_RESOURCE_VIEW_DESC v;
            std::memcpy(&v, r.desc, 40);
            if (v.ViewDimension == D3D12_SRV_DIMENSION_BUFFER) {
                if (!w || !w->own().title_memory) return line("follow heap %u: buffer view %u without title memory", slot, d[1]);
                const bool typed = !v.Buffer.StructureByteStride && v.Format != DXGI_FORMAT_R32_TYPELESS;
                const UINT element = v.Buffer.StructureByteStride ? v.Buffer.StructureByteStride : typed ? x2w_startup::typed_buffer_format_bytes(v.Format) : 4;
                return write_cpu(name, reinterpret_cast<u64>(w->own().title_memory) + v.Buffer.FirstElement * element,
                                 std::size_t{v.Buffer.NumElements} * element, "follow heap %u buffer view %u element %u", slot, d[1], element);
            }
            ID3D12Resource* texture = sampled_address_texture(w);
            line("follow heap %u texture view %u dim %d fmt %d desktop %p", slot, d[1], static_cast<int>(v.ViewDimension), static_cast<int>(v.Format),
                 static_cast<void*>(texture));
            if (texture) textures_.push_back(Tex{name, texture, false});
            return;
        }
        line("follow heap %u: view %u slot %u not followed", slot, d[1], r.slot);
    }

    void copy_texture(ID3D12GraphicsCommandList* list, ID3D12Resource* texture, const std::string& file_label, const char* what, const char* phase)
    {
        auto* device = device_of(the_device);
        const D3D12_RESOURCE_DESC rd = texture->GetDesc();
        if (rd.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) return line("tex %s %s skipped: a buffer", what, phase);
        if (rd.SampleDesc.Count != 1) return line("tex %s %s skipped: multisampled (%u samples)", what, phase, rd.SampleDesc.Count);
        const bool depth = (rd.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) != 0;
        D3D12_FEATURE_DATA_FORMAT_INFO info{rd.Format, 1};
        if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO, &info, sizeof(info)))) info.PlaneCount = 1;
        const UINT per_plane = rd.MipLevels * (rd.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? 1u : rd.DepthOrArraySize);
        static const UINT64 full_mib = env_number("X2W_DRAW_DUMP_INPUTS_FULL_MIB", 4), depth_mib = env_number("X2W_DRAW_DUMP_INPUTS_DEPTH_MIB", 32);
        const UINT64 full_limit = (depth ? depth_mib : full_mib) << 20;
        // The state this list holds the texture in now, restored after the copy. A texture this list has not moved is in
        // COMMON (lists return what they moved to COMMON at Close), but it is copied only once something filled it: the
        // upload record says the title memory was staged into it or a list wrote it. Any other untracked texture is
        // skipped (a heap texture never uploaded, copied as if in COMMON, made Close fail with E_INVALIDARG).
        D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_COMMON;
        bool tracked = false;
        if (const auto* entry = pass_of(self_).states.find(texture)) before = entry->depth, tracked = true;
        if (!tracked) {
            const auto found = texture_uploads().find(texture);
            const bool filled = found != texture_uploads().end() && (found->second.staged || found->second.gpu_written) && !found->second.pending_list;
            if (!filled)
                return line("tex %s %s desktop %p skipped: not moved by this list and not uploaded or written yet", what, phase, static_cast<void*>(texture));
        }
        for (UINT plane = 0; plane < (info.PlaneCount ? info.PlaneCount : 1u); ++plane) {
            const UINT sub = plane * per_plane;
            Copy c;
            UINT64 total = 0;
            device->GetCopyableFootprints(&rd, sub, 1, 0, &c.footprint, &c.rows, &c.row_bytes, &total);
            D3D12_BOX box{0, 0, 0, static_cast<UINT>(rd.Width), rd.Height, c.footprint.Footprint.Depth};
            const bool whole = total <= full_limit;
            if (!whole) {
                if (depth) return line("tex %s %s skipped: depth resource above X2W_DRAW_DUMP_INPUTS_DEPTH_MIB (D3D12 copies depth only whole)", what, phase);
                unsigned bx = 0, by = 0, bw = 64, bh = 64;
                char value[64]{};
                const bool given = GetEnvironmentVariableA("X2W_DRAW_DUMP_INPUTS_BOX", value, sizeof(value)) &&
                                   std::sscanf(value, "%u,%u,%u,%u", &bx, &by, &bw, &bh) == 4;
                const UINT w = static_cast<UINT>(rd.Width), h = rd.Height;
                if (!given) bx = w > 64 ? (w / 2 - 32) : 0, by = h > 64 ? (h / 2 - 32) : 0;
                bx &= ~3u, by &= ~3u, bw = (bw + 3) & ~3u, bh = (bh + 3) & ~3u;
                if (bx >= w || by >= h) return line("tex %s %s skipped: box outside the texture", what, phase);
                if (bx + bw > w) bw = w - bx;
                if (by + bh > h) bh = h - by;
                D3D12_RESOURCE_DESC part = rd;
                part.Width = bw, part.Height = bh, part.MipLevels = 1, part.DepthOrArraySize = 1;
                device->GetCopyableFootprints(&part, 0, 1, 0, &c.footprint, &c.rows, &c.row_bytes, &total);
                box = D3D12_BOX{bx, by, 0, bx + bw, by + bh, 1};
            }
            c.readback = cx_dump::buffer(D3D12_HEAP_TYPE_READBACK, total);
            if (!c.readback) return line("tex %s %s skipped: readback allocation failed", what, phase);
            char name[160], text[512];
            if (info.PlaneCount > 1) std::snprintf(name, sizeof(name), "%s_%s_p%u.bin", stem_.c_str(), file_label.c_str(), plane);
            else std::snprintf(name, sizeof(name), "%s_%s.bin", stem_.c_str(), file_label.c_str());
            std::snprintf(text, sizeof(text),
                          "tex %s %s desktop %p dim %d %llux%ux%u mips %u fmt %d flags 0x%x plane %u box %u,%u %ux%u row_bytes %llu rows %u file %s", what,
                          phase, static_cast<void*>(texture), static_cast<int>(rd.Dimension), static_cast<unsigned long long>(rd.Width), rd.Height,
                          rd.DepthOrArraySize, rd.MipLevels, static_cast<int>(rd.Format), static_cast<unsigned>(rd.Flags), plane, box.left, box.top,
                          box.right - box.left, box.bottom - box.top, static_cast<unsigned long long>(c.row_bytes), c.rows, name);
            c.name = name;
            c.line = text;
            require_state(self_, texture, D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION to{c.readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            to.PlacedFootprint = c.footprint;
            const D3D12_TEXTURE_COPY_LOCATION from{texture, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {sub}};
            list->CopyTextureRegion(&to, 0, 0, 0, &from, whole ? nullptr : &box);
            require_state(self_, texture, before);
            copies_.push_back(std::move(c));
        }
    }

    void finish(ID3D12GraphicsCommandList* list)
    {
        finished_ = true;
        std::lock_guard<std::mutex> guard(lock());
        entry_->done = true;
        Pending p;
        p.key = key_;
        p.stem = stem_;
        p.copies = std::move(copies_);
        p.lines = std::move(lines_);
        p.marker_upload = cx_dump::buffer(D3D12_HEAP_TYPE_UPLOAD, 16);
        p.marker_readback = cx_dump::buffer(D3D12_HEAP_TYPE_READBACK, 16);
        p.magic[0] = 0x504d4444;  // "DDMP"
        p.magic[1] = static_cast<std::uint32_t>(key_);
        p.magic[2] = static_cast<std::uint32_t>(frame_);
        p.magic[3] = static_cast<std::uint32_t>(GetTickCount()) ^ static_cast<std::uint32_t>(pending().size());
        void* memory = nullptr;
        if (!p.marker_upload || !p.marker_readback || FAILED(p.marker_upload->Map(0, nullptr, &memory))) {
            log("x2w-dddump %016llx: marker buffers failed; nothing is read back", static_cast<unsigned long long>(key_));
            return;
        }
        std::memcpy(memory, p.magic, 16);
        p.marker_upload->Unmap(0, nullptr);
        list->CopyBufferRegion(p.marker_readback.Get(), 0, p.marker_upload.Get(), 0, 16);
        p.active = true;
        log("x2w-dddump %016llx: %zu texture copies and %zu lines recorded; read back at a later present", static_cast<unsigned long long>(key_),
            p.copies.size(), p.lines.size());
        pending().push_back(std::move(p));
    }

    Wrapper* self_;
    u64 key_;
    Entry* entry_ = nullptr;
    unsigned counted_ = 0;  // bits: the counters this draw incremented
    bool active_ = false, finished_ = false, issued_ = false;
    const char* why_ = nullptr;
    LONG frame_ = 0;
    std::string stem_, dir_;
    std::vector<std::string> lines_;
    std::vector<Tex> textures_;
    std::vector<Copy> copies_;
    int pending_srv_ = -1;
    char srv_text_[400]{};
    u64 srv_address_ = 0;
    std::size_t srv_bytes_ = 0;
};

// Each present (and the test export): write a dump once its marker has arrived and, unless `now`, two more presents
// have passed. Returns the number of dumps written.
inline int poll(bool now = false)
{
    if (!on()) return 0;
    std::lock_guard<std::mutex> guard(lock());
    int written = 0;
    const std::string dir = out_dir();
    for (auto it = pending().begin(); it != pending().end();) {
        Pending& p = *it;
        if (p.seen < 0) {
            void* memory = nullptr;
            const D3D12_RANGE read{0, 16}, none{};
            if (SUCCEEDED(p.marker_readback->Map(0, &read, &memory))) {
                if (!std::memcmp(memory, p.magic, 16)) p.seen = 0;
                p.marker_readback->Unmap(0, &none);
            }
            if (p.seen < 0) {
                if (!now && ++p.waited > 600) {
                    log("x2w-dddump %016llx: marker never arrived (list not executed?); dump dropped", static_cast<unsigned long long>(p.key));
                    it = pending().erase(it);
                } else {
                    ++it;
                }
                continue;
            }
        }
        if (!now && ++p.seen < 2) {
            ++it;
            continue;
        }
        std::FILE* index = std::fopen((dir + "\\" + p.stem + ".txt").c_str(), "w");
        if (index) std::fprintf(index, "x2w-dddump 1 %s\n", p.stem.c_str());
        for (const auto& l : p.lines)
            if (index) std::fprintf(index, "%s\n", l.c_str());
        for (Copy& c : p.copies) {
            void* memory = nullptr;
            // The last row ends at row_bytes, not at a full pitch (the buffer is sized by GetCopyableFootprints).
            const UINT64 lines = UINT64{c.rows} * c.footprint.Footprint.Depth;
            const D3D12_RANGE read{0, static_cast<SIZE_T>(c.footprint.Offset + (lines ? (lines - 1) * c.footprint.Footprint.RowPitch : 0) + c.row_bytes)},
                none{};
            bool ok = false;
            if (SUCCEEDED(c.readback->Map(0, &read, &memory))) {
                if (std::FILE* f = std::fopen((dir + "\\" + c.name).c_str(), "wb")) {
                    const auto* base = static_cast<const unsigned char*>(memory) + c.footprint.Offset;
                    for (UINT z = 0; z < c.footprint.Footprint.Depth; ++z)
                        for (UINT y = 0; y < c.rows; ++y)
                            std::fwrite(base + (UINT64{z} * c.rows + y) * c.footprint.Footprint.RowPitch, 1, static_cast<std::size_t>(c.row_bytes), f);
                    ok = std::fclose(f) == 0;
                }
                c.readback->Unmap(0, &none);
            }
            if (index) std::fprintf(index, "%s%s\n", c.line.c_str(), ok ? "" : " (write failed)");
        }
        if (index) std::fclose(index);
        log("x2w-dddump %016llx: %zu lines and %zu texture copies written to %s\\%s.*", static_cast<unsigned long long>(p.key), p.lines.size(),
            p.copies.size(), dir.c_str(), p.stem.c_str());
        ++written;
        it = pending().erase(it);
    }
    return written;
}

}  // namespace dd_dump

// From the draw calls (slots 12 and 13) when a translated draw returned false: the first refusal of a listed key is
// logged once (a refusal before the dump hook leaves no other trace in the dump log).
void draw_dump_note_not_drawn(u64 key, const char* why)
{
    if (!key || !dd_dump::on()) return;
    std::lock_guard<std::mutex> guard(dd_dump::lock());
    dd_dump::Selection& s = dd_dump::selection();
    for (unsigned c = 0; c < s.counter_count; ++c) {
        dd_dump::Counter& k = s.counters[c];
        if (k.key != key || k.refusal_noted) continue;
        k.refusal_noted = true;
        log("x2w-dddump %016llx: draw not issued (first refusal of this key): %s", static_cast<unsigned long long>(key), why ? why : "draw failed");
    }
}

// Test export: op 1 writes every dump whose marker has arrived now (the caller waited for its list); returns the count.
extern "C" __declspec(dllexport) int WINAPI x2w_draw_dump_inputs_test(int op)
{
    return op == 1 ? dd_dump::poll(true) : -1;
}
