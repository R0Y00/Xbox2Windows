// LOCAL (lt-lead probe, opt-in; nothing changes while the switches are unset).
//
// X2W_CX_TRACE_CLEARS=1: in a traced frame (X2W_D3D12_TRACE_BUSY / X2W_D3D12_TRACE_FRAME), one line per
// ClearRenderTargetView, ClearDepthStencilView and ClearUnorderedAccessView* call:
//   x2w-frame clear rtv|dsv|uav-float|uav-uint view=0x<title handle> target <desktop resource> <w>x<h> fmt <n>
//   [title 0x<address>] values ... rects <n> [dropped: <reason>]
// The desktop resource pointer is the "desktop" field of the trace's cx-img lines, so a clear joins those lines.
//
// X2W_CX_DUMP_INPUTS=<key hex>[,<occurrence>]: the occurrence-th dispatch (default 1) of that compute key in a traced
// frame. Only one dispatch per run is dumped. For it, the runtime writes:
//   - every bound buffer: the title bytes the CPU sees when the dispatch is recorded (the GPU reads the same pages in
//     place when the list runs; a later title or GPU write is not seen); at most 64 KiB per buffer;
//   - every bound texture (SRV and UAV): a GPU copy of mip 0 / slice 0 taken just BEFORE the dispatch ("pre"), and
//     for UAV textures a second copy just AFTER it ("post"). Each copy is the whole subresource when that is at most
//     X2W_CX_DUMP_FULL_MIB MiB (default 1), else the box X2W_CX_DUMP_BOX=x,y,w,h (default 64x64 at the centre,
//     aligned to 4). Depth/stencil resources can only be copied whole (D3D12), one file per plane, or are skipped
//     above X2W_CX_DUMP_DEPTH_MIB (default 32).
// Files go to X2W_CX_DUMP_DIR, else the folder of X2W_TRAP_LOG, else the current folder:
//   cxdump_<key>_f<frame>.txt        index: one line per buffer / texture copy (see the board/ready note)
//   cxdump_<key>_f<frame>_buf<slot>.bin, cxdump_<key>_f<frame>_<pre|post>_<slot>[_p<plane>].bin
// Texture files hold tightly packed rows (row pitch removed). The copies are read back at a later present, after a
// marker copied at the end of the same command list has arrived plus two more presents.
namespace cx_dump {

inline bool clears_on()
{
    static const bool on = x2w_local::env_on("X2W_CX_TRACE_CLEARS");
    return on;
}

inline u64 title_of(u64 title_resource)
{
    if (!title_resource) return 0;
    Wrapper* w = as_wrapper(reinterpret_cast<const void*>(title_resource));
    return w && w->own().kind == resource_kind ? reinterpret_cast<u64>(w->own().title_memory) : 0;
}

// kind: "rtv", "dsv", "uav-float", "uav-uint". values: 4 floats / uints (rtv, uav) or null (dsv: depth, stencil).
inline void note_clear(const char* kind, u64 handle, ID3D12Resource* target, u64 title_resource, const void* values, bool floats,
                       float depth, unsigned stencil, unsigned flags, UINT rects, const char* dropped)
{
    if (!clears_on() || !trace_frame_active) return;
    char text[512];
    int n = std::snprintf(text, sizeof(text), "x2w-frame clear %s view=0x%llx target %p", kind, static_cast<unsigned long long>(handle),
                          static_cast<void*>(target));
    if (target) {
        const D3D12_RESOURCE_DESC d = target->GetDesc();
        n += std::snprintf(text + n, sizeof(text) - n, " %llux%u fmt %d", static_cast<unsigned long long>(d.Width), d.Height, static_cast<int>(d.Format));
    }
    if (const u64 title = title_of(title_resource)) n += std::snprintf(text + n, sizeof(text) - n, " title 0x%llx", static_cast<unsigned long long>(title));
    if (values && x2w_local::readable(values, 16)) {
        if (floats) {
            const float* v = static_cast<const float*>(values);
            n += std::snprintf(text + n, sizeof(text) - n, " colour %.9g %.9g %.9g %.9g", v[0], v[1], v[2], v[3]);
        } else {
            const UINT* v = static_cast<const UINT*>(values);
            n += std::snprintf(text + n, sizeof(text) - n, " values %08x %08x %08x %08x", v[0], v[1], v[2], v[3]);
        }
    } else if (!std::strcmp(kind, "dsv")) {
        n += std::snprintf(text + n, sizeof(text) - n, " flags 0x%x depth %.9g stencil %u", flags, depth, stencil);
    } else {
        n += std::snprintf(text + n, sizeof(text) - n, " values unreadable");
    }
    n += std::snprintf(text + n, sizeof(text) - n, " rects %u", rects);
    if (dropped) std::snprintf(text + n, sizeof(text) - n, " dropped: %s", dropped);
    log("%s", text);
}

struct Selection {
    u64 key = 0;
    LONG occurrence = 1;
};
inline const Selection& selection()
{
    static const Selection s = [] {
        Selection out;
        char value[64]{};
        const DWORD n = GetEnvironmentVariableA("X2W_CX_DUMP_INPUTS", value, sizeof(value));
        if (!n || n >= sizeof(value)) return out;
        char* end = nullptr;
        out.key = std::strtoull(value, &end, 16);
        if (end && *end == ',') out.occurrence = std::atol(end + 1);
        if (out.occurrence < 1) out.occurrence = 1;
        log("x2w-cxdump X2W_CX_DUMP_INPUTS: key %016llx occurrence %ld in a traced frame", static_cast<unsigned long long>(out.key), out.occurrence);
        return out;
    }();
    return s;
}
inline unsigned env_number(const char* name, unsigned fallback)
{
    char value[32]{};
    return GetEnvironmentVariableA(name, value, sizeof(value)) ? static_cast<unsigned>(std::strtoul(value, nullptr, 10)) : fallback;
}
inline std::string out_dir()
{
    char path[MAX_PATH]{};
    DWORD n = GetEnvironmentVariableA("X2W_CX_DUMP_DIR", path, MAX_PATH);
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
    std::string name;   // file name
    std::string line;   // index line
    Microsoft::WRL::ComPtr<ID3D12Resource> readback;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 row_bytes = 0;
};
struct Pending {
    u64 key = 0;
    LONG frame = 0;
    std::vector<Copy> copies;
    std::vector<std::pair<std::string, std::string>> buffers;  // name, line (bytes written at record time)
    Microsoft::WRL::ComPtr<ID3D12Resource> marker_upload, marker_readback;
    std::uint32_t magic[4]{};
    int seen = -1;        // presents since the marker arrived (-1: not yet)
    int waited = 0;
    bool active = false;
};
inline std::mutex& lock()
{
    static std::mutex m;
    return m;
}
inline Pending& pending()
{
    static Pending p;
    return p;
}
inline LONG& seen_in_frame()
{
    static LONG count = 0;
    return count;
}
inline LONG& counted_frame()
{
    static LONG frame = 0;
    return frame;
}
inline bool& done()
{
    static bool d = false;
    return d;
}

inline Microsoft::WRL::ComPtr<ID3D12Resource> buffer(D3D12_HEAP_TYPE type, UINT64 bytes)
{
    Microsoft::WRL::ComPtr<ID3D12Resource> out;
    const D3D12_HEAP_PROPERTIES heap{type};
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = bytes ? bytes : 1;
    d.Height = d.DepthOrArraySize = d.MipLevels = d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    auto* device = device_of(the_device);
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d,
                                               type == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST,
                                               nullptr, IID_PPV_ARGS(&out))))
        out.Reset();
    else
        x2w_budget::committed(device, out.Get(), type);
    return out;
}

// One selected dispatch, from the binding loop to the end of recording. Inactive (and free) for every other dispatch.
class Dispatch {
public:
    Dispatch(u64 key, UINT x, UINT y, UINT z) : key_(key)
    {
        const Selection& s = selection();
        if (!s.key || s.key != key || !trace_frame_active) return;
        std::lock_guard<std::mutex> guard(lock());
        if (done() || pending().active) return;
        if (counted_frame() != frame_number) counted_frame() = frame_number, seen_in_frame() = 0;
        if (++seen_in_frame() != s.occurrence) return;
        active_ = true;
        frame_ = frame_number + 1;  // the frame being recorded (presented next)
        std::snprintf(stem_, sizeof(stem_), "cxdump_%016llx_f%ld", static_cast<unsigned long long>(key), frame_);
        log("x2w-cxdump %016llx dispatch (%u, %u, %u): occurrence %ld of frame %ld selected", static_cast<unsigned long long>(key), x, y, z,
            s.occurrence, frame_);
    }
    ~Dispatch()
    {
        if (active_ && !finished_) {
            std::lock_guard<std::mutex> guard(lock());
            seen_in_frame() -= 1;  // the dispatch did not run: the next one of this key takes its place
            log("x2w-cxdump %016llx: selected dispatch did not run; re-armed", static_cast<unsigned long long>(key_));
        }
    }
    explicit operator bool() const { return active_; }

    // From the binding loop: slot i as bound. title_resource is the view record's resource (textures), address/bytes
    // the bound buffer range.
    void slot(UINT i, bool uav, int kind, bool bound, u64 title_resource, u64 address, std::size_t bytes)
    {
        if (!active_) return;
        Slot s{i, uav, kind, bound, nullptr, title_of(title_resource), address, bytes};
        if (kind && bound) {
            Wrapper* w = title_resource ? as_wrapper(reinterpret_cast<const void*>(title_resource)) : nullptr;
            if (w && w->own().kind == resource_kind) s.texture = static_cast<ID3D12Resource*>(w->own().real);
        }
        slots_.push_back(s);
    }

    // Buffers (CPU bytes, now) and texture copies recorded into `list`. phase "pre": every texture; "post": UAV textures.
    void copy(Wrapper* self, ID3D12GraphicsCommandList* list, const char* phase)
    {
        if (!active_) return;
        const bool pre = !std::strcmp(phase, "pre");
        if (pre) write_buffers();
        for (const Slot& s : slots_) {
            if (!s.texture || (!pre && !s.uav)) continue;
            copy_texture(self, list, s, phase);
        }
    }

    // Marker copy, then hand the copies to poll(). Called once after the dispatch was recorded.
    void finish(ID3D12GraphicsCommandList* list)
    {
        if (!active_) return;
        finished_ = true;
        std::lock_guard<std::mutex> guard(lock());
        Pending& p = pending();
        p = Pending{};
        p.key = key_;
        p.frame = frame_;
        p.copies = std::move(copies_);
        p.buffers = std::move(buffers_);
        p.marker_upload = buffer(D3D12_HEAP_TYPE_UPLOAD, 16);
        p.marker_readback = buffer(D3D12_HEAP_TYPE_READBACK, 16);
        p.magic[0] = 0x504d4443;  // "CDMP"
        p.magic[1] = static_cast<std::uint32_t>(key_);
        p.magic[2] = static_cast<std::uint32_t>(frame_);
        p.magic[3] = static_cast<std::uint32_t>(GetTickCount());
        void* memory = nullptr;
        if (!p.marker_upload || !p.marker_readback || FAILED(p.marker_upload->Map(0, nullptr, &memory))) {
            log("x2w-cxdump %016llx: marker buffers failed; nothing is read back", static_cast<unsigned long long>(key_));
            p = Pending{};
            done() = true;
            return;
        }
        std::memcpy(memory, p.magic, 16);
        p.marker_upload->Unmap(0, nullptr);
        list->CopyBufferRegion(p.marker_readback.Get(), 0, p.marker_upload.Get(), 0, 16);
        p.active = true;
        done() = true;
        log("x2w-cxdump %016llx: %zu texture copies and %zu buffers recorded; read back at a later present", static_cast<unsigned long long>(key_),
            p.copies.size(), p.buffers.size());
    }

private:
    struct Slot {
        UINT index;
        bool uav;
        int kind;
        bool bound;
        ID3D12Resource* texture;
        u64 title;
        u64 address;
        std::size_t bytes;
    };

    void write_buffers()
    {
        const std::string dir = out_dir();
        for (const Slot& s : slots_) {
            if (s.kind) continue;
            char name[96], line[256];
            if (!s.address || !s.bytes) {
                std::snprintf(line, sizeof(line), "buf %u %s unbound", s.index, s.uav ? "uav" : "srv");
                buffers_.emplace_back(std::string(), line);
                continue;
            }
            const std::size_t bytes = s.bytes < (64u << 10) ? s.bytes : (64u << 10);
            std::snprintf(name, sizeof(name), "%s_buf%u.bin", stem_, s.index);
            const bool readable = x2w_local::readable(reinterpret_cast<const void*>(s.address), bytes);
            if (readable) {
                if (std::FILE* f = std::fopen((dir + "\\" + name).c_str(), "wb")) {
                    std::fwrite(reinterpret_cast<const void*>(s.address), 1, bytes, f);
                    std::fclose(f);
                }
            }
            std::snprintf(line, sizeof(line), "buf %u %s title 0x%llx bytes %zu written %zu file %s", s.index, s.uav ? "uav" : "srv",
                          static_cast<unsigned long long>(s.address), s.bytes, readable ? bytes : 0, readable ? name : "-");
            buffers_.emplace_back(name, line);
        }
    }

    void copy_texture(Wrapper* self, ID3D12GraphicsCommandList* list, const Slot& s, const char* phase)
    {
        auto* device = device_of(the_device);
        const D3D12_RESOURCE_DESC rd = s.texture->GetDesc();
        if (rd.SampleDesc.Count != 1) return note_skip(s, phase, "multisampled");
        const bool depth = (rd.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) != 0;
        D3D12_FEATURE_DATA_FORMAT_INFO info{rd.Format, 1};
        if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO, &info, sizeof(info)))) info.PlaneCount = 1;
        const UINT per_plane = rd.MipLevels * (rd.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? 1u : rd.DepthOrArraySize);
        const UINT64 full_limit = UINT64{env_number(depth ? "X2W_CX_DUMP_DEPTH_MIB" : "X2W_CX_DUMP_FULL_MIB", depth ? 32u : 1u)} << 20;
        for (UINT plane = 0; plane < (info.PlaneCount ? info.PlaneCount : 1u); ++plane) {
            const UINT sub = plane * per_plane;
            Copy c;
            UINT64 total = 0;
            device->GetCopyableFootprints(&rd, sub, 1, 0, &c.footprint, &c.rows, &c.row_bytes, &total);
            D3D12_BOX box{0, 0, 0, static_cast<UINT>(rd.Width), rd.Height, c.footprint.Footprint.Depth};
            const bool whole = total <= full_limit;
            if (!whole) {
                if (depth) return note_skip(s, phase, "depth resource above X2W_CX_DUMP_DEPTH_MIB (D3D12 copies depth only whole)");
                unsigned bx = 0, by = 0, bw = 64, bh = 64;
                char value[64]{};
                const bool given = GetEnvironmentVariableA("X2W_CX_DUMP_BOX", value, sizeof(value)) &&
                                   std::sscanf(value, "%u,%u,%u,%u", &bx, &by, &bw, &bh) == 4;
                const UINT w = static_cast<UINT>(rd.Width), h = rd.Height;
                if (!given) bx = w > 64 ? (w / 2 - 32) : 0, by = h > 64 ? (h / 2 - 32) : 0;
                bx &= ~3u, by &= ~3u, bw = (bw + 3) & ~3u, bh = (bh + 3) & ~3u;
                if (bx >= w || by >= h) return note_skip(s, phase, "box outside the texture");
                if (bx + bw > w) bw = w - bx;
                if (by + bh > h) bh = h - by;
                D3D12_RESOURCE_DESC part = rd;
                part.Width = bw, part.Height = bh, part.MipLevels = 1, part.DepthOrArraySize = 1;
                device->GetCopyableFootprints(&part, 0, 1, 0, &c.footprint, &c.rows, &c.row_bytes, &total);
                box = D3D12_BOX{bx, by, 0, bx + bw, by + bh, 1};
            }
            c.readback = buffer(D3D12_HEAP_TYPE_READBACK, total);
            if (!c.readback) return note_skip(s, phase, "readback allocation failed");
            char name[112], line[384];
            if (info.PlaneCount > 1) std::snprintf(name, sizeof(name), "%s_%s_%u_p%u.bin", stem_, phase, s.index, plane);
            else std::snprintf(name, sizeof(name), "%s_%s_%u.bin", stem_, phase, s.index);
            std::snprintf(line, sizeof(line),
                          "tex %u %s %s desktop %p title 0x%llx %llux%ux%u mips %u fmt %d flags 0x%x plane %u box %u,%u %ux%u row_bytes %llu rows %u file %s",
                          s.index, s.uav ? "uav" : "srv", phase, static_cast<void*>(s.texture), static_cast<unsigned long long>(s.title),
                          static_cast<unsigned long long>(rd.Width), rd.Height, rd.DepthOrArraySize, rd.MipLevels, static_cast<int>(rd.Format),
                          static_cast<unsigned>(rd.Flags), plane, box.left, box.top, box.right - box.left, box.bottom - box.top,
                          static_cast<unsigned long long>(c.row_bytes), c.rows, name);
            c.name = name;
            c.line = line;
            // The state this list holds the texture in now (set by the binding), restored after the copy.
            D3D12_RESOURCE_STATES before = D3D12_RESOURCE_STATE_COMMON;
            if (const auto* entry = pass_of(self).states.find(s.texture)) before = entry->depth;
            require_state(self, s.texture, D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION to{c.readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
            to.PlacedFootprint = c.footprint;
            const D3D12_TEXTURE_COPY_LOCATION from{s.texture, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {sub}};
            list->CopyTextureRegion(&to, 0, 0, 0, &from, whole ? nullptr : &box);
            require_state(self, s.texture, before);
            copies_.push_back(std::move(c));
        }
    }

    void note_skip(const Slot& s, const char* phase, const char* why)
    {
        char line[256];
        std::snprintf(line, sizeof(line), "tex %u %s %s desktop %p title 0x%llx skipped: %s", s.index, s.uav ? "uav" : "srv", phase,
                      static_cast<void*>(s.texture), static_cast<unsigned long long>(s.title), why);
        buffers_.emplace_back(std::string(), line);
    }

    u64 key_;
    bool active_ = false, finished_ = false;
    LONG frame_ = 0;
    char stem_[64]{};
    std::vector<Slot> slots_;
    std::vector<Copy> copies_;
    std::vector<std::pair<std::string, std::string>> buffers_;
};

// Each present: read the copies back once the marker has arrived and two more presents have passed.
inline void poll()
{
    if (!selection().key) return;
    std::lock_guard<std::mutex> guard(lock());
    Pending& p = pending();
    if (!p.active) return;
    if (p.seen < 0) {
        void* memory = nullptr;
        const D3D12_RANGE read{0, 16}, none{};
        if (SUCCEEDED(p.marker_readback->Map(0, &read, &memory))) {
            if (!std::memcmp(memory, p.magic, 16)) p.seen = 0;
            p.marker_readback->Unmap(0, &none);
        }
        if (p.seen < 0 && ++p.waited > 600) {
            log("x2w-cxdump %016llx: marker never arrived (list not executed?); dump dropped", static_cast<unsigned long long>(p.key));
            p = Pending{};
        }
        return;
    }
    if (++p.seen < 2) return;
    const std::string dir = out_dir();
    char stem[64];
    std::snprintf(stem, sizeof(stem), "cxdump_%016llx_f%ld", static_cast<unsigned long long>(p.key), p.frame);
    std::FILE* index = std::fopen((dir + "\\" + stem + ".txt").c_str(), "w");
    if (index) std::fprintf(index, "x2w-cxdump 1 key %016llx frame %ld\n", static_cast<unsigned long long>(p.key), p.frame);
    for (const auto& b : p.buffers)
        if (index) std::fprintf(index, "%s\n", b.second.c_str());
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
                for (UINT zslice = 0; zslice < c.footprint.Footprint.Depth; ++zslice)
                    for (UINT y = 0; y < c.rows; ++y)
                        std::fwrite(base + (UINT64{zslice} * c.rows + y) * c.footprint.Footprint.RowPitch, 1, static_cast<std::size_t>(c.row_bytes), f);
                ok = std::fclose(f) == 0;
            }
            c.readback->Unmap(0, &none);
        }
        if (index) std::fprintf(index, "%s%s\n", c.line.c_str(), ok ? "" : " (write failed)");
    }
    if (index) std::fclose(index);
    log("x2w-cxdump %016llx frame %ld: %zu buffers and %zu texture copies written to %s\\%s.*", static_cast<unsigned long long>(p.key), p.frame,
        p.buffers.size(), p.copies.size(), dir.c_str(), stem);
    p = Pending{};
}

}  // namespace cx_dump
