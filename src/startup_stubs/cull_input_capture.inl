// LOCAL (diagnostic): X2W_CULL_INPUT_CAPTURE. In the traced frame (X2W_D3D12_TRACE_FRAME), for selected translated
// compute dispatches (key and occurrence), copy what the dispatch reads, as the GPU sees it at that point of the list,
// into read-back buffers, and write them with an index next to the trap log (<trap dir>/cullcap/). Nothing is written
// back to the title or changes what the dispatch does; the copies only add barriers and copies around it.
//
//   X2W_CULL_INPUT_CAPTURE=<key>:<n>[,...]      the n-th (0-based) translated dispatch of <key> in the traced frame,
//                                               in recording order over all threads; <key>:* = every one (max 8)
//   X2W_CULL_CAPTURE_MAX_MB=<m>                 cap per buffer copy (default 256)
//   X2W_CULL_CAPTURE_FRAMES=<n>                 frames to capture (default 1: the first traced frame with a selection)
//   X2W_CULL_CAPTURE_TOTAL_MB=<m>               cap on all copies of one frame (default 1024; logged when hit)
//   X2W_CULL_CAPTURE_EXTRA=<addr>:<bytes>[,...] more title ranges (hex address, decimal or 0x bytes) copied before
//                                               every selected dispatch (for reads the replay reports as unmapped)
//
// Before the dispatch: every bound table buffer (manifest resource; title address, bytes, stride, V# word 3), every
// bound texture (all mips of the resource), the indirect record the dispatch consumes, and the EXTRA ranges. For a
// heap shader (SM 6.6 heap): the heapmem view the shader reads (GPU copy of its window range), the vinfo words, and
// for every entry constant of kind 3 that names a heap slot holding a view token, the buffer the raw view installed
// for that token addresses (copied from the install's own window and offset: exactly what the shader reads), with
// the install's state. SRV and heap buffers above 1 MiB are copied once per frame per range (later dispatches get a
// "dup" line naming the earlier file: ASSUMED unchanged in between); UAV buffers are copied for every dispatch.
// CPU side, at record time: the root constants, the title view heap's bytes, the root tables and every view record
// (id -> address, bytes, descriptor words).
// After the dispatch (behind the UAV barrier): every bound UAV buffer up to 4 MiB (the X records, the list).
// Index: cullcap/index.tsv, one line per blob: frame, key, occ, phase, kind, label, title address, bytes, file, extra.
namespace cull_capture {

using Microsoft::WRL::ComPtr;

struct Want { u64 key; int occ; };  // occ -1: every occurrence (at most 8)

struct Blob {
    ComPtr<ID3D12Resource> readback;
    ID3D12GraphicsCommandList* list = nullptr;
    std::string name, extra;   // file name, index columns after the file
    std::string head;          // index columns before the file
    u64 bytes = 0;             // bytes to write
    // texture mip: rows are copied with a pitch; written tight
    UINT rows = 0;
    u64 row_bytes = 0, row_pitch = 0, offset = 0;
    ComPtr<ID3D12Fence> fence;
    bool submitted = false, live = true;
};

struct State {
    std::mutex lock;
    std::vector<Want> wants;
    std::vector<std::pair<u64, u64>> extra;
    u64 max_bytes = 256ull << 20;
    u64 total_max = 1024ull << 20, frame_bytes = 0;
    bool total_logged = false;
    int frames_max = 1, frames_done = 0;
    bool frame_had = false;
    LONG frame = -1;
    std::vector<std::pair<u64, int>> seen;   // key -> occurrences counted in `frame`
    struct Large { u64 address, bytes; std::string name; };
    std::vector<Large> large;               // > 1 MiB SRV / heap copies made in `frame`
    std::vector<Blob> blobs;
    std::wstring dir;
    bool views_written_frame = false;
    unsigned captured = 0;
};

inline u64 parse_number(const char* s, char** end)
{
    if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) return std::strtoull(s + 2, end, 16);
    return std::strtoull(s, end, 10);
}

State* cc_state()
{
    static State* const s = [] () -> State* {
        char value[1024]{};
        const DWORD n = GetEnvironmentVariableA("X2W_CULL_INPUT_CAPTURE", value, sizeof(value));
        if (!n || n >= sizeof(value) || std::strcmp(value, "0") == 0) return nullptr;
        auto* st = new State;
        const std::string spec(value);
        for (std::size_t at = 0; at < spec.size();) {
            const std::size_t end = (std::min)(spec.find(',', at), spec.size());
            const std::string item = spec.substr(at, end - at);
            const std::size_t colon = item.find(':');
            if (colon == 16) {
                const u64 key = std::strtoull(item.substr(0, 16).c_str(), nullptr, 16);
                const std::string occ = item.substr(17);
                st->wants.push_back({key, occ == "*" ? -1 : std::atoi(occ.c_str())});
            } else {
                log("x2w-cullcap: ignored selection '%s' (want <16 hex key>:<n|*>)", item.c_str());
            }
            at = end + 1;
        }
        if (GetEnvironmentVariableA("X2W_CULL_CAPTURE_MAX_MB", value, sizeof(value)))
            st->max_bytes = (std::max<u64>)(1, std::strtoull(value, nullptr, 10)) << 20;
        if (GetEnvironmentVariableA("X2W_CULL_CAPTURE_FRAMES", value, sizeof(value)))
            st->frames_max = (std::max)(1, std::atoi(value));
        if (GetEnvironmentVariableA("X2W_CULL_CAPTURE_TOTAL_MB", value, sizeof(value)))
            st->total_max = (std::max<u64>)(1, std::strtoull(value, nullptr, 10)) << 20;
        if (const DWORD m = GetEnvironmentVariableA("X2W_CULL_CAPTURE_EXTRA", value, sizeof(value)); m && m < sizeof(value)) {
            char* p = value;
            while (*p) {
                char* e = nullptr;
                const u64 address = std::strtoull(p, &e, 16);
                if (e == p || *e != ':') break;
                const u64 bytes = parse_number(e + 1, &e);
                if (address && bytes) st->extra.emplace_back(address, bytes);
                if (*e != ',') break;
                p = e + 1;
            }
        }
        wchar_t trap[MAX_PATH]{};
        const DWORD length = GetEnvironmentVariableW(L"X2W_TRAP_LOG", trap, MAX_PATH);
        if (length && length < MAX_PATH) {
            st->dir.assign(trap, length);
            st->dir.resize(st->dir.find_last_of(L"/\\") + 1);
        }
        st->dir += L"cullcap\\";
        CreateDirectoryW(st->dir.c_str(), nullptr);
        std::string keys;
        for (const Want& w : st->wants) {
            char t[40];
            std::snprintf(t, sizeof(t), "%s%016llx:%s", keys.empty() ? "" : ",", w.key, w.occ < 0 ? "*" : std::to_string(w.occ).c_str());
            keys += t;
        }
        log("x2w-cullcap: armed for the traced frame: %s; max %llu MiB per buffer; %zu extra ranges", keys.c_str(),
            st->max_bytes >> 20, st->extra.size());
        return st;
    }();
    return s;
}

// The occurrence of `key` this dispatch is (>= 0) when it is selected, else -1. Counts every translated dispatch of a
// wanted key in the traced frame, selected or not.
int pick(u64 key)
{
    State* st = cc_state();
    if (!st || !trace_frame_active) return -1;
    std::lock_guard<std::mutex> guard(st->lock);
    if (st->frame != frame_number) {
        st->frame = frame_number;
        st->seen.clear();
        st->large.clear();
        st->views_written_frame = false;
        st->frame_bytes = 0;
        st->total_logged = false;
        if (st->frame_had) ++st->frames_done;
        st->frame_had = false;
    }
    bool wanted = false;
    for (const Want& w : st->wants) wanted |= w.key == key;
    if (!wanted) return -1;
    int occ = 0;
    bool found = false;
    for (auto& s : st->seen)
        if (s.first == key) { occ = s.second++; found = true; }
    if (!found) st->seen.emplace_back(key, 1);
    if (st->frames_done >= st->frames_max) return -1;
    for (const Want& w : st->wants)
        if (w.key == key && (w.occ == occ || (w.occ < 0 && occ < 8))) {
            st->frame_had = true;
            return occ;
        }
    return -1;
}

std::string hex(u64 v)
{
    char t[24];
    std::snprintf(t, sizeof(t), "0x%llx", v);
    return t;
}

ComPtr<ID3D12Resource> readback_buffer(u64 bytes)
{
    ComPtr<ID3D12Resource> r;
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = bytes ? bytes : 16;
    d.Height = d.DepthOrArraySize = d.MipLevels = d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    auto* device = device_of(the_device);
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&r))))
        return nullptr;
    x2w_budget::committed(device, r.Get(), D3D12_HEAP_TYPE_READBACK);
    return r;
}

struct Ctx {
    State* st;
    ID3D12GraphicsCommandList* list;
    u64 key;
    int occ;
    const char* phase;
    std::string prefix;  // file name prefix
    std::string head;    // index columns frame..phase
};

void index_line(State* st, const std::string& line);

// GPU copy of [at, at + bytes) of desktop buffer `window` (in UNORDERED_ACCESS on the list, as every window the
// translated paths bind), which holds title range [address, address + bytes).
void copy_window(Ctx& c, const std::string& label, ID3D12Resource* window, u64 at, u64 bytes, u64 address, const std::string& extra, bool once_large)
{
    if (!window || !bytes) return;
    bool clipped = false;
    if (bytes > c.st->max_bytes) bytes = c.st->max_bytes, clipped = true;
    const std::string name = c.prefix + "-" + label + ".bin";
    if (once_large && bytes > (1u << 20)) {
        for (const auto& l : c.st->large)
            if (l.address == address && l.bytes == bytes) {
                index_line(c.st, c.head + "\tdup\t" + label + "\t" + hex(address) + "\t" + std::to_string(bytes) + "\t" + l.name + "\t" + extra +
                                     ";copied earlier in this frame (ASSUMED unchanged since)");
                return;
            }
        c.st->large.push_back({address, bytes, name});
    }
    if (c.st->frame_bytes + bytes > c.st->total_max) {
        if (!c.st->total_logged) log("x2w-cullcap: X2W_CULL_CAPTURE_TOTAL_MB reached (%llu MiB); later copies of this frame skipped", c.st->total_max >> 20);
        c.st->total_logged = true;
        index_line(c.st, c.head + "\tskipped\t" + label + "\t" + hex(address) + "\t" + std::to_string(bytes) + "\t-\t" + extra + ";total cap");
        return;
    }
    c.st->frame_bytes += bytes;
    Blob b;
    b.readback = readback_buffer(bytes);
    if (!b.readback) {
        log("x2w-cullcap %016llx occ %d %s %s: read-back allocation of %llu bytes failed", c.key, c.occ, c.phase, label.c_str(), bytes);
        return;
    }
    D3D12_RESOURCE_BARRIER t{};
    t.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    t.Transition.pResource = window;
    t.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    t.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;  // as indirect_stage: windows rest in COMMON / UAV
    t.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    c.list->ResourceBarrier(1, &t);
    c.list->CopyBufferRegion(b.readback.Get(), 0, window, at, bytes);
    std::swap(t.Transition.StateBefore, t.Transition.StateAfter);
    c.list->ResourceBarrier(1, &t);
    b.list = c.list;
    b.bytes = bytes;
    b.name = name;
    b.head = c.head + "\tbuf\t" + label + "\t" + hex(address) + "\t" + std::to_string(bytes);
    b.extra = extra + (clipped ? ";clipped" : "");
    c.st->blobs.push_back(std::move(b));
}

// GPU copy of title range [address, address + bytes) through its desktop window (desktop_range: the window the
// table binding path binds).
void copy_range(Ctx& c, const std::string& label, u64 address, u64 bytes, const std::string& extra, bool once_large)
{
    if (!address || !bytes) return;
    if (bytes > c.st->max_bytes) bytes = c.st->max_bytes;
    ID3D12Resource* window = nullptr;
    u64 at = 0;
    if (!desktop_range(address, bytes, &window, &at)) {
        index_line(c.st, c.head + "\tmissing\t" + label + "\t" + hex(address) + "\t" + std::to_string(bytes) + "\t-\t" + extra + ";no desktop window");
        return;
    }
    copy_window(c, label, window, at, bytes, address, extra, once_large);
}

void copy_texture(Wrapper* self, Ctx& c, const char* label, ID3D12Resource* texture, bool uav, const std::string& extra)
{
    const D3D12_RESOURCE_DESC desc = texture->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.DepthOrArraySize != 1 || desc.SampleDesc.Count != 1) {
        log("x2w-cullcap %016llx occ %d %s %s: texture not a single 2D (dimension %d array %u)", c.key, c.occ, c.phase, label,
            desc.Dimension, desc.DepthOrArraySize);
        return;
    }
    auto* device = device_of(the_device);
    require_state(self, texture, D3D12_RESOURCE_STATE_COPY_SOURCE);
    for (UINT mip = 0; mip < desc.MipLevels; ++mip) {
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
        UINT rows = 0;
        UINT64 row_bytes = 0, total = 0;
        device->GetCopyableFootprints(&desc, mip, 1, 0, &fp, &rows, &row_bytes, &total);
        if (!total || total > c.st->max_bytes) continue;
        Blob b;
        b.readback = readback_buffer(total);
        if (!b.readback) continue;
        const D3D12_TEXTURE_COPY_LOCATION from{texture, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {mip}};
        D3D12_TEXTURE_COPY_LOCATION to{b.readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        to.PlacedFootprint = fp;
        c.list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        b.list = c.list;
        b.rows = rows;
        b.row_bytes = row_bytes;
        b.row_pitch = fp.Footprint.RowPitch;
        b.offset = fp.Offset;
        b.bytes = u64{rows} * row_bytes;
        b.name = c.prefix + "-" + label + "-mip" + std::to_string(mip) + ".bin";
        b.head = c.head + "\ttex\t" + label + "\t0\t" + std::to_string(b.bytes);
        b.extra = extra + ";mip=" + std::to_string(mip) + ";width=" + std::to_string(fp.Footprint.Width) + ";height=" +
                  std::to_string(fp.Footprint.Height) + ";mips=" + std::to_string(desc.MipLevels) + ";dxgi=" + std::to_string(desc.Format) +
                  ";row_bytes=" + std::to_string(row_bytes);
        c.st->blobs.push_back(std::move(b));
    }
    require_state(self, texture, uav ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}

bool write_file(const std::wstring& path, const void* data, std::size_t bytes)
{
    FILE* f = nullptr;
    if (_wfopen_s(&f, path.c_str(), L"wb") || !f) return false;
    const bool ok = !bytes || std::fwrite(data, 1, bytes, f) == bytes;
    return std::fclose(f) == 0 && ok;
}

void index_line(State* st, const std::string& line)
{
    FILE* f = nullptr;
    if (_wfopen_s(&f, (st->dir + L"index.tsv").c_str(), L"ab") || !f) return;
    std::fprintf(f, "%s\n", line.c_str());
    std::fclose(f);
}

std::wstring wide(const std::string& s) { return std::wstring(s.begin(), s.end()); }

// CPU-side blob, written now (title memory as the CPU holds it at record time).
void cpu_blob(Ctx& c, const char* label, u64 address, const void* data, std::size_t bytes, const std::string& extra)
{
    const std::string name = c.prefix + "-" + label + ".bin";
    const bool ok = write_file(c.st->dir + wide(name), data, bytes);
    index_line(c.st, c.head + "\tcpu\t" + label + "\t" + hex(address) + "\t" + std::to_string(bytes) + "\t" + (ok ? name : "WRITE-FAILED") + "\t" + extra);
}

// Every view record the runtime knows: id, slot, image (T#), title address, bytes, descriptor dwords 0..7 as the
// translated shaders see them (view_record_words; a buffer's address is filled in here from the record).
void write_view_records(Ctx& c)
{
    if (c.st->views_written_frame) return;
    c.st->views_written_frame = true;
    std::string out = "id\tslot\timage\taddress\tbytes\tw0\tw1\tw2\tw3\tw4\tw5\tw6\tw7\n";
    AcquireSRWLockShared(&note_lock);
    const std::vector<ViewRecord> records = view_records();  // processed outside note_lock (as fill_views does)
    ReleaseSRWLockShared(&note_lock);
    for (std::size_t id = 0; id < records.size(); ++id) {
        const ViewRecord& record = records[id];
        std::uint32_t v[8]{};
        bool image = false, unmapped = false;
        if (!view_record_words(record, v, &image, &unmapped)) continue;
        u64 address = 0, bytes = 0;
        const Wrapper* resource = as_wrapper(reinterpret_cast<const void*>(record.resource));
        const u64 base = resource && resource->own().title_memory ? reinterpret_cast<u64>(resource->own().title_memory) : 0;
        if (record.slot == 17 && record.desc_bytes >= 12) {
            std::uint32_t size = 0;
            std::memcpy(&address, record.desc, 8);
            std::memcpy(&size, record.desc + 8, 4);
            bytes = size;
        } else if (!image) {
            x2w_startup::BufferViewLayout l;
            if (base && x2w_startup::buffer_view_layout_from_desc(record.slot == 19, record.desc, record.desc_bytes, l, typed_views_off()))
                address = base + l.offset, bytes = l.bytes;
        } else {
            address = base;  // the T# base stays 0 (the replay places the image itself)
        }
        if (!image) {
            v[0] = static_cast<std::uint32_t>(address);
            v[1] = (v[1] & ~0xffffu) | (static_cast<std::uint32_t>(address >> 32) & 0xffff);
        }
        char line[260];
        std::snprintf(line, sizeof(line), "%zu\t%u\t%d\t0x%llx\t%llu\t%08x\t%08x\t%08x\t%08x\t%08x\t%08x\t%08x\t%08x\n", static_cast<std::size_t>(record.id), record.slot,
                      image ? 1 : 0, address, bytes, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
        out += line;
    }
    const std::string name = "f" + std::to_string(c.st->frame + 1) + "-view-records.tsv";
    const bool ok = write_file(c.st->dir + wide(name), out.data(), out.size());
    index_line(c.st, c.head + "\tcpu\tview-records\t0\t" + std::to_string(out.size()) + "\t" + (ok ? name : "WRITE-FAILED") + "\tid = word 1 of the heap view token (X2WV)");
}

}  // namespace cull_capture

// Called by execute_dispatch (d3d12_compute.inl) right before the dispatch is issued (pipeline, root and views bound;
// an indirect record already staged), with the manifest's resources as bound.
void cull_capture_before(Wrapper* self, ID3D12GraphicsCommandList* list, u64 key, int occ, const ComputeTranslation& tr,
                         const u64* res_address, const std::size_t* res_bytes, const std::uint32_t* strides, const std::uint32_t* word3,
                         Microsoft::WRL::ComPtr<ID3D12Resource>* textures, const std::uint32_t* values, UINT value_count,
                         const TitleHeapView* heap_view, u64 heap_address, UINT x, UINT y, UINT z)
{
    using namespace cull_capture;
    State* st = cc_state();
    if (!st || occ < 0) return;
    // the heap slots the kind-3 entry constants name, and their raw view installs, read before st->lock: draw_lock
    // (which guards raw_srv_installs) is held around submissions, where st->lock is taken too
    struct HeapUse { std::size_t ud; std::uint32_t index, id; RawSrvInstall install; };
    std::vector<HeapUse> heap_uses;
    if (tr.heap && heap_view && heap_view->window) {
        for (std::size_t i = 0; i < tr.uds.size() && i < value_count; ++i) {
            if (tr.uds[i].kind != 3) continue;
            const std::uint32_t index = values[i];
            const u64 byte_index = u64{index} << 5;
            if (byte_index + 8 > heap_view->bytes) continue;
            UINT words[2]{};
            const auto* at = reinterpret_cast<const void*>(heap_address + byte_index);
            if (!x2w_local::readable(at, sizeof(words))) continue;
            std::memcpy(words, at, sizeof(words));
            if (words[0] != view_token_magic || words[1] >= view_ids_max) continue;  // not a view token: not a heap index
            heap_uses.push_back({i, index, words[1], {}});
        }
        AcquireSRWLockExclusive(&draw_lock);
        for (HeapUse& u : heap_uses)
            if (u.id < raw_srv_installs.size()) u.install = raw_srv_installs[u.id];
        ReleaseSRWLockExclusive(&draw_lock);
    }
    std::lock_guard<std::mutex> guard(st->lock);
    const Pass& pass = pass_of(self);
    char prefix[96];
    std::snprintf(prefix, sizeof(prefix), "f%ld-%016llx-o%d", frame_number + 1, key, occ);
    Ctx c{st, list, key, occ, "before", prefix, ""};
    c.head = std::to_string(frame_number + 1) + "\t" + [&] { char k[20]; std::snprintf(k, sizeof(k), "%016llx", key); return std::string(k); }() +
             "\t" + std::to_string(occ) + "\tbefore";
    // dispatch facts
    std::string facts = "list=" + hex(reinterpret_cast<u64>(list)) + ";xyz=" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z) + ";trace_seq_before=" + std::to_string(dispatch_trace_seq) +
                        ";heap_address=" + hex(heap_address) + ";numthreads=" + std::to_string(tr.threads[0]) + "x" + std::to_string(tr.threads[1]) + "x" + std::to_string(tr.threads[2]);
    for (int p = 0; p < 16; ++p) {
        if (pass.compute_kind[p] == 1) {
            facts += ";p" + std::to_string(p) + "_constants=";
            for (UINT w = 0; w < pass.compute_count[p] && w < 16; ++w) facts += (w ? "," : "") + hex(pass.compute_constants[p][w]);
        } else if (pass.compute_kind[p]) {
            facts += ";p" + std::to_string(p) + "_kind" + std::to_string(pass.compute_kind[p]) + "=" + hex(pass.compute_address[p]);
        }
    }
    facts += ";entry=";
    for (UINT i = 0; i < value_count && i < 64; ++i) facts += (i ? "," : "") + hex(values[i]);
    if (IndirectPending* p = indirect_current; p && p->args_window) {
        // the indirect record this dispatch consumes (copied from the window as the ring copy read it)
        Blob b;
        b.readback = readback_buffer(16);
        if (b.readback) {
            D3D12_RESOURCE_BARRIER t{};
            t.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            t.Transition.pResource = p->args_window;
            t.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            t.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            t.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
            list->ResourceBarrier(1, &t);
            list->CopyBufferRegion(b.readback.Get(), 0, p->args_window, p->args_at, 12);
            std::swap(t.Transition.StateBefore, t.Transition.StateAfter);
            list->ResourceBarrier(1, &t);
            b.list = list;
            b.bytes = 12;
            b.name = std::string(prefix) + "-dispatch-args.bin";
            b.head = c.head + "\tbuf\tdispatch-args\t0\t12";
            b.extra = "stride=" + std::to_string(p->stride) + ";max_count=" + std::to_string(p->max_count);
            st->blobs.push_back(std::move(b));
        }
        facts += ";indirect=1";
    }
    index_line(st, c.head + "\tdispatch\tfacts\t0\t0\t-\t" + facts);
    log("x2w-cullcap %016llx occ %d frame %ld list %p: capturing (%s)", key, occ, frame_number + 1, list, facts.c_str());
    // table resources
    for (std::size_t i = 0; i < tr.res.size() && i < 64; ++i) {
        const auto& r = tr.res[i];
        char label[32];
        std::snprintf(label, sizeof(label), "%s%zu", r.uav ? "u" : "t", i);
        const std::string extra = "res=" + std::to_string(i) + ";uav=" + std::to_string(r.uav) + ";kind=" + std::to_string(r.kind) +
                                  ";source=" + std::to_string(r.source) + ";param=" + std::to_string(r.a) + ";offset=" + std::to_string(r.b) +
                                  ";stride=" + std::to_string(strides[i]) + ";word3=" + hex(word3[i]);
        if (r.kind == 0) copy_range(c, label, res_address[i], res_bytes[i], extra, !r.uav);
        else if (textures[i]) copy_texture(self, c, label, textures[i].Get(), r.uav, extra);
        else index_line(st, c.head + "\tmissing\t" + label + "\t0\t0\t-\t" + extra + ";not bound or not traced");
    }
    // heap shader: the heapmem view as the GPU reads it, the vinfo words, and the buffers behind the entry constants
    if (tr.heap && heap_view && heap_view->window) {
        copy_window(c, "heapmem", heap_view->window, heap_view->offset, heap_view->bytes, heap_address,
                    "title_heap_bytes=" + std::to_string(heap_view->heap_bytes) + ";view_bytes=" + std::to_string(heap_view->bytes), true);
        if (vinfo_words) cpu_blob(c, "vinfo", 0, vinfo_words, std::size_t{view_ids_max} * 16, "16 bytes per view id: stride, format code, 0, 0");
        for (const HeapUse& u : heap_uses) {
            const std::size_t i = u.ud;
            const std::uint32_t index = u.index;
            const std::uint32_t words[2] = {view_token_magic, u.id};
            const RawSrvInstall& install = u.install;
            const std::string label = "heap" + std::to_string(index);
            const std::string extra = "ud=" + std::to_string(i) + ";param=" + std::to_string(tr.uds[i].a) + ";word=" + std::to_string(tr.uds[i].b) +
                                      ";index=" + std::to_string(index) + ";view_id=" + std::to_string(words[1]) + ";install=" + install.state +
                                      ";install_address=" + hex(install.address) + ";install_bytes=" + std::to_string(install.bytes) +
                                      ";gpuva=" + hex(install.gpuva) + ";window_offset=" + hex(install.offset);
            if (install.window) copy_window(c, label, install.window, install.offset, install.bytes, install.address, extra, true);
            else index_line(st, c.head + "\tmissing\t" + label + "\t" + hex(install.address) + "\t" + std::to_string(install.bytes) + "\t-\t" + extra +
                                    ";no raw view installed (the shader reads a null view: zeros)");
        }
    }
    const TraceHeap heap = record_trace_heap(pass);
    if (heap.base && heap.bytes && x2w_local::readable(reinterpret_cast<const void*>(heap.base), static_cast<std::size_t>(heap.bytes)))
        cpu_blob(c, "title-heap", heap.base, reinterpret_cast<const void*>(heap.base), static_cast<std::size_t>(heap.bytes),
                 "count=" + std::to_string(heap.count) + ";increment=" + std::to_string(heap.increment));
    for (int p = 0; p < 16; ++p) {
        if (pass.compute_kind[p] != 5 || !pass.compute_address[p]) continue;
        const u64 table = pass.compute_address[p];
        std::size_t bytes = 0;  // the bytes the manifest reads from this table (entries are 32 bytes)
        for (const auto& r : tr.res)
            if (r.source == 0 && r.a == p) bytes = (std::max)(bytes, static_cast<std::size_t>(r.b) + 32);
        for (const auto& u : tr.uds)
            if (u.kind == 7 && u.a == p) bytes = (std::max)(bytes, static_cast<std::size_t>(u.b) + 4);
        bytes = bytes ? (bytes + 31) & ~std::size_t{31} : 0x200;
        if (x2w_local::readable(reinterpret_cast<const void*>(table), bytes)) {
            char label[24];
            std::snprintf(label, sizeof(label), "table-p%d", p);
            cpu_blob(c, label, table, reinterpret_cast<const void*>(table), bytes, "param=" + std::to_string(p));
        }
    }
    write_view_records(c);
    for (std::size_t i = 0; i < st->extra.size(); ++i) {
        char label[24];
        std::snprintf(label, sizeof(label), "extra%zu", i);
        copy_range(c, label, st->extra[i].first, st->extra[i].second, "", true);
    }
}

// After the dispatch's UAV barrier: the bound UAV buffers up to 4 MiB.
void cull_capture_after(Wrapper* self, ID3D12GraphicsCommandList* list, u64 key, int occ, const ComputeTranslation& tr,
                        const u64* res_address, const std::size_t* res_bytes)
{
    using namespace cull_capture;
    State* st = cc_state();
    if (!st || occ < 0) return;
    std::lock_guard<std::mutex> guard(st->lock);
    char prefix[96];
    std::snprintf(prefix, sizeof(prefix), "f%ld-%016llx-o%d-after", frame_number + 1, key, occ);
    char k[20];
    std::snprintf(k, sizeof(k), "%016llx", key);
    Ctx c{st, list, key, occ, "after", prefix, std::to_string(frame_number + 1) + "\t" + k + "\t" + std::to_string(occ) + "\tafter"};
    for (std::size_t i = 0; i < tr.res.size() && i < 64; ++i) {
        const auto& r = tr.res[i];
        if (!r.uav || r.kind != 0 || res_bytes[i] > (4u << 20)) continue;
        char label[32];
        std::snprintf(label, sizeof(label), "u%zu", i);
        copy_range(c, label, res_address[i], res_bytes[i], "res=" + std::to_string(i), false);
    }
    ++st->captured;
    (void)self;
}

bool cull_capture_armed(u64 key)
{
    using namespace cull_capture;
    const State* st = cc_state();
    if (!st || !trace_frame_active) return false;
    for (const Want& w : st->wants)
        if (w.key == key) return true;
    return false;
}
int cull_capture_pick(u64 key) { return cull_capture::pick(key); }
bool cull_capture_on() { return cull_capture::cc_state() != nullptr; }

// After ExecuteCommandLists: a fence for the blobs recorded on these lists.
void cull_capture_submitted(ID3D12CommandQueue* queue, ID3D12CommandList* const* lists, UINT count)
{
    using namespace cull_capture;
    State* st = cc_state();
    if (!st) return;
    std::lock_guard<std::mutex> guard(st->lock);
    ComPtr<ID3D12Fence> fence;
    for (Blob& b : st->blobs) {
        if (b.submitted || !b.live || std::find(lists, lists + count, static_cast<ID3D12CommandList*>(b.list)) == lists + count) continue;
        if (!fence) {
            if (FAILED(device_of(the_device)->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))) || FAILED(queue->Signal(fence.Get(), 1))) {
                log("x2w-cullcap: fence failed; blobs of this submission dropped");
                fence.Reset();
                return;
            }
        }
        b.fence = fence;
        b.submitted = true;
    }
}

// A list reset before it was submitted: its blobs will never be filled.
void cull_capture_discard(ID3D12GraphicsCommandList* list)
{
    using namespace cull_capture;
    State* st = cc_state();
    if (!st) return;
    std::lock_guard<std::mutex> guard(st->lock);
    for (Blob& b : st->blobs)
        if (b.list == list && !b.submitted) b.live = false;
}

// At present: write the blobs whose fence passed.
void cull_capture_flush()
{
    using namespace cull_capture;
    State* st = cc_state();
    if (!st) return;
    std::lock_guard<std::mutex> guard(st->lock);
    std::vector<Blob> keep;
    for (Blob& b : st->blobs) {
        if (!b.live) {
            index_line(st, b.head + "\tDISCARDED\t" + b.extra + ";list reset before submission");
            continue;
        }
        if (!b.submitted || b.fence->GetCompletedValue() < 1) {
            keep.push_back(std::move(b));
            continue;
        }
        void* memory = nullptr;
        const D3D12_RANGE none{};
        bool ok = false;
        if (SUCCEEDED(b.readback->Map(0, nullptr, &memory))) {
            const auto* base = static_cast<const unsigned char*>(memory);
            if (b.rows) {
                std::vector<unsigned char> tight(static_cast<std::size_t>(b.bytes));
                for (UINT y = 0; y < b.rows; ++y)
                    std::memcpy(tight.data() + y * b.row_bytes, base + b.offset + y * b.row_pitch, static_cast<std::size_t>(b.row_bytes));
                ok = write_file(st->dir + wide(b.name), tight.data(), tight.size());
            } else {
                ok = write_file(st->dir + wide(b.name), base, static_cast<std::size_t>(b.bytes));
            }
            b.readback->Unmap(0, &none);
        }
        index_line(st, b.head + "\t" + (ok ? b.name : "WRITE-FAILED") + "\t" + b.extra);
        if (b.name.find("dispatch-args") != std::string::npos && ok) {
            std::uint32_t xyz[3]{};
            void* m = nullptr;
            if (SUCCEEDED(b.readback->Map(0, nullptr, &m))) {
                std::memcpy(xyz, m, 12);
                b.readback->Unmap(0, &none);
            }
            log("x2w-cullcap %s: dispatch args %u %u %u", b.name.c_str(), xyz[0], xyz[1], xyz[2]);
        }
    }
    const std::size_t written = st->blobs.size() - keep.size();
    st->blobs.swap(keep);
    if (written) log("x2w-cullcap: wrote %zu blobs (%zu pending) to cullcap\\index.tsv", written, st->blobs.size());
}
