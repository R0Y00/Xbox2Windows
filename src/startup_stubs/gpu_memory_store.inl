// Included inside vbind after memory_window and the heap/binding helpers.
x2w_vmemory::Arenas& builder_arenas()
{
    static auto* arenas = new x2w_vmemory::Arenas;
    return *arenas;
}
bool directory_only_builder(const x2w_gpu::GpuShader& sh)
{
    if (!sh.builder || !sh.root_cbvs.empty()) return false;
    for (auto i : sh.root_srvs) if (i != abi::slot_memory) return false;
    for (auto i : sh.root_uavs) if (i != abi::slot_faults) return false;
    for (const auto& u : sh.used)
        if (u.root != GpuRootType::descriptor && u.root != GpuRootType::constants) return false;
    return true;
}

// A fallback is limited to an allocation the title owns and a whole committed,
// readable page. The immutable upload snapshot belongs to this recording.
UINT64 cpu_page(Wrapper* self, x2w_vmemory::BuildRecording& build, u64 at, ID3D12Resource** source, const char*& why)
{
    using namespace x2w_vmemory;
    if (build.cpu_pages >= 512) { why = "cpu-page-limit"; return 0; }
    if (!build.took_reservations) {
        using Ranges = SIZE_T(WINAPI*)(UINT_PTR*,SIZE_T);
        static const auto ranges = reinterpret_cast<Ranges>(GetProcAddress(GetModuleHandleW(L"xmem.dll"),"x2w_xmem_reserved_ranges"));
        if (ranges) {
            build.reservations.resize(2*4096);
            const auto n = ranges(build.reservations.data(),4096);
            build.reservations.resize(2*(std::min<size_t>)(n,4096));
        }
        build.took_reservations = true;
    }
    u64 first = 0, last = 0;
    bool owned = (known_private(at,&first,&last) || buffer_extent(at,&first,&last,nullptr,page_bytes)) && last-at >= page_bytes;
    for (size_t i = 0; !owned && i+1 < build.reservations.size(); i+=2) {
        const auto start = build.reservations[i], len = build.reservations[i+1];
        owned = len >= page_bytes && at >= start && at-start <= len-page_bytes;
    }
    if (!owned) {
        std::lock_guard<std::mutex> guard(builder_upload_lock);
        owned = std::any_of(builder_upload_blocks.begin(),builder_upload_blocks.end(),
                            [&](u64 base){return at>=base && at-base <= (64ull<<20)-page_bytes;});
    }
    if (!owned && self->own().list_uploads)
        for (const auto& p : self->own().list_uploads->pages) {
            const auto base = reinterpret_cast<u64>(p->cpu);
            if (p->size >= page_bytes && at >= base && at-base <= p->size-page_bytes) { owned = true; break; }
        }
    MEMORY_BASIC_INFORMATION info{};
    if (!owned) { why = "unowned-page"; return 0; }
    why = "unreadable-page";
    if (VirtualQuery(reinterpret_cast<void*>(at),&info,sizeof(info)) != sizeof(info) ||
        info.State != MEM_COMMIT || (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) ||
        at - reinterpret_cast<u64>(info.BaseAddress) + page_bytes > info.RegionSize) return 0;
    ComPtr<ID3D12Resource> up;
    why = "cpu-snapshot-allocation";
    if (!buffer(device_of(the_device),page_bytes,D3D12_HEAP_TYPE_UPLOAD,false,&up)) return 0;
    void* data = nullptr;
    D3D12_RANGE none{};
    if (FAILED(up->Map(0,&none,&data))) return 0;
    SIZE_T got = 0;
    why = "cpu-snapshot-read";
    const bool copied = ReadProcessMemory(GetCurrentProcess(),reinterpret_cast<void*>(at),data,page_bytes,&got) && got == page_bytes;
    up->Unmap(0,nullptr);
    if (!copied) return 0;
    *source = up.Detach();
    ++build.cpu_pages;
    return reinterpret_cast<u64>(info.AllocationBase);
}

bool builder_pages(Wrapper* self, ID3D12GraphicsCommandList* list, GpuListState& s,
                   const Bound& b, const x2w_gpu::GpuShader& shader, x2w_vmemory::Recording& m,
                   std::vector<x2w_vmemory::CopyBack>& pending)
{
    using namespace x2w_vmemory;
    auto& arenas = builder_arenas();
    const char* cause = nullptr;
    u64 fault_address = 0;
    auto reject = [&](const char* reason, u64 at) {
        if (!cause) { cause = reason; fault_address = at; }
        return false;
    };
    auto report = [&] {
        static LONG refusals;
        const auto n = InterlockedIncrement(&refusals);
        if (n <= 8 || (n & (n - 1)) == 0)
            log("x2w-gpu arena refusal: frame=%ld count=%ld key=%s cause=%s address=0x%llx pages=%zu windows=%zu updates=%u",
                frame_number, n, shader.key, cause, fault_address, m.addresses.size(), m.windows.size(), m.updates);
        return false;
    };
    if (!arenas.create(device_of(the_device))) { reject("device-setup", 0); return report(); }
    if (!m.build) m.build = std::make_shared<BuildRecording>();
    auto& build = *m.build;
    const auto state = [&](auto* resource, auto next) { require_state(self, resource, next); };
    const auto view = [&](UINT index, auto* resource, UINT64 bytes) { gpu_heap_raw_uav_at(m.heap, index, resource, 0, bytes); };
    std::vector<std::pair<u64,u64>> small_outputs;
    if (shader.builder) {
        for (const auto& u : shader.used) {
            const auto* root = s.roots.at(true, u.slot);
            if (u.kind != GpuResourceType::uav || u.root != GpuRootType::descriptor || !root || !root->address) continue;
            u64 start = 0, end = 0;
            const Wrapper* owner = nullptr;
            if (!buffer_extent(root->address, &start, &end, &owner)) continue; // outside a buffer: the arena takes its writes
            arenas.open(owner, true);
            if (end - start <= (256u << 10)) {
                if ((start | end) & 3) { reject("output-not-dword-aligned", root->address); return report(); }
                small_outputs.emplace_back(start,end);
            }
        }
    }
    const auto epoch = arenas.grant_epoch();
    if (build.grant_epoch != epoch) { m.ranges.clear(); build.grant_epoch = epoch; }
    auto page = [&](UINT n) -> PageChoice* {
        if (!title_page(n) || build.unavailable_pages.count(n)) return nullptr;
        auto it = build.pages.find(n);
        const u64 at = u64{n} << 16;
        if (it == build.pages.end()) {
            PageChoice selected;
            UINT64 cpu = 0;
            ComPtr<ID3D12Resource> snapshot;
            if (!memory_window(at, &selected.source, &selected.source_offset)) {
                const char* why = nullptr;
                cpu = cpu_page(self,build,at,&snapshot,why);
                if (!cpu) { build.unavailable_pages.emplace(n, why); return nullptr; }
                selected.source = snapshot.Get();
                selected.source_offset = 0;
            }
            u64 start = 0, end = 0;
            const Wrapper* owner = nullptr;
            if (!cpu && buffer_extent(at, &start, &end, &owner, page_bytes) && end - at >= page_bytes) selected.owner = owner;
            else {
                bool full = false;
                if (!arenas.place(n, selected.source, selected.source_offset, selected.overlay, cpu, &full)) {
                    build.unavailable_pages.emplace(n, full ? "arena-page-limit" : "arena-allocation");
                    return nullptr;
                }
            }
            it = build.pages.emplace(n,std::move(selected)).first;
        }
        auto& p = it->second;
        MapError error{};
        if (p.owner) {
            m.map_to(list,n,p.source,p.source_offset,shader.builder && arenas.open(p.owner,false),view,state,false,&error);
        } else {
            if (m.refreshed.insert(n).second) {
                arenas.merge(list,p.overlay,0,page_bytes,0,state);
                s.host_roots[1] = nullptr;
                s.host_pso = nullptr;
            }
            m.map_to(list,n,p.overlay.arena->bytes.Get(),p.overlay.offset,shader.builder,view,state,true,&error);
        }
        if (error != MapError::none) {
            static constexpr const char* errors[] = {"", "invalid-page", "invalid-window-view", "descriptor-slots", "directory-updates"};
            reject(errors[unsigned(error)], at);
        }
        return &p;
    };
    const auto missing = [&](UINT n, const char* fallback) {
        auto it = build.unavailable_pages.find(n);
        return it == build.unavailable_pages.end() ? fallback : it->second;
    };
    const auto span = [&](u64 at,u64 len) {
        if (at < (1ull << 32) || at >= (u64{abi::memory_pages - 1} << 16) || !len) return;
        const auto first = at >> 16, last = (at + len - 1) >> 16;
        if (m.ranges.insert((first << 32) | last | (shader.builder ? 1ull<<63 : 0)).second)
            for (auto n = first; n <= last; ++n) page(UINT(n));
    };
    // A checked packed header covers its actual structure. Re-seed just those bytes
    // if the GPU sees a new header at submission, or the last use was >8 frames ago.
    const auto packed = [&](u64 at) {
        if ((at & 255) || !m.packed_seen.insert(at).second) return;
        UINT64 header = 0;
        SIZE_T got = 0;
        if (!ReadProcessMemory(GetCurrentProcess(),reinterpret_cast<void*>(at),&header,8,&got) || got != 8) return;
        const UINT len = packed_bytes(header);
        if (!len) return;
        auto* start = page(UINT(at >> 16));
        if (!start) { reject(missing(UINT(at >> 16), "packed-missing-page"), at); return; }
        if (!start->overlay.arena) return;
        const UINT slot = arenas.packed_slot(at);
        if (slot == UINT32_MAX) { reject("packed-header-slots", at); return; }
        const auto first = start->overlay;
        arenas.run(list,first,UINT(first.source_offset + (at & 65535)),slot*16,len,UINT32_MAX,frame_number,state);
        s.host_roots[1] = nullptr;
        s.host_pso = nullptr;
        for (u64 p = at >> 16; p <= (at + len - 1) >> 16; ++p) {
            auto* c = page(UINT(p));
            if (!c || !c->overlay.arena) { reject(c ? "packed-crosses-owned-page" : missing(UINT(p), "packed-missing-page"), p << 16); return; }
            const u64 base = p << 16;
            const UINT lo = UINT((std::max)(base,at)-base);
            const UINT hi = UINT((std::min)(base+page_bytes,at+len)-base);
            arenas.merge(list,c->overlay,lo,hi-lo,slot+2,state);
        }
    };
    const auto in_upload = [&](u64 address) {
        u64 start = 0, size = 0;
        {
            std::lock_guard<std::mutex> guard(builder_upload_lock);
            auto it = builder_upload_spans.upper_bound(address);
            if (it != builder_upload_spans.begin()) {
                --it;
                if (address - it->first < it->second) { start = it->first; size = it->second; }
            }
        }
        if (!start && self->own().list_uploads)
            for (const auto& a : self->own().list_uploads->allocations)
                if (address >= a.cpu && address-a.cpu < a.bytes) { start = a.cpu; size = a.bytes; break; }
        if (!start || !build.scanned_uploads.insert(start).second) return;
        // These legacy upload blocks are process-owned and never unmapped.
        for (auto* words = reinterpret_cast<const u64*>(start); size >= 8; ++words, size -= 8) {
            const auto value = *words;
            if (!(value & 255) && title_page(value >> 16)) page(UINT(value >> 16));
        }
    };
    for (const auto& root : s.roots.compute) {
        if (shader.builder && x2w_vbind::is_view(root.kind) && title_page(root.address >> 16)) {
            packed(root.address);
            in_upload(root.address);
            span(root.address,root.kind == GpuRootKind::uav ? 4ull << 20 : 256ull << 10);
        }
        if (root.kind != GpuRootKind::constants) continue;
        for (size_t i = 0; i < root.dwords.size(); ++i) {
            const UINT lo = root.dwords[i];
            if (i+1 < root.dwords.size() && lo >= 65536 && root.dwords[i+1] >= 1 && root.dwords[i+1] <= 31)
                span((u64{root.dwords[i+1]} << 32) | lo,1);
            if (shader.builder && lo >= 0x01000000 && lo < 0x20000000) span(u64{lo} << 8,1);
        }
    }
    for (const auto& u : shader.user_data)
        if (u.root == GpuRootType::other && u.other == abi::user_view_address)
            span(buffer_address(b,s,true,u.slot,u.dword),256ull << 10);
    if (!m.faults_seeded) {
        memory_fault_pages.seed(frame_number, [&](UINT n) { page(n); return m.addresses.count(n) != 0; });
        m.faults_seeded = true;
    }
    for (const auto& out : small_outputs) {
        for (u64 n = out.first >> 16; n <= (out.second-1) >> 16; ++n) {
            auto* c = page(UINT(n));
            if (!c) { reject(missing(UINT(n), "output-missing-page"), n << 16); continue; }
            if (c->owner) continue;
            if (c->overlay.cpu_identity) { reject("output-needs-cpu-copyback", n << 16); continue; }
            const u64 base = n << 16;
            const UINT lo = UINT((std::max)(base,out.first)-base);
            const UINT hi = UINT((std::min)(base+page_bytes,out.second)-base);
            pending.push_back({c->overlay,lo,hi-lo});
        }
    }
    m.begin(list,state);
    if (m.writable_mode && m.addresses.empty()) reject("store-with-empty-directory", 0);
    return cause ? report() : true;
}
