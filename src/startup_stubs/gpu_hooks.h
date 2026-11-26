// Hook declarations for translated shaders. Include after Wrapper inside the device shim's anonymous namespace.
// Shared types live in gpu_abi.h. Definitions follow the shim helpers so they can use desktop_range,
// upload_texture and desktop_format.
//
// Heap, descriptor, copy and list lifecycle hooks augment existing handlers. Copy passes, clears, barriers and
// queries may replace the bound heaps, root signature and PSO. Refresh these before the next draw or dispatch.
// Pipeline creation and draw, dispatch, mesh and indirect calls use exclusive handlers that count refusals.
//
// Different command lists record concurrently. Unless indicated, callers hold no shim lock. Wrapper state uses
// own().gpu[0] for pipelines, descriptor heaps or list bindings, and [1] for indirect list state.
// Last-reference release dispatches cleanup by object kind.
using x2w_gpu::GpuCreateDesc;
using x2w_gpu::GpuFound;
using x2w_gpu::GpuHeapInfo;
using x2w_gpu::GpuHostHeap;
using x2w_gpu::GpuPipeline;
using x2w_gpu::GpuRootKind;
using x2w_gpu::GpuTargets;
using x2w_gpu::GpuViewInfo;

// ----- WP0 (gpu_hooks.inl) ---------------------------------------------------------------------------------
// The release of the last reference of any wrapper whose own().gpu[0] or [1] is set; forwards by kind to
// gpu_pipeline_released / gpu_heap_released / gpu_list_released.
void gpu_released(Wrapper* object);
// G0 / tests: a hook counter (x2w_gpu::GpuCounter).
long long gpu_count(int counter);

// -----
// WP2 vpipe (gpu_pipeline.inl) -------------------------------------------------------------------------
// Device slots 10, 11, 62, 86 under ABI (EXCLUSIVE). `pipeline` is the placeholder wrapper WP0 already made and
// returns to the title with S_OK whatever this returns. Records the stages and an owned copy of the state; makes
// nothing (pipelines are made lazily). Returns false when nothing was kept.
bool gpu_create_pipeline(Wrapper* pipeline, const GpuCreateDesc& desc);
// At draw / dispatch / mesh / indirect (WP3, WP4): the made pipeline for the bound targets (null `targets`: the
// stream's own formats), made synchronously on first use; null while it cannot be made (the draw is dropped).
enum class GpuPsoStatus { unavailable, compiling, missing_shader, warm_bad, worker_failure, refused, count };
struct GpuPsoDrop {
    GpuPsoStatus status = GpuPsoStatus::unavailable;
    const char* detail = "";
};
const GpuPipeline* gpu_pipeline_ready(Wrapper* pipeline, const GpuTargets* targets, GpuPsoDrop* drop = nullptr);
void gpu_pipeline_released(Wrapper* pipeline);

// ----- WP5 rawtwins (gpu_heap.inl, gpu_views.inl) -------------------------------------------------------
// Device slot 14 (ADDITIVE): after X2W made the desktop heap and its title memory. `heap` is the new wrapper (out),
// `title_cpu` the title descriptor memory (0 when none), `title_gpu` the GPU start handed to the title (0: not
// shader visible). Under ABI the title stride is abi::view_increment / sampler_increment for those types.
void gpu_heap_created(Wrapper* device, Wrapper* heap, const D3D12_DESCRIPTOR_HEAP_DESC& desc, std::uint64_t title_cpu,
                          std::uint64_t title_gpu);
void gpu_heap_released(Wrapper* heap);
// Device view slots (ADDITIVE, after X2W's view_slot / memory handlers and their X2WV token write; for 70/71 the title
// slot then holds X2W's X2WV token or the GFX10 V# words of write_buffer_descriptor, as without the switch). The raw title
// arguments as passed (wrapped pointers), by slot:
//   17 CBV (desc, dest)  18 SRV (resource, desc, dest)  19 UAV (resource, counter, desc, dest)
//   20 RTV (resource, desc, dest)  21 DSV (resource, desc, dest)  22 sampler (desc, dest)
//   61 console sampler (desc [27 words], dest)  70 placed SRV (resource desc, view desc+address, dest)
//   71 placed UAV (resource desc, view desc+address at +0x28, dest)
void gpu_heap_view(Wrapper* device, int slot, std::uint64_t a, std::uint64_t b, std::uint64_t c, std::uint64_t d);
// Device slots 24 and 23 (ADDITIVE, after X2W's desktop copies; title CPU handles).
void gpu_heap_copy_simple(Wrapper* device, UINT count, std::uint64_t dest, std::uint64_t src, D3D12_DESCRIPTOR_HEAP_TYPE type);
void gpu_heap_copy(Wrapper* device, UINT dest_count, const D3D12_CPU_DESCRIPTOR_HANDLE* dest_starts, const UINT* dest_sizes,
                       UINT src_count, const D3D12_CPU_DESCRIPTOR_HANDLE* src_starts, const UINT* src_sizes,
                       D3D12_DESCRIPTOR_HEAP_TYPE type);
// Before each submission (called from gpu_list_submit): mirror the records the title byte-copied, upload changed
// texture slots (PLAN 4b). Runs with draw_lock and allocator_lock held.
void gpu_heap_submit(ID3D12CommandQueue* queue, UINT count, Wrapper* const* lists);
// Mirror heap operations used during shader binding.
bool gpu_heap_for(Wrapper* title_heap, GpuHeapInfo* out);       // a heap the title set (slot 32)
bool gpu_heap_at(std::uint64_t title_gpu_handle, GpuHeapInfo* out);  // the heap a handle lies in
bool gpu_heap_view_at(GpuHostHeap* heap, std::uint32_t index, GpuViewInfo* out);  // the title's view there
u64 gpu_heap_title_address_at(GpuHostHeap* heap, std::uint32_t index);
ID3D12Resource* gpu_heap_kept_at(GpuHostHeap* heap, std::uint32_t index);
void gpu_material_slots(GpuHostHeap* heap, std::vector<x2w_material_trace::Slot>& slots);
x2w_material_trace::Record gpu_material_record(GpuHostHeap* heap, UINT slot, UINT base, x2w_material::Mask fields);
unsigned gpu_materialize(Wrapper* list, GpuHostHeap* heap, const x2w_material_trace::Words& words, x2w_material::Mask fields);
bool gpu_depth_barrier(Wrapper* list, Wrapper* resource);
bool gpu_heap_mirror(GpuHostHeap* heap, std::uint32_t index, std::uint64_t title_handle, bool for_draw,
                         GpuViewInfo* out);  // make host slot `index` from the title slot
bool gpu_heap_dynamic(GpuHostHeap* heap, std::uint32_t n, std::uint32_t* first);
void gpu_heap_release(GpuHostHeap* heap, std::uint32_t first, std::uint32_t n);
bool gpu_heap_make_listed(GpuHostHeap* heap, std::uint32_t source, const std::uint32_t (&record)[8], std::uint32_t index,
                              GpuViewInfo* out);
void gpu_heap_make_null(GpuHostHeap* heap, std::uint32_t index);
void gpu_heap_raw_at(GpuHostHeap* heap, std::uint32_t index, ID3D12Resource* buffer, std::uint64_t offset, std::uint64_t bytes);
void gpu_heap_raw_uav_at(GpuHostHeap* heap, std::uint32_t index, ID3D12Resource* buffer, std::uint64_t offset,
                             std::uint64_t bytes);
void gpu_heap_null_raw_at(GpuHostHeap* heap, std::uint32_t index);
void gpu_heap_null_raw_uav_at(GpuHostHeap* heap, std::uint32_t index);
void gpu_heap_constant_at(GpuHostHeap* heap, std::uint32_t index, ID3D12Resource* buffer, std::uint64_t gpu, std::uint32_t bytes);
D3D12_GPU_DESCRIPTOR_HANDLE gpu_heap_gpu_handle(GpuHostHeap* heap, std::uint32_t index);
std::uint32_t gpu_heap_count(GpuHostHeap* heap);  // title slots
ID3D12DescriptorHeap* gpu_heap_host(GpuHostHeap* heap);
std::uint64_t gpu_heap_strides_gpu(GpuHostHeap* heap);  // the SLOT_STRIDES buffer's GPU address, 0: none
// Bindless textures (PLAN 4b): upload up to `max` textures WP5 has queued (made into mirror slots, or changed), with
// upload_texture on this open list, and move them to SRV | non-pixel SRV. Called by WP3 on the draw/dispatch path
// before the bind. False: nothing was due.
bool gpu_heap_upload_due(Wrapper* list, std::size_t max);

// ----- WP3 vbind (gpu_bind.inl, gpu_find.h) -------------------------------------------------------------
bool gpu_find(std::uint64_t address, std::uint64_t bytes, GpuFound* out);
// List setters 32-46 (EXCLUSIVE under ABI: X2W's handlers are not called, since they issue desktop root calls with
// the title's parameter indices; the parameter is unbounded). X2W's Pass root state is therefore stale under ABI.
void gpu_set_heaps(Wrapper* list, UINT count, void* const* heaps);                         // 32
void gpu_roots_signature(Wrapper* list, bool compute, Wrapper* root);                       // 33 / 34
void gpu_roots_table(Wrapper* list, bool compute, UINT parameter, std::uint64_t handle);    // 35 / 36
void gpu_roots_constants(Wrapper* list, bool compute, UINT parameter, UINT count, const std::uint32_t* data,
                             UINT offset);                                                      // 37 / 38 (count 1), 39 / 40
void gpu_roots_view(Wrapper* list, bool compute, UINT parameter, std::uint64_t address, GpuRootKind kind);  // 41-46
// Non-root draw state (ADDITIVE, after X2W's handler; slots that X2W does not install are installed under ABI).
void gpu_topology(Wrapper* list, UINT topology);                                            // 20 (console values)
void gpu_viewports(Wrapper* list, UINT count, const D3D12_VIEWPORT* viewports);             // 21
void gpu_scissors(Wrapper* list, UINT count, const D3D12_RECT* rects);                      // 22
void gpu_blend_factor(Wrapper* list, const float* factor);                                  // 23
void gpu_stencil_ref(Wrapper* list, UINT ref);                                              // 24
void gpu_set_pipeline(Wrapper* list, Wrapper* pipeline);                                    // 25
void gpu_barriers(Wrapper* list, UINT count, const D3D12_RESOURCE_BARRIER* barriers);       // 26
void gpu_ia_index(Wrapper* list, const D3D12_INDEX_BUFFER_VIEW* view);                      // 47
void gpu_ia_vertex(Wrapper* list, UINT start, UINT count, const D3D12_VERTEX_BUFFER_VIEW* views);  // 48
void gpu_targets(Wrapper* list, UINT count, const D3D12_CPU_DESCRIPTOR_HANDLE* rtvs, BOOL single,
                     const D3D12_CPU_DESCRIPTOR_HANDLE* dsv);                                    // 50 (title handles)
void gpu_depth_bounds(Wrapper* list, float min, float max);                                 // 70 (newer layout 68+2)
// Draws (EXCLUSIVE). `list` is recording(self) (null when the list is not open).
bool gpu_draw(Wrapper* self, ID3D12GraphicsCommandList* list, UINT vertices, UINT instances, UINT first_vertex,
                  UINT first_instance);                                                         // 12
bool gpu_draw_indexed(Wrapper* self, ID3D12GraphicsCommandList* list, UINT count, UINT instances, UINT start,
                          INT base_vertex, UINT first_instance);                                // 13
bool gpu_dispatch(Wrapper* self, ID3D12GraphicsCommandList* list, UINT x, UINT y, UINT z);  // 14
// The bind sequence without the issue, for WP4. False: the call would be dropped.
bool gpu_bind_draw(Wrapper* self, ID3D12GraphicsCommandList* list, bool indexed);
bool gpu_bind_mesh(Wrapper* self, ID3D12GraphicsCommandList* list);
bool gpu_bind_compute(Wrapper* self, ID3D12GraphicsCommandList* list);
// WP4's clamp pass set its own compute root and PSO: the next compute bind must set them again.
void gpu_bind_forget_compute(Wrapper* self);
// The made pipeline bound on the list (gpu_set_pipeline + gpu_pipeline_ready for the bound targets), or null;
// *title_topology = the console topology last set at slot 20 (17 = rectangles).
const GpuPipeline* gpu_bound_pipeline(Wrapper* self, UINT* title_topology, GpuPsoDrop* drop = nullptr);
// List lifecycle, WP3's part (dispatched from the gpu_list_* hooks below).
void gpu_bind_list_reset(Wrapper* list);
void gpu_bind_list_close(Wrapper* list);
void gpu_bind_submitted(ID3D12CommandQueue* queue, UINT count, Wrapper* const* lists);
void gpu_bind_first_submit(UINT count, Wrapper* const* lists);
void gpu_list_released(Wrapper* list);

// ----- WP4 vindirect (gpu_indirect.inl) ---------------------------------------------------------------------
// List slot 61 ExecuteIndirect (EXCLUSIVE; with X2W_GPU_INDIRECT=0 WP0 drops it without calling).
bool gpu_indirect(Wrapper* self, ID3D12GraphicsCommandList* list, void* signature, UINT max_count, Wrapper* arguments,
                      std::uint64_t offset, Wrapper* count_resource, std::uint64_t count_offset);
// List slot 129 DispatchMesh(x, y, z) (newer layout 127+2; EXCLUSIVE; X2W_GPU_MESH=0: dropped by WP0).
bool gpu_mesh(Wrapper* self, ID3D12GraphicsCommandList* list, UINT x, UINT y, UINT z);
// List slot 80 DispatchX(x, y, z, flags) (newer layout 78+2; EXCLUSIVE): flags & 0x20 = the counts are threads.
bool gpu_dispatch_x(Wrapper* self, ID3D12GraphicsCommandList* list, UINT x, UINT y, UINT z, UINT flags);
void gpu_indirect_list_reset(Wrapper* list);
void gpu_indirect_list_close(Wrapper* list);  // clears the per-list budget slot (a list may record without a reset)

// ----- WP0 dispatchers (gpu_hooks.inl) ----------------------------------------------------------------------
// List slot 139 Reset: after a successful reset (the desktop list is open). Calls bind then indirect reset.
void gpu_list_reset(Wrapper* list);
// List slot 9 Close: before X2W's close (the desktop list is still open). Calls bind then indirect close.
void gpu_list_close(Wrapper* list);
// Queue ExecuteCommandLists: before the desktop ExecuteCommandLists (heap submit), and after it (bind submitted).
// Both run with draw_lock and allocator_lock held. `lists` are the submitted (ready) wrappers.
void gpu_list_submit(ID3D12CommandQueue* queue, UINT count, Wrapper* const* lists);
void gpu_list_submitted(ID3D12CommandQueue* queue, UINT count, Wrapper* const* lists);
