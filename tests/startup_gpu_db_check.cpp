// startup_gpu_db_check: the runtime's C5 lookup (src/startup_stubs/gpu_db.cpp) over a synthetic database.
//   startup_gpu_db_check            a record found by its bytecode and stage, converted to gpu_abi.h's types;
//                                       another stage and other bytes miss; the manifest is reported.
//   startup_gpu_db_check --mutate   the database claims raw_views 0 (made without twins) while this run has
//                                       twins: the load must be refused, so the lookup fails (WILL_FAIL).
#include <windows.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "shader_db/shader_db_writer.h"
#include "startup_stubs/gpu_db.h"

using namespace x2w_gpu;

int main(int argc, char** argv)
{
    const bool mutate = argc >= 2 && !std::strcmp(argv[1], "--mutate");
    int failures = 0;
    const auto check = [&](bool ok, const char* what) {
        if (!ok) {
            std::printf("FAIL: %s\n", what);
            ++failures;
        }
    };
    const unsigned char title[] = "DXBC a title's container bytes";
    char key[17];
    gpu_key(title, sizeof(title), key);

    shader_db::ShaderRecord r;
    r.key = std::strtoull(key, nullptr, 16);
    r.stage = shader_db::Stage::ps;
    r.container_bytes = sizeof(title);
    r.dxil = {'D', 'X', 'B', 'C', 1, 2, 3};
    r.user_data = {{0, 2, shader_db::root_table, 0}, {4, 0, shader_db::root_other_base + abi::user_view_bias, 0}};
    r.user_data_dwords = 8;
    r.samplers = {{1, 2, 3, 4}};
    r.static_samplers = {{0x15, {1, 2, 3}, 0.5f, 16, 0, 0, 0.0f, 13.0f}};
    r.root_srvs = {abi::slot_strides};
    r.root_indexed_views = {{1, 2, 3, 64, 4}};
    r.used = {{shader_db::resource_uav, shader_db::root_table, 8, 0, 16, 8, 2, 1}, {shader_db::resource_other_base + 9, shader_db::root_constants, 0, 0, 0, 4, 0, 0}};
    r.stores = true;
    wchar_t dir[MAX_PATH];
    GetTempPathW(MAX_PATH, dir);
    const std::wstring path = std::wstring(dir) + (mutate ? L"startup_gpu_db_check_mutate.vxdb" : L"startup_gpu_db_check.vxdb");
    const std::string manifest = std::string("format gpu-x2w-1\nview_heap_limit 2097151\nraw_views ") + (mutate ? "0" : "2097152") +
                                 "\nmax_wave 64\nhost_rdna 15\nshader_opt 0x3f\ngroup_sync 2\nX2W_SHADERCFG_HELPER_OWN \ncommit abc123\n";
    std::string error;
    check(shader_db::write_db(path, {r}, manifest, error), "write");
    SetEnvironmentVariableW(L"X2W_GPU_DB", path.c_str());
    SetEnvironmentVariableW(L"X2W_GPU_ABI", L"1");

    const GpuShader* s = gpu_shader(title, sizeof(title), GpuStage::pixel);
    check(s != nullptr, "the record is found by its bytecode and stage");
    if (s) {
        check(!std::strcmp(s->key, key), "key");
        check(s->hw_stage == GpuHwStage::ps && s->stage == GpuStage::pixel, "stages");
        check(s->dxil_bytes == 7 && !std::memcmp(s->dxil, "DXBC", 4), "dxil");
        check(s->user_data.count == 2 && s->user_data[0].root == GpuRootType::table && s->user_data[0].slot == 2 &&
                  s->user_data[1].root == GpuRootType::other && s->user_data[1].other == abi::user_view_bias && s->user_data[1].index == 4,
              "user data and an Other root code");
        check(s->user_data_dwords() == 8, "user_data_dwords");
        check(s->samplers.count == 1 && s->samplers[0].words[3] == 4, "sampler words");
        check(s->static_samplers.count == 1 && s->static_samplers[0].max_lod == 13.0f && s->static_samplers[0].address[2] == 3, "static sampler");
        check(s->root_srvs.count == 1 && s->root_srvs[0] == abi::slot_strides && s->root_cbvs.empty(), "root views");
        check(s->root_indexed_views.count == 1 && s->root_indexed_views[0].byte_offset == 64, "root indexed views");
        check(s->used.count == 2 && s->used[0].kind == GpuResourceType::uav && s->used[0].stride == 16 && s->used[0].dwords == 8 &&
                  s->used[1].kind == GpuResourceType::other && s->used[1].other_kind == 9,
              "used resources");
        check(s->stores && !s->stores_by_address && !s->has_threads, "stores and threads");
    }
    check(!gpu_shader(title, sizeof(title), GpuStage::vertex), "another stage misses");
    check(!gpu_shader(title, sizeof(title) - 1, GpuStage::pixel), "other bytes miss");
    check(gpu_shader_by_key(key, GpuStage::pixel) == s, "by key");
    const GpuManifest* m = gpu_manifest();
    check(m && m->view_heap_limit == abi::view_heap_limit && m->raw_views == abi::raw_views && m->max_wave == 64 && m->host_rdna &&
              m->opt_bits == 0x3f && m->group_sync && !m->helper_own && !std::strcmp(m->commit, "abc123"),
          "manifest");
    std::printf("%s\n", failures ? "startup_gpu_db_check: FAILED" : "startup_gpu_db_check: ok");
    return failures ? 1 : 0;
}
