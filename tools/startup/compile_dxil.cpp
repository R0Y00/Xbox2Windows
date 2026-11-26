#include "shader_dxil_compiler.h"
#include "direct_compute_cache.h"
#include "gfx1013_pixel.h"
#include "gfx1013_sampled_pixel.h"
#include "gfx1013_vertex_id.h"
#include "gfx1013_compute_clear.h"
#include "gfx1013_compute_native.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <set>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string_view>

using Clock = std::chrono::steady_clock;
static double elapsed_ms(Clock::time_point since) {
    return std::chrono::duration<double,std::milli>(Clock::now()-since).count();
}

namespace fs = std::filesystem;

static void publish_atomic(const fs::path& destination, const void* data, size_t size) {
    static std::atomic<unsigned long long> serial{};
    auto temp=destination;
    temp += L"." + std::to_wstring(GetCurrentProcessId()) + L"." + std::to_wstring(serial++) + L".tmp";
    try {
        std::ofstream file(temp,std::ios::binary);
        file.exceptions(std::ios::badbit | std::ios::failbit);
        if (size) file.write(static_cast<const char*>(data),size);
        file.close();
        if(!MoveFileExW(temp.c_str(),destination.c_str(),MOVEFILE_REPLACE_EXISTING))
            throw std::runtime_error("could not publish output");
    } catch(...) {
        std::error_code ignored;
        fs::remove(temp,ignored);
        throw;
    }
}

static std::vector<unsigned char> compile_to_dxil(x2w::ShaderDxilCompiler& compiler, const std::string& source,
                                                  const std::wstring& profile, const fs::path& cache, bool direct,
                                                  std::ostream& report) {
    auto start=Clock::now();
    auto output=compiler.compile(source,profile,cache,direct?x2w::ShaderSourceFormat::dxil_ir:x2w::ShaderSourceFormat::hlsl);
    report << (output.cache_hit?"cache_hit_ms=":"cold_compile_and_publish_ms=")
              << elapsed_ms(start) << " bytes=" << output.dxil.size() << '\n';
    return std::move(output.dxil);
}

static void compile_source(x2w::ShaderDxilCompiler& compiler,const std::string& source,const std::wstring& profile,
                           const fs::path& destination,const fs::path& cache,bool direct,std::ostream& report) {
    auto dxil = compile_to_dxil(compiler, source, profile, cache, direct, report);
    publish_atomic(destination, dxil.data(), dxil.size());
}

static std::optional<std::string> compile_one(x2w::ShaderDxilCompiler& compiler, std::wstring profile,
                                              const fs::path& input_path, const fs::path& destination,
                                              const fs::path& cache, std::ostream& report,
                                              const x2w::NativeComputeShader *linked=nullptr,
                                              bool emit_manifest=false) {
    Clock::time_point start;
    std::string source;
    std::optional<std::string> manifest;
    const std::wstring clear_prefix=L"gfx1013-group-clear@";
    const std::wstring direct_prefix=L"gfx1013-direct-cs@";
    const bool container=profile==L"gfx1013-container-cs";
    const bool graphics=profile==L"gfx1013-container-vs" || profile==L"gfx1013-container-ps";
    const bool direct=container || graphics || profile==L"gfx1013-direct-constant-ps" || profile.compare(0,direct_prefix.size(),direct_prefix)==0;
    const std::wstring compute_prefix=direct?direct_prefix:L"gfx1013-compute-wave32@";
    if(container || profile.compare(0,compute_prefix.size(),compute_prefix)==0){
        auto offset=container?L"0":profile.substr(compute_prefix.size());
        if(offset.empty()||offset.size()>7||offset.find_first_not_of(L"0123456789")!=offset.npos)throw std::runtime_error("invalid compute entry");
        start=Clock::now();
        const auto entry=static_cast<unsigned>(std::stoul(offset));
        bool frontend_hit=false;
        auto program=direct?x2w::prepare_direct_compute(input_path,entry,container,cache,frontend_hit)
                           :x2w::translate_compute(x2w::read_compute_request(input_path),entry,false);
        report<<"frontend=native_cpp instructions="<<program.instruction_count<<" frontend_cache_hit="<<frontend_hit
              <<" request_and_translation_ms="<<elapsed_ms(start)<<"\n";
        for (const auto &b : program.bindings) {
            using Table = x2w::ComputeViewBinding::ScalarTable;
            report << (x2w::compute_uav(b.kind) ? "u" : "t") << b.shader_register << " root=" << b.root_parameter
                   << " bytes=" << b.byte_offset << " min_size=" << b.required_bytes;
            if (b.scalar_table == Table::reflected_cbv) report << " table_view=reflected-cbv";
            if (b.scalar_table == Table::declared_cbv) report << " table_view=declared-cbv";
            if (b.scalar_table == Table::srv) report << " table_view=srv";
            report << '\n';
        }
        if (emit_manifest && !container) manifest = x2w::format_compute_runtime_manifest(program);
        source=direct?std::move(program.dxil_ir):std::move(program.hlsl);profile=L"cs_6_6";
    } else if (profile.compare(0,clear_prefix.size(),clear_prefix)==0) {
        auto offset=profile.substr(clear_prefix.size());
        if (offset.empty() || offset.size()>7 || offset.find_first_not_of(L"0123456789")!=offset.npos)
            throw std::runtime_error("expected entry byte offset");
        start=Clock::now();
        auto program=x2w::replay_group_clear(input_path,static_cast<unsigned>(std::stoul(offset)));
        report << "request_and_translation_ms=" << elapsed_ms(start) << '\n';
        for(size_t reg=0;reg<program.bindings.size();++reg)
            report << "u" << reg << " parameter=" << program.bindings[reg].parameter
                      << " byte_offset=" << program.bindings[reg].byte_offset << " R32_UINT[256]\n";
        source=std::move(program.hlsl);
        profile=L"cs_6_0";
    } else if(graphics){
        if(!linked)source=x2w::read_shader_container_bytes(input_path);
    } else {
        std::ifstream input(input_path,std::ios::binary);
        if (!input) throw std::runtime_error("cannot open source");
        source.assign(std::istreambuf_iterator<char>(input),{});
        if (input.bad()) throw std::runtime_error("source read failed");
    }
    if(graphics){
        bool pixel=profile==L"gfx1013-container-ps";
        if(linked)source=linked->dxil_ir;
        else source=x2w::translate_graphics_container(source,pixel).dxil_ir;
        profile=pixel?L"ps_6_6":L"vs_6_6";
    } else if (profile == L"gfx1013-direct-constant-ps") {
        source=x2w::direct_gfx1013_constant_pixel(source);
        profile=L"ps_6_0";
    } else if (profile == L"gfx1013-constant-ps") {
        start=Clock::now();
        source=x2w::translate_gfx1013_constant_pixel(source);
        report << "translation_ms=" << elapsed_ms(start) << '\n';
        profile=L"ps_6_0";
    } else if (profile == L"gfx1013-vertex-id-vs") {
        start=Clock::now();
        source=x2w::translate_gfx1013_vertex_id(source);
        report << "translation_ms=" << elapsed_ms(start) << '\n';
        profile=L"vs_6_0";
    } else if (profile == L"gfx1013-buffered-vs") {
        start=Clock::now();
        auto program=x2w::translate_gfx1013_buffered_vertex(source);
        report << "translation_ms=" << elapsed_ms(start) << '\n' << program.bindings;
        source=std::move(program.hlsl);
        profile=L"vs_6_0";
    } else if (profile == L"gfx1013-rgb11-r16-ps") {
        start=Clock::now();
        auto program=x2w::translate_gfx1013_sampled_pixel(source);
        report << "translation_ms=" << elapsed_ms(start) << '\n' << program.bindings;
        source=std::move(program.hlsl);
        profile=L"ps_6_0";
    }
    compile_source(compiler,source,profile,destination,cache,direct,report);
    return manifest;
}

// Outputs of concurrent jobs must be independent. Compare Windows paths
// case-insensitively, including junctions in their parent directories.
struct PathLess {
    bool operator()(const fs::path& a, const fs::path& b) const {
        return CompareStringOrdinal(a.c_str(),-1,b.c_str(),-1,TRUE)==CSTR_LESS_THAN;
    }
};

int wmain(int argc, wchar_t** argv) {
    if (argc == 4 && std::wstring_view(argv[1]) == L"--index-graphics-interfaces") {
        try {
            const auto stages = x2w::index_graphics_interfaces(argv[2],argv[3]);
            std::cout << "indexed_graphics_containers=" << stages << "\n";
            return 0;
        } catch (const std::exception &error) {
            std::cerr << error.what() << "\n";
            return 1;
        }
    }
    const bool parallel = argc > 3 && std::wstring_view(argv[3]) == L"--jobs";
    const int profile_index = parallel ? 5 : 3;
    const int first = profile_index+1;
    const bool directory = argc > first && std::wstring_view(argv[first]) == L"--requests";
    if (directory ? argc != first+3 : (argc < first+2 || (argc-first)%2)) {
        std::cerr << "usage: compile_dxil SDK_DIRECTORY CACHE_DIRECTORY [--jobs N] PROFILE INPUT OUTPUT.cso [INPUT OUTPUT ...]\n"
                     "Use a DXC profile for HLSL, gfx1013-constant-ps or gfx1013-rgb11-r16-ps for a raw pixel program, or\n"
                     "gfx1013-vertex-id-vs or gfx1013-buffered-vs for a raw vertex program. gfx1013-sampled-pair takes one VS and one PS (in order).\n"
                     "gfx1013-buffered-pair takes a buffered vertex program and a pixel HLSL source.\n"
                     "gfx1013-group-clear@BYTE_OFFSET and gfx1013-compute-wave32@BYTE_OFFSET take a req_<key>.c67.\n"
                     "Direct DXIL profiles: gfx1013-direct-cs@BYTE_OFFSET, gfx1013-container-cs, gfx1013-container-vs, gfx1013-container-ps.\n"
                     "gfx1013-container-pair accepts two containers and emits VS then PS using the runtime register allocation.\n"
                     "For capture directories use PROFILE --requests DIRECTORY OUTPUT_DIRECTORY. "
                     "gfx1013-container-rect also emits the rectangle geometry stage.\n"
                     "Use --index-graphics-interfaces REVIEWED_DIRECTORY CACHE_DIRECTORY to index native graphics interfaces.\n";
        return 2;
    }
    try {
        const std::wstring_view profile = argv[profile_index];
        const bool rectangles=profile==L"gfx1013-container-rect";
        if(rectangles && !directory) throw std::runtime_error("rectangle compilation takes a graphics request directory");
        const bool direct_pair = profile == L"gfx1013-container-pair" || rectangles;
        const bool graphics_batch = directory && direct_pair;
        const bool compute_request = profile.rfind(L"gfx1013-group-clear@",0)==0 ||
                                     profile.rfind(L"gfx1013-compute-wave32@",0)==0 || profile.rfind(L"gfx1013-direct-cs@",0)==0 || profile==L"gfx1013-container-cs";
        std::vector<fs::path> inputs, outputs, stems;
        if (directory) {
            if (!compute_request && !graphics_batch) throw std::runtime_error("profile does not take capture directories");
            const auto extension = graphics_batch ? L".a" : profile == L"gfx1013-container-cs" ? L".cdx" : L".c67";
            for (const auto& file : fs::directory_iterator(argv[first+1])) {
                const auto path = file.path();
                if (file.is_regular_file() && path.extension()==extension && path.stem().wstring().rfind(L"req_",0)==0)
                    inputs.push_back(fs::weakly_canonical(path));
            }
            if (inputs.empty()) throw std::runtime_error("no ready shader requests");
            std::sort(inputs.begin(),inputs.end());
            fs::create_directories(argv[first+2]);
            for (const auto& path : inputs) {
                auto key_stem = path.stem().wstring().substr(4);
                if (key_stem.empty()) key_stem = L"_invalid";
                stems.push_back(fs::weakly_canonical(fs::path(argv[first+2]) / key_stem));
                outputs.push_back(fs::path(stems.back()) += (graphics_batch ? L".vs.cso" : L".cs.cso"));
            }
        } else {
            for(int i=first;i<argc;i+=2) {
                inputs.push_back(fs::weakly_canonical(fs::absolute(argv[i])));
                outputs.push_back(fs::weakly_canonical(fs::absolute(argv[i+1])));
            }
        }
        const int count = static_cast<int>(inputs.size());
        const bool sampled_pair = std::wstring_view(argv[profile_index]) == L"gfx1013-sampled-pair";
        const bool buffered_pair = std::wstring_view(argv[profile_index]) == L"gfx1013-buffered-pair";
        if (!directory && (direct_pair || sampled_pair || buffered_pair) && count != 2) throw std::runtime_error("graphics pair requires a vertex and a pixel job");
        std::optional<x2w::DirectGraphicsPair> linked;
        if (direct_pair && !directory) {
            bool hit = false;
            linked = x2w::prepare_direct_graphics(x2w::read_shader_container_bytes(inputs[0]),
                x2w::read_shader_container_bytes(inputs[1]), argv[2], hit);
            std::cout << "graphics_pair frontend_cache_hit=" << hit << "\n";
        }
        auto linked_stage=[&](int index)->const x2w::NativeComputeShader*{
            if(!linked)return nullptr;
            return index?&linked->pixel:&linked->vertex;
        };
        const auto job_profile = [&](int index) -> const wchar_t* {
            if(direct_pair)return index?L"gfx1013-container-ps":L"gfx1013-container-vs";
            if (sampled_pair) return index == 0 ? L"gfx1013-vertex-id-vs" : L"gfx1013-rgb11-r16-ps";
            if (buffered_pair) return index == 0 ? L"gfx1013-buffered-vs" : L"ps_6_0";
            return argv[profile_index];
        };
        int jobs=1;
        if (parallel) {
            std::wstring number=argv[4];
            if (number.empty() || number.size()>2 || number.find_first_not_of(L"0123456789")!=number.npos)
                throw std::runtime_error("--jobs must be between 1 and 64");
            jobs=std::stoi(number);
            if (jobs<1 || jobs>64) throw std::runtime_error("--jobs must be between 1 and 64");
            jobs=(std::min)(jobs,count);
        }
        auto start=Clock::now();
        std::set<fs::path,PathLess> occupied(inputs.begin(),inputs.end());
        if(compute_request) {
            for(const auto& input : inputs) {
                for(auto suffix : {L".root",L".data",L".sa",L".sb",L".desc"})
                    occupied.insert(fs::weakly_canonical(fs::path(input).replace_extension(suffix)));
            }
        }
        if(graphics_batch) {
            for(const auto& input : inputs) {
                for(auto suffix : {L".b",L".root",L".ia",L".psin"})
                    occupied.insert(fs::weakly_canonical(fs::path(input).replace_extension(suffix)));
            }
        }
        auto reserve_output = [&](const fs::path& path) {
            if(!occupied.insert(path).second)
                throw std::runtime_error("parallel outputs must be distinct from all inputs and other outputs");
        };
        for(int i=0;i<count;++i) {
            reserve_output(outputs[i]);
            if(directory) {
                reserve_output(fs::path(stems[i]) += L".txt");
                if(graphics_batch) {
                    reserve_output(fs::path(stems[i]) += L".ps.cso");
                    reserve_output(fs::path(stems[i]) += L".psdep");
                    if(rectangles) reserve_output(fs::path(stems[i]) += L".gs.cso");
                }
            }
        }
        auto compile_job = [&](x2w::ShaderDxilCompiler &compiler, int i, std::ostream &report) {
            if (!graphics_batch) {
                if (!directory) {
                    compile_one(compiler, job_profile(i), inputs[i], outputs[i], argv[2], report, linked_stage(i));
                    return;
                }
                const fs::path manifest_path = fs::path(stems[i]) += L".txt";
                std::error_code ignored;
                fs::remove(manifest_path, ignored);
                auto manifest = compile_one(compiler, job_profile(i), inputs[i], outputs[i], argv[2], report, nullptr, true);
                if (manifest) publish_atomic(manifest_path, manifest->data(), manifest->size());
                return;
            }
            const fs::path manifest_path = fs::path(stems[i]) += L".txt";
            const fs::path psdep_path = fs::path(stems[i]) += L".psdep";
            std::error_code ignored;
            fs::remove(manifest_path, ignored);
            auto input = x2w::read_graphics_request(inputs[i]);
            bool hit = false;
            auto pair = x2w::prepare_direct_graphics(input.a, input.b, argv[2], hit, input.root, input.input_layout, input.pixel_entry);
            report << "graphics_pair frontend_cache_hit=" << hit << '\n';
            auto manifest = rectangles ? std::nullopt : x2w::format_graphics_runtime_manifest(pair);
            std::vector<unsigned char> gs_dxil;
            if (rectangles) {
                auto ir=x2w::native_rectangle_geometry(pair.vertex.outputs,pair.pixel.inputs);
                gs_dxil = compile_to_dxil(compiler, ir, L"gs_6_6", argv[2], true, report);
            }
            auto vs_dxil = compile_to_dxil(compiler, pair.vertex.dxil_ir, L"vs_6_6", argv[2], true, report);
            auto ps_dxil = compile_to_dxil(compiler, pair.pixel.dxil_ir, L"ps_6_6", argv[2], true, report);
            if (rectangles)
                publish_atomic(fs::path(stems[i]) += L".gs.cso", gs_dxil.data(), gs_dxil.size());
            publish_atomic(outputs[i], vs_dxil.data(), vs_dxil.size());
            publish_atomic(fs::path(stems[i]) += L".ps.cso", ps_dxil.data(), ps_dxil.size());
            if (manifest) {
                if (!input.pixel_input_description.empty())
                    publish_atomic(psdep_path, input.pixel_input_description.data(), input.pixel_input_description.size());
                publish_atomic(manifest_path, manifest->data(), manifest->size());
            }
        };
        if(jobs==1 && !directory) {
            x2w::ShaderDxilCompiler compiler(argv[1]);
            std::cout << "compiler_init_ms=" << elapsed_ms(start) << '\n';
            for(int i=0;i<count;++i)
                compile_job(compiler, i, std::cout);
            return 0;
        }
        std::atomic<int> next{};
        std::vector<std::string> reports(count), errors(count);
        x2w::ShaderDxilCompiler coordinator(argv[1]);
        std::cout << "compiler_init_ms=" << elapsed_ms(start) << '\n';
        std::vector<std::future<void>> workers;
        for(int w=0;w<jobs;++w) {
            workers.emplace_back(std::async(std::launch::async,[&] {
                auto init=Clock::now();
                auto compiler=coordinator.new_worker();
                double init_ms=elapsed_ms(init);
                for(int i=next++;i<count;i=next++) {
                    std::ostringstream report;
                    if(init_ms>=0) { report << "worker_init_ms=" << init_ms << '\n'; init_ms=-1; }
                    try { compile_job(compiler, i, report); }
                    catch(const std::exception& error) { errors[i]=error.what(); }
                    reports[i]=report.str();
                }
            }));
        }
        for(auto& worker:workers) worker.get();
        int rejected=0;
        for(int i=0;i<count;++i) {
            std::cout << "job=" << i+1 << '\n' << reports[i];
            if(!errors[i].empty()) {
                ++rejected;
                std::cerr << "job=" << i+1 << " input=" << inputs[i].filename().string() << " " << errors[i] << '\n';
            }
        }
        std::cout << "batch_ms=" << elapsed_ms(start) << " workers=" << jobs << '\n';
        if(directory) std::cout << "requests=" << count << " compiled=" << count-rejected << " refused=" << rejected << '\n';
        return rejected ? 1 : 0;
    } catch(const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
