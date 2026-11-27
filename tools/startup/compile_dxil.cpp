#include "shader_dxil_compiler.h"
#include "direct_compute_cache.h"
#include "gfx1013_pixel.h"
#include "gfx1013_sampled_pixel.h"
#include "gfx1013_vertex_id.h"
#include "gfx1013_compute_clear.h"
#include "gfx1013_compute_native.h"
#include "startup_stubs/native_global_srv.h"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <future>
#include <map>
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

// The view metadata captured with a compute request (req_<key>.views), read before the program is translated. Absent:
// the request has no captured selection and a GPU-selected heap read is refused. Present: every refusal below happens
// before any output is published.
struct CapturedViews {
    x2w::HeapViewContract contract = x2w::HeapViewContract::none;
    x2w_startup::GlobalSrvRequest request;
    bool present = false;
};

static CapturedViews read_captured_views(const fs::path& request_path) {
    CapturedViews views;
    const auto path = fs::path(request_path).replace_extension(L".views");
    std::ifstream file(path, std::ios::binary);
    if (!file) return views;
    std::string text((std::istreambuf_iterator<char>(file)), {});
    if (file.bad()) throw std::runtime_error("global SRV view metadata read failed");
    std::string error;
    if (!x2w_startup::decode_global_srv_request(text, views.request, error)) throw std::runtime_error(error);
    views.present = true;
    const auto& request = views.request;
    auto refusal = x2w_startup::refuse_global_srv_selection(request.views, request.selected, request.needed);
    if (!refusal.empty()) throw std::runtime_error(refusal);
    if (request.needed != x2w_startup::GlobalSrvType::raw_buffer)
        throw std::runtime_error(std::string("GPU-selected heap slot view type ") +
                                 x2w_startup::global_srv_type_name(request.needed) +
                                 " is not raw_buffer; only raw buffer heap reads are translated");
    views.contract = x2w::HeapViewContract::raw_buffer;
    return views;
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
        CapturedViews views;
        if(container)views=read_captured_views(input_path);
        auto program=direct?x2w::prepare_direct_compute(input_path,entry,container,cache,frontend_hit,views.contract)
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
        if(views.present){
            const bool heap_read=std::any_of(program.bindings.begin(),program.bindings.end(),[](const auto& b){
                return b.kind==x2w::ComputeViewKind::heap_buffer_srv || b.kind==x2w::ComputeViewKind::heap_stride_srv;});
            if(heap_read){
                auto refusal=x2w_startup::refuse_global_srv_selection(views.request.views,views.request.selected,views.request.needed);
                if(!refusal.empty())throw std::runtime_error(refusal);
            }
            report<<"global_srv select="<<views.request.selected<<" need="<<x2w_startup::global_srv_type_name(views.request.needed)<<'\n';
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

static std::string collapse_ascii_line(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    bool space = true;
    for (unsigned char c : text) {
        if (c == '\r' || c == '\n' || c == '\t' || c == ' ') {
            if (!space) {
                out.push_back(' ');
                space = true;
            }
        } else {
            out.push_back(static_cast<char>(c));
            space = false;
        }
    }
    if (!out.empty() && out.back() == ' ')
        out.pop_back();
    return out;
}

static std::string format_fail_line(std::string_view message) {
    std::string out = collapse_ascii_line(message);
    if (out.empty())
        out = "unknown refusal";
    out.push_back('\n');
    return out;
}

static bool is_ident_char(unsigned char c) { return std::isalnum(c) != 0 || c == '_'; }

static std::string normalize_refusal_reason(std::string_view raw) {
    std::string s = collapse_ascii_line(raw);
    if (s.rfind("native compute +0x", 0) == 0) {
        const size_t colon = s.find(": ");
        if (colon != std::string::npos)
            s.erase(0, colon + 2);
    }
    auto replace_all_with = [](std::string& str, auto&& matcher) {
        std::string out;
        out.reserve(str.size());
        size_t i = 0;
        while (i < str.size()) {
            size_t end = 0;
            std::string rep;
            if (matcher(str, i, end, rep)) {
                out += rep;
                i = end;
            } else {
                out.push_back(str[i++]);
            }
        }
        str.swap(out);
    };
    replace_all_with(s, [](const std::string& str, size_t i, size_t& end, std::string& rep) {
        if (i + 2 >= str.size() || str[i] != '0' || (str[i + 1] != 'x' && str[i + 1] != 'X'))
            return false;
        if (i > 0 && is_ident_char(static_cast<unsigned char>(str[i - 1])))
            return false;
        size_t j = i + 2;
        while (j < str.size() && std::isxdigit(static_cast<unsigned char>(str[j])))
            ++j;
        if (j == i + 2)
            return false;
        if (j < str.size() && (str[j] == 'u' || str[j] == 'U') &&
            (j + 1 == str.size() || !is_ident_char(static_cast<unsigned char>(str[j + 1]))))
            ++j;
        end = j;
        rep = "0xN";
        return true;
    });
    for (const std::string_view prefix : {" at byte ", " VGPR "}) {
        replace_all_with(s, [&](const std::string& str, size_t i, size_t& end, std::string& rep) {
            if (str.compare(i, prefix.size(), prefix) != 0)
                return false;
            size_t j = i + prefix.size();
            if (j >= str.size() || !std::isdigit(static_cast<unsigned char>(str[j])))
                return false;
            while (j < str.size() && std::isdigit(static_cast<unsigned char>(str[j])))
                ++j;
            if (j < str.size() && is_ident_char(static_cast<unsigned char>(str[j])))
                return false;
            end = j;
            rep = std::string(prefix) + "N";
            return true;
        });
    }
    replace_all_with(s, [](const std::string& str, size_t i, size_t& end, std::string& rep) {
        if (str[i] != 's' && str[i] != 'v')
            return false;
        if (i > 0 && is_ident_char(static_cast<unsigned char>(str[i - 1])))
            return false;
        size_t j = i + 1;
        if (j < str.size() && str[j] == '[') {
            ++j;
            const size_t a = j;
            while (j < str.size() && std::isdigit(static_cast<unsigned char>(str[j])))
                ++j;
            if (j == a || j >= str.size() || str[j] != ':')
                return false;
            const size_t b = ++j;
            while (j < str.size() && std::isdigit(static_cast<unsigned char>(str[j])))
                ++j;
            if (j == b || j >= str.size() || str[j] != ']')
                return false;
            ++j;
            if (j < str.size() && is_ident_char(static_cast<unsigned char>(str[j])))
                return false;
            end = j;
            rep = std::string(1, str[i]) + "[N:N]";
            return true;
        }
        const size_t a = j;
        while (j < str.size() && std::isdigit(static_cast<unsigned char>(str[j])))
            ++j;
        if (j == a || (j < str.size() && is_ident_char(static_cast<unsigned char>(str[j]))))
            return false;
        end = j;
        rep = std::string(1, str[i]) + "N";
        return true;
    });
    if (s.empty())
        s = "unknown refusal";
    return s;
}

static int parse_jobs_count(std::wstring_view number) {
    if (number.empty() || number.size() > 2 || number.find_first_not_of(L"0123456789") != number.npos)
        throw std::runtime_error("--jobs must be between 1 and 64");
    const int jobs = std::stoi(std::wstring(number));
    if (jobs < 1 || jobs > 64)
        throw std::runtime_error("--jobs must be between 1 and 64");
    return jobs;
}

static std::string json_escape(std::string_view text) {
    std::string out;
    out.reserve(text.size() + 2);
    out.push_back('"');
    for (unsigned char c : text) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            } else {
                out.push_back(static_cast<char>(c));
            }
            break;
        }
    }
    out.push_back('"');
    return out;
}

static int run_refusals(int argc, wchar_t** argv) {
    fs::path request_dir;
    bool has_dir = false;
    bool json = false;
    int jobs = 1;
    try {
        for (int i = 2; i < argc; ++i) {
            const std::wstring_view arg = argv[i];
            if (arg == L"--json") {
                json = true;
            } else if (arg == L"--jobs") {
                if (i + 1 >= argc)
                    throw std::runtime_error("missing value for --jobs");
                jobs = parse_jobs_count(argv[++i]);
            } else if (!arg.empty() && arg.front() == L'-') {
                std::cerr << "usage: compile_dxil --refusals REQUEST_DIRECTORY [--jobs N] [--json]\n";
                return 2;
            } else if (has_dir) {
                std::cerr << "usage: compile_dxil --refusals REQUEST_DIRECTORY [--jobs N] [--json]\n";
                return 2;
            } else {
                request_dir = fs::path(arg);
                has_dir = true;
            }
        }
        if (!has_dir) {
            std::cerr << "usage: compile_dxil --refusals REQUEST_DIRECTORY [--jobs N] [--json]\n";
            return 2;
        }
        if (!fs::exists(request_dir) || !fs::is_directory(request_dir)) {
            std::cerr << "request path must be an existing directory\n";
            return 1;
        }
        std::vector<fs::path> graphics_inputs, compute_inputs;
        for (const auto& file : fs::directory_iterator(request_dir)) {
            if (!file.is_regular_file())
                continue;
            const auto path = file.path();
            if (path.stem().wstring().rfind(L"req_", 0) != 0)
                continue;
            if (path.extension() == L".a")
                graphics_inputs.push_back(fs::weakly_canonical(path));
            else if (path.extension() == L".c67")
                compute_inputs.push_back(fs::weakly_canonical(path));
        }
        std::sort(graphics_inputs.begin(), graphics_inputs.end(), PathLess{});
        std::sort(compute_inputs.begin(), compute_inputs.end(), PathLess{});

        struct Task {
            bool compute;
            size_t index;
        };
        std::vector<Task> tasks;
        tasks.reserve(graphics_inputs.size() + compute_inputs.size());
        for (size_t i = 0; i < graphics_inputs.size(); ++i)
            tasks.push_back({false, i});
        for (size_t i = 0; i < compute_inputs.size(); ++i)
            tasks.push_back({true, i});

        std::vector<std::string> results(tasks.size());
        if (!tasks.empty()) {
            const int workers_count = (std::min)(jobs, static_cast<int>(tasks.size()));
            std::atomic<size_t> next{};
            auto run_worker = [&]() {
                for (size_t t = next++; t < tasks.size(); t = next++) {
                    const auto& task = tasks[t];
                    try {
                        if (task.compute) {
                            auto req = x2w::read_compute_request(compute_inputs[task.index]);
                            (void)x2w::translate_compute(req, 0, true);
                        } else {
                            auto input = x2w::read_graphics_request(graphics_inputs[task.index]);
                            bool hit = false;
                            (void)x2w::prepare_direct_graphics(input.a, input.b, request_dir, hit,
                                                               input.root, input.input_layout, input.pixel_entry, false);
                        }
                    } catch (const fs::filesystem_error&) {
                        results[t] = "unreadable shader request";
                    } catch (const std::exception& error) {
                        results[t] = normalize_refusal_reason(error.what());
                    }
                }
            };
            if (workers_count == 1) {
                run_worker();
            } else {
                std::vector<std::future<void>> workers;
                workers.reserve(workers_count);
                for (int w = 0; w < workers_count; ++w)
                    workers.emplace_back(std::async(std::launch::async, run_worker));
                for (auto& w : workers)
                    w.get();
            }
        }

        struct Section {
            unsigned examined = 0, no_refusal = 0, refused = 0;
            std::map<std::string, unsigned> counts;
            std::vector<std::pair<std::string, unsigned>> sorted_reasons() const {
                std::vector<std::pair<std::string, unsigned>> rows(counts.begin(), counts.end());
                std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
                    if (a.second != b.second)
                        return a.second > b.second;
                    return a.first < b.first;
                });
                return rows;
            }
        } graphics_sec, compute_sec;

        for (size_t t = 0; t < tasks.size(); ++t) {
            Section& sec = tasks[t].compute ? compute_sec : graphics_sec;
            ++sec.examined;
            if (results[t].empty()) {
                ++sec.no_refusal;
            } else {
                ++sec.refused;
                ++sec.counts[results[t]];
            }
        }

        if (json) {
            auto write_json_sec = [](std::ostream& os, const Section& sec) {
                os << "{\"examined\":" << sec.examined
                   << ",\"no_refusal\":" << sec.no_refusal
                   << ",\"refused\":" << sec.refused
                   << ",\"reasons\":[";
                const auto rows = sec.sorted_reasons();
                for (size_t i = 0; i < rows.size(); ++i) {
                    if (i) os << ',';
                    os << "{\"reason\":" << json_escape(rows[i].first) << ",\"count\":" << rows[i].second << '}';
                }
                os << "]}";
            };
            std::cout << "{\"graphics\":";
            write_json_sec(std::cout, graphics_sec);
            std::cout << ",\"compute\":";
            write_json_sec(std::cout, compute_sec);
            std::cout << "}\n";
        } else {
            auto write_text_sec = [](std::ostream& os, const char* title, const char* count_label, const Section& sec) {
                os << title << '\n'
                   << count_label << ": " << sec.examined << '\n'
                   << "No refusal recorded (not compiled): " << sec.no_refusal << '\n'
                   << "Refused: " << sec.refused << '\n'
                   << "Count\tReason\n";
                for (const auto& [reason, count] : sec.sorted_reasons())
                    os << count << '\t' << reason << '\n';
            };
            write_text_sec(std::cout, "Graphics requests", "Requests examined", graphics_sec);
            write_text_sec(std::cout, "Compute requests (device slot 67)", "Compute requests examined", compute_sec);
        }
        return 0;
    } catch (const fs::filesystem_error&) {
        std::cerr << "unreadable shader request directory\n";
        return 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

int wmain(int argc, wchar_t** argv) {
    if (argc >= 2 && std::wstring_view(argv[1]) == L"--refusals")
        return run_refusals(argc, argv);
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
                     "gfx1013-container-cs reads an optional req_<key>.views next to the .cdx (GPU-selected heap slot view types).\n"
                     "gfx1013-container-pair accepts two containers and emits VS then PS using the runtime register allocation.\n"
                     "For capture directories use PROFILE --requests DIRECTORY OUTPUT_DIRECTORY. "
                     "gfx1013-container-rect also emits the rectangle geometry stage.\n"
                     "Use --refusals REQUEST_DIRECTORY [--jobs N] [--json] for a read-only refusal summary.\n"
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
        if (parallel)
            jobs = (std::min)(parse_jobs_count(argv[4]), count);
        auto start=Clock::now();
        std::set<fs::path,PathLess> occupied(inputs.begin(),inputs.end());
        if(compute_request) {
            for(const auto& input : inputs) {
                for(auto suffix : {L".root",L".data",L".sa",L".sb",L".desc",L".views"})
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
                reserve_output(fs::path(stems[i]) += L".fail");
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
                fs::remove(fs::path(stems[i]) += L".fail", ignored);
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
            fs::remove(fs::path(stems[i]) += L".fail", ignored);
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
                    catch(const std::exception& error) {
                        errors[i]=error.what();
                        if (directory) {
                            try {
                                const auto line = format_fail_line(errors[i]);
                                publish_atomic(fs::path(stems[i]) += L".fail", line.data(), line.size());
                            } catch (...) {
                            }
                        }
                    }
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
