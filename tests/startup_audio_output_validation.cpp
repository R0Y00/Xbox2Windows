// Output validation for the real XAudio2 provider (Refs #1145).
//
// The question this answers: when a source-built caller drives the *real* XAudio2 provider, what reaches
// the endpoint - a known tone, intentional silence, a drained queue and a resumption - and when, on the
// endpoint's own clock. Earlier fixtures validate a sink we own (#1250) and the voice lifecycle (#1399);
// neither can say what the endpoint actually produced.
//
// Method: the caller submits a plan of distinguishable blocks through XAudio2 while a WASAPI loopback
// client captures the endpoint the provider renders to. Each captured 10 ms window is classified as tone
// or silence by the power at the caller's tone frequency, so the report is transitions and timing, not
// samples. Audio bytes stay in memory and are never stored or printed.
//
// Isolation: Windows has no API to create an audio endpoint, so the capture is of the shared endpoint the
// provider renders to. The fixture reports the isolation it actually had (`shared`, since the real provider
// uses the default endpoint) rather than assuming one, and skips (77) when there is no endpoint or the
// loopback client is refused. Replacing the real provider is explicitly out of scope for this slice.
//
// The caller keeps using the real provider exactly as the launch path does: no import patching, no
// replacement voice, no change to provider selection.
#define NOMINMAX
#include <windows.h>
#include <audioclient.h>
#include <audioclientactivationparams.h>
#include <mmdeviceapi.h>
#include <xaudio2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int skip_no_endpoint = 77;
constexpr double tone_hz = 440.0;
constexpr double window_ms = 10.0;

template <class T>
void release(T*& pointer)
{
    if (pointer) {
        pointer->Release();
        pointer = nullptr;
    }
}

double now_ms()
{
    LARGE_INTEGER counter{}, frequency{};
    QueryPerformanceCounter(&counter);
    QueryPerformanceFrequency(&frequency);
    return 1000.0 * static_cast<double>(counter.QuadPart) / static_cast<double>(frequency.QuadPart);
}

double frames_to_ms(uint64_t frames, uint32_t rate)
{
    return rate ? 1000.0 * static_cast<double>(frames) / rate : 0.0;
}

// Level of a window: RMS, which separates a submitted tone from the endpoint's numeric silence without
// depending on where the window falls against the tone's phase. A frequency-share metric was tried first
// and made a steady tone read anywhere between 0.06 and 0.32, so it crossed the threshold at random.
// Share of a window's energy at one frequency, used only by the isolation control: our own signal is
// 440 Hz and the interfering process plays 880 Hz, so the captured tone can be attributed.
double frequency_share(const std::vector<float>& mono, uint32_t rate, double hz)
{
    if (mono.empty() || rate == 0) return 0.0;
    const double omega = 2.0 * 3.14159265358979323846 * hz / rate;
    const double coeff = 2.0 * std::cos(omega);
    double s1 = 0, s2 = 0, energy = 0;
    for (double sample : mono) {
        const double s0 = sample + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
        energy += sample * sample;
    }
    const double n = static_cast<double>(mono.size());
    if (energy <= 1e-12) return 0.0;
    const double power = s1 * s1 + s2 * s2 - coeff * s1 * s2;
    return (power / (n * n)) / (energy / n);
}
double window_level(const std::vector<float>& mono)
{
    if (mono.empty()) return 0.0;
    double energy = 0;
    for (double sample : mono) energy += sample * sample;
    return std::sqrt(energy / static_cast<double>(mono.size()));
}
// Process-scoped capture: the loopback client is activated for this process tree, so other clients on
// the endpoint cannot supply the tone or turn a silence window into a failure. Unavailable activation is
// a capability skip with a named reason, never an assertion failure.
class ActivationHandler : public IActivateAudioInterfaceCompletionHandler
{
public:
    ActivationHandler() : done_(CreateEventW(nullptr, FALSE, FALSE, nullptr)) {}
    ~ActivationHandler() { if (done_) CloseHandle(done_); }
    HANDLE done() const { return done_; }
    HANDLE& done_handle() { return done_; }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** object) override
    {
        if (!object) return E_POINTER;
        if (riid == __uuidof(IUnknown) || riid == __uuidof(IActivateAudioInterfaceCompletionHandler)) {
            *object = static_cast<IActivateAudioInterfaceCompletionHandler*>(this);
            AddRef();
            return S_OK;
        }
        *object = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return static_cast<ULONG>(InterlockedIncrement(&refs_)); }
    ULONG STDMETHODCALLTYPE Release() override
    {
        const LONG left = InterlockedDecrement(&refs_);
        if (!left) delete this;
        return static_cast<ULONG>(left);
    }
    HRESULT STDMETHODCALLTYPE ActivateCompleted(IActivateAudioInterfaceAsyncOperation*) override
    {
        SetEvent(done_);
        return S_OK;
    }
private:
    LONG refs_ = 1;
    HANDLE done_ = nullptr;
};

// Activates the process-loopback client for this process tree and returns a named reason on failure.
HRESULT process_loopback_client(IAudioClient** client, std::string& reason)
{
    AUDIOCLIENT_ACTIVATION_PARAMS params{};
    params.ActivationType = AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK;
    params.ProcessLoopbackParams.TargetProcessId = GetCurrentProcessId();
    params.ProcessLoopbackParams.ProcessLoopbackMode = PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE;
    PROPVARIANT property{};
    property.vt = VT_BLOB;
    property.blob.cbSize = sizeof(params);
    property.blob.pBlobData = reinterpret_cast<BYTE*>(&params);
    auto* handler = new ActivationHandler();
    IActivateAudioInterfaceAsyncOperation* operation = nullptr;
    HRESULT hr = ActivateAudioInterfaceAsync(VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK, __uuidof(IAudioClient),
                                             &property, handler, &operation);
    if (FAILED(hr) || !operation) {
        handler->Release();
        { static char why[96]; std::snprintf(why, sizeof(why), "process loopback activation unavailable (0x%08lx)", static_cast<unsigned long>(hr)); reason = why; }
        return FAILED(hr) ? hr : E_FAIL;
    }
    // Pump messages while waiting: the completion callback is dispatched on this apartment.
    DWORD waited = WAIT_TIMEOUT;
    for (const ULONGLONG deadline = GetTickCount64() + 5000; GetTickCount64() < deadline;) {
        waited = MsgWaitForMultipleObjects(1, &handler->done_handle(), FALSE, 50, QS_ALLINPUT);
        if (waited == WAIT_OBJECT_0) break;
        MSG message;
        while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
    HRESULT status = E_FAIL;
    IUnknown* iface = nullptr;
    if (waited != WAIT_OBJECT_0 || FAILED(operation->GetActivateResult(&status, &iface)) || FAILED(status) || !iface) {
        if (iface) iface->Release();
        operation->Release();
        handler->Release();
        { static char why[96]; std::snprintf(why, sizeof(why), "process loopback activation did not complete (0x%08lx)", static_cast<unsigned long>(status)); reason = why; }
        return FAILED(status) ? status : E_FAIL;
    }
    hr = iface->QueryInterface(IID_PPV_ARGS(client));
    iface->Release();
    operation->Release();
    handler->Release();
    if (FAILED(hr) || !*client) {
        reason = "process loopback client unavailable";
        return FAILED(hr) ? hr : E_NOINTERFACE;
    }
    return S_OK;
}
struct Plan {
    int tone_blocks = 8;          // known tone
    int silence_blocks = 4;       // intentional silence
    int withheld_ms = 300;        // queue drained on purpose
    int resume_blocks = 6;        // resumption after the drain
    uint32_t block_frames = 480;  // 10 ms at 48 kHz
    uint32_t rate = 48000;
};

struct Capture {
    IMMDevice* device = nullptr;
    IAudioClient* client = nullptr;
    IAudioCaptureClient* capture = nullptr;
    WAVEFORMATEX* format = nullptr;
    bool started = false;
    bool float32_samples = false;  // validated in open(): float32, otherwise PCM16
    std::vector<double> window_ms_at;  // arrival time of each classified window, relative to playback start
    std::vector<double> window_share;
    std::vector<double> window_440;   // own signal (isolation control)
    std::vector<double> window_880;   // interfering process's signal (isolation control)
    std::string isolation = "shared";
    uint32_t buffer_frames = 0;  // shared-mode endpoint buffer: the latency floor
    uint64_t frames = 0;
    std::vector<float> pending;  // samples of the window being assembled across packet boundaries
    uint64_t appended = 0;        // samples pushed into the window stream (silent packets included)
    uint64_t begin_frames = 0;    // appended count when the measured plan started
    uint64_t window_start = 0;    // appended count at the start of the window being assembled
    double pending_at_ms = 0;

    std::string open()
    {
        IMMDeviceEnumerator* enumerator = nullptr;
        HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, IID_PPV_ARGS(&enumerator));
        if (FAILED(hr)) return "no device enumerator";
        hr = enumerator->GetDefaultAudioEndpoint(eRender, eConsole, &device);
        enumerator->Release();
        if (FAILED(hr) || !device) return "no default render endpoint";
        // The endpoint client is used only to learn the mix format; the capture stream itself is activated
        // for this process tree below.
        hr = device->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, reinterpret_cast<void**>(&client));
        if (FAILED(hr) || !client) return "endpoint refuses IAudioClient";
        hr = client->GetMixFormat(&format);
        if (FAILED(hr) || !format) return "endpoint has no mix format";
        // Validate the sample representation before decoding: a 32-bit mix format is not necessarily IEEE
        // float, WAVE_FORMAT_EXTENSIBLE carries the subformat in a GUID, and 8/24-bit PCM would be misread.
        const bool extensible = format->wFormatTag == WAVE_FORMAT_EXTENSIBLE &&
                                format->cbSize >= sizeof(WAVEFORMATEXTENSIBLE) - sizeof(WAVEFORMATEX);
        static const GUID float_subformat = {0x00000003, 0x0000, 0x0010,
                                             {0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}};
        static const GUID pcm_subformat = {0x00000001, 0x0000, 0x0010,
                                           {0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}};
        const GUID subformat = extensible
            ? reinterpret_cast<const WAVEFORMATEXTENSIBLE*>(format)->SubFormat : GUID{};
        const bool float32 = (format->wFormatTag == WAVE_FORMAT_IEEE_FLOAT && format->wBitsPerSample == 32) ||
                             (extensible && std::memcmp(&subformat, &float_subformat, sizeof(GUID)) == 0 &&
                              format->wBitsPerSample == 32);
        const bool pcm16 = (format->wFormatTag == WAVE_FORMAT_PCM && format->wBitsPerSample == 16) ||
                           (extensible && std::memcmp(&subformat, &pcm_subformat, sizeof(GUID)) == 0 &&
                            format->wBitsPerSample == 16);
        if (!float32 && !pcm16) {
            static char why[96];
            std::snprintf(why, sizeof(why), "unsupported mix format tag=%u bits=%u", format->wFormatTag,
                          format->wBitsPerSample);
            return why;
        }
        float32_samples = float32;
        // Release the endpoint client and activate the capture stream for this process tree: another client
        // on the endpoint can then neither supply the apparent tone nor break a silence window.
        client->Release();
        client = nullptr;
        std::string reason;
        hr = process_loopback_client(&client, reason);
        if (FAILED(hr) || !client) return reason;
        isolation = "process";
        hr = client->Initialize(AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_LOOPBACK, 10000000, 0, format, nullptr);
        if (FAILED(hr)) {
            static char why[96];
            std::snprintf(why, sizeof(why), "process loopback capture refused (0x%08lx)",
                          static_cast<unsigned long>(hr));
            return why;
        }
        hr = client->GetService(IID_PPV_ARGS(&capture));
        if (FAILED(hr) || !capture) return "no capture client";
        UINT32 buffer = 0;
        if (SUCCEEDED(client->GetBufferSize(&buffer))) buffer_frames = buffer;
        hr = client->Start();
        if (FAILED(hr)) return "capture did not start";
        started = true;
        return {};
    }

    // Drains complete packets and classifies whole windows. Timestamps come from the device position the
    // capture reports, so the numbers are on the endpoint's clock rather than the fixture's.
    void drain(double /*unused*/)
    {
        if (!capture || !format) return;
        const auto per_window = static_cast<uint32_t>(format->nSamplesPerSec * window_ms / 1000.0);
        const UINT32 channels = format->nChannels ? format->nChannels : 1;
        const bool is_float = float32_samples;
        auto append = [&](float sample) {
            if (pending.empty()) window_start = appended;
            pending.push_back(sample);
            ++appended;
            if (per_window && pending.size() == per_window) {
                // Stamped on the sample clock: samples appended since the plan started, not when the
                // packet happened to arrive.
                window_ms_at.push_back(
                    frames_to_ms(window_start >= begin_frames ? window_start - begin_frames : 0, format->nSamplesPerSec));
                window_share.push_back(window_level(pending));
                window_440.push_back(frequency_share(pending, format->nSamplesPerSec, 440.0));
                window_880.push_back(frequency_share(pending, format->nSamplesPerSec, 880.0));
                pending.clear();
            }
        };
        UINT32 packet = 0;
        while (SUCCEEDED(capture->GetNextPacketSize(&packet)) && packet) {
            BYTE* data = nullptr;
            UINT32 got = 0;
            DWORD flags = 0;
            UINT64 position = 0;
            if (FAILED(capture->GetBuffer(&data, &got, &flags, &position, nullptr))) return;
            if (flags & AUDCLNT_BUFFERFLAGS_SILENT) {
                // A silent packet is still time on the endpoint. Skipping it spliced the stream across
                // gaps, which made a short tone look like it lasted for hundreds of milliseconds.
                for (UINT32 i = 0; i < got; ++i) append(0.0f);
            } else if (data) {
                for (UINT32 i = 0; i < got; ++i) {
                    double sum = 0;
                    for (UINT32 c = 0; c < channels; ++c) {
                        const size_t at = static_cast<size_t>(i) * channels + c;
                        sum += is_float ? static_cast<double>(reinterpret_cast<const float*>(data)[at])
                                        : static_cast<double>(reinterpret_cast<const int16_t*>(data)[at]) / 32768.0;
                    }
                    append(static_cast<float>(sum / channels));
                }
            }
            capture->ReleaseBuffer(got);
            frames += got;
        }
    }
    ~Capture()
    {
        if (client && started) client->Stop();
        release(capture);
        release(client);
        release(device);
        if (format) CoTaskMemFree(format);
    }
};

// The caller's plan, played through the real provider. Nothing about the provider is altered here.
struct Caller {
    IXAudio2* engine = nullptr;
    IXAudio2MasteringVoice* master = nullptr;
    IXAudio2SourceVoice* voice = nullptr;
    std::vector<int16_t> tone, silence;

    std::string open()
    {
        HRESULT hr = XAudio2Create(&engine, 0, XAUDIO2_DEFAULT_PROCESSOR);
        if (FAILED(hr) || !engine) return "XAudio2Create failed";
        hr = engine->CreateMasteringVoice(&master);
        if (FAILED(hr) || !master) return "CreateMasteringVoice failed";
        WAVEFORMATEX format{};
        format.wFormatTag = WAVE_FORMAT_PCM;
        format.nChannels = 1;
        format.nSamplesPerSec = 48000;
        format.wBitsPerSample = 16;
        format.nBlockAlign = static_cast<WORD>(format.nChannels * format.wBitsPerSample / 8);
        format.nAvgBytesPerSec = format.nSamplesPerSec * format.nBlockAlign;
        hr = engine->CreateSourceVoice(&voice, &format);
        if (FAILED(hr) || !voice) return "CreateSourceVoice failed";
        return {};
    }
    void fill(uint32_t frames)
    {
        tone.resize(frames);
        for (uint32_t i = 0; i < frames; ++i)
            tone[i] = static_cast<int16_t>(3200.0 * std::sin(2.0 * 3.14159265358979323846 * tone_hz * i / 48000.0));
        silence.assign(frames, 0);
    }
    bool submit(const std::vector<int16_t>& pcm)
    {
        XAUDIO2_BUFFER buffer{};
        buffer.AudioBytes = static_cast<UINT32>(pcm.size() * sizeof(int16_t));
        buffer.pAudioData = reinterpret_cast<const BYTE*>(pcm.data());
        return SUCCEEDED(voice->SubmitSourceBuffer(&buffer));
    }
    ~Caller()
    {
        if (voice) voice->DestroyVoice();
        if (master) master->DestroyVoice();
        release(engine);
    }
};


// Device-free self-test of the metric the isolation control uses. It runs on hosts where process loopback is
// unavailable, so the classification is still checked when the endpoint arm skips. The discrimination check is
// the control: a 440 Hz tone must score high at 440 and low at 880, otherwise "our tone was present" would be
// an unfalsifiable statement.
int classifier_self_test()
{
    const uint32_t rate = 48000;
    const std::size_t samples = 4800;  // 100 ms
    auto tone = [&](double hz) {
        std::vector<float> mono(samples);
        for (std::size_t i = 0; i < samples; ++i)
            mono[i] = static_cast<float>(0.5 * std::sin(2.0 * 3.14159265358979323846 * hz * i / rate));
        return mono;
    };
    int failures = 0;
    auto require = [&](bool condition, const char* what) {
        if (!condition) {
            std::printf("FAIL: %s\n", what);
            ++failures;
        }
    };

    const auto at440 = tone(440.0);
    const auto at880 = tone(880.0);
    const double own = frequency_share(at440, rate, 440.0);
    const double own_other = frequency_share(at440, rate, 880.0);
    const double other = frequency_share(at880, rate, 880.0);
    const double other_own = frequency_share(at880, rate, 440.0);
    std::printf("classifier: 440 tone -> 440 share %.3f, 880 share %.3f\n", own, own_other);
    std::printf("classifier: 880 tone -> 880 share %.3f, 440 share %.3f\n", other, other_own);
    require(own > 0.30, "a 440 Hz tone does not score high at 440 Hz");
    require(other > 0.30, "an 880 Hz tone does not score high at 880 Hz");
    require(own > own_other * 8.0, "a 440 Hz tone is not separated from 880 Hz");
    require(other > other_own * 8.0, "an 880 Hz tone is not separated from 440 Hz");

    const std::vector<float> silence(samples, 0.0f);
    const double silent = frequency_share(silence, rate, 440.0);
    std::printf("classifier: silence -> 440 share %.3f\n", silent);
    require(silent <= 0.01, "silence scores as tone energy, so the gap check could never pass");

    // White noise must not be mistaken for a tone; otherwise the control would accept any capture.
    std::vector<float> noise(samples);
    uint32_t seed = 12345;
    for (auto& sample : noise) {
        seed = seed * 1664525u + 1013904223u;
        sample = static_cast<float>(((seed >> 8) & 0xFFFF) / 65536.0) - 0.5f;
    }
    const double noisy = frequency_share(noise, rate, 440.0);
    std::printf("classifier: white noise -> 440 share %.3f\n", noisy);
    require(noisy < 0.05, "white noise scores as a tone");

    require(frequency_share({}, rate, 440.0) == 0.0, "an empty window scores non-zero");
    require(frequency_share(at440, 0, 440.0) == 0.0, "a zero sample rate scores non-zero");

    if (failures) return 1;
    std::printf("PASS: 440 and 880 tones separated, silence and noise do not score as tone\n");
    return 0;
}

int run()
{
    Capture capture;
    const std::string capture_error = capture.open();
    if (!capture_error.empty()) {
        std::printf("skip: %s\n", capture_error.c_str());
        return skip_no_endpoint;
    }
    // Isolation control: a second process on the endpoint plays 880 Hz for the whole run. A capture scoped
    // to this process must not see it; a shared capture would show it during our silence.
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION interference{};
    wchar_t self[MAX_PATH]{};
    if (!GetModuleFileNameW(nullptr, self, MAX_PATH)) {
        std::printf("skip: own path unavailable for the isolation control\n");
        return skip_no_endpoint;
    }
    std::wstring command = std::wstring(L"\"") + self + L"\" noise";
    if (!CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &startup,
                        &interference)) {
        std::printf("skip: interference process unavailable for the isolation control\n");
        return skip_no_endpoint;
    }
    Caller caller;
    const std::string caller_error = caller.open();
    if (!caller_error.empty()) {
        std::printf("skip: %s\n", caller_error.c_str());
        return skip_no_endpoint;
    }
    // Every phase must outlast the endpoint buffer, otherwise the whole plan is smeared into one blob and
    // nothing about tone, silence or resumption can be told apart. Sizing from the buffer keeps the fixture
    // honest on machines with very different shared-mode latencies.
    Plan plan;
    {
        const uint32_t blocks = (std::max)(3u, capture.buffer_frames * 2 / plan.block_frames);
        plan.tone_blocks = static_cast<int>(blocks);
        plan.silence_blocks = static_cast<int>(blocks);
        plan.withheld_ms = static_cast<int>(1000.0 * blocks * plan.block_frames / plan.rate);
        plan.resume_blocks = static_cast<int>(blocks);
        std::printf("plan: buffer=%u frames (~%.0f ms), %d blocks per phase (%.0f ms), withheld %d ms\n",
                    capture.buffer_frames, 1000.0 * capture.buffer_frames / plan.rate, plan.tone_blocks,
                    1000.0 * plan.tone_blocks * plan.block_frames / plan.rate, plan.withheld_ms);
    }
    caller.fill(plan.block_frames);
    const double block_ms = 1000.0 * plan.block_frames / plan.rate;
    std::printf("endpoint: capture %u Hz %u ch %u bits, isolation=%s, block=%.1f ms\n", capture.format->nSamplesPerSec,
                capture.format->nChannels, capture.format->wBitsPerSample, capture.isolation.c_str(), block_ms);

    caller.voice->Start(0);
    double begin = now_ms();
    auto serve = [&](double until) {
        while (now_ms() < until) {
            Sleep(2);
            capture.drain(now_ms() - begin);
        }
        capture.drain(now_ms() - begin);
    };
    // Two things have to be true before the plan can be measured: the endpoint must be quiet (no audio
    // left from a previous run) and it must already be streaming (a stopped shared-mode endpoint takes
    // hundreds of milliseconds to deliver its first loopback packet, which would otherwise be read as a
    // late plan). Probe with a short tone burst, measure that startup latency, then drain to quiet and
    // measure from a clean slate. The latency is reported so the timings can be read in context.
    double stream_latency_ms = -1;
    {
        // Quiet first: submit silence until the capture shows a quiet run, so stale audio cannot wake it.
        const double quiet_start = now_ms();
        bool quiet = false;
        while (!quiet && now_ms() - quiet_start < 4000.0) {
            if (!caller.submit(caller.silence)) return 2;
            serve(now_ms() + block_ms);
            const size_t have = capture.window_share.size();
            if (have < 5) continue;
            quiet = true;
            for (size_t i = have - 5; i < have; ++i)
                if (capture.window_share[i] >= 0.005) quiet = false;
        }
        if (!quiet) {
            std::printf("FAIL: endpoint never went quiet within 4 s; cannot establish a baseline\n");
            return 1;
        }
        // Probe: play tone until the capture sees it. Whatever delay this takes is the stream-start latency.
        const double probe_at = now_ms();
        const size_t probe_from = capture.window_share.size();
        while (now_ms() - probe_at < 4000.0 && stream_latency_ms < 0) {
            if (!caller.submit(caller.tone)) return 2;
            serve(now_ms() + block_ms);
            const size_t have = capture.window_share.size();
            for (size_t i = probe_from + 2; i < have; ++i)
                if (capture.window_share[i] >= 0.02 && capture.window_share[i - 1] >= 0.02 &&
                    capture.window_share[i - 2] >= 0.02) {
                    stream_latency_ms = capture.window_ms_at[i - 2] - (probe_at - begin);
                    break;
                }
        }
        if (stream_latency_ms < 0) {
            std::printf("FAIL: endpoint never delivered the probe tone within 4 s\n");
            return 1;
        }
        std::printf("stream_latency: probe tone reached the capture after %.1f ms\n", stream_latency_ms);
        // Drop what is still queued instead of submitting more silence: queueing silence to wait out a
        // backlog adds that backlog to the start of the measured plan, which is what made the plan look
        // like it began hundreds of milliseconds late.
        caller.voice->Stop(0);
        caller.voice->FlushSourceBuffers();
        caller.voice->Start(0);
        for (int i = 0; i < 3; ++i) {
            if (!caller.submit(caller.silence)) return 2;
        }
        serve(now_ms() + 3 * block_ms);
        capture.window_ms_at.clear();
        capture.window_share.clear();
        capture.window_440.clear();
        capture.window_880.clear();
        capture.begin_frames = capture.appended;
        begin = now_ms();  // the capture windows were rebased above, so the plan must be scheduled
                            // from the same instant or a slow endpoint starts the plan with expired
                            // deadlines and the resumption is queued early.
        capture.pending.clear();
    }
    for (int i = 0; i < plan.tone_blocks; ++i) {
        if (!caller.submit(caller.tone)) return 2;
        serve(begin + (i + 1) * block_ms);
    }
    for (int i = 0; i < plan.silence_blocks; ++i) {
        if (!caller.submit(caller.silence)) return 2;
        serve(begin + (plan.tone_blocks + i + 1) * block_ms);
    }
    serve(now_ms() + plan.withheld_ms);
    const double resume_at = now_ms();
    for (int i = 0; i < plan.resume_blocks; ++i) {
        if (!caller.submit(caller.tone)) return 2;
        serve(now_ms() + block_ms);
    }
    serve(now_ms() + 900);  // room for a slow endpoint to reach the resumption
    // Leave the endpoint quiet: stopping the voice with tone still queued leaks audio into the next run.
    for (int i = 0; i < 4; ++i) {
        if (!caller.submit(caller.silence)) return 2;
        serve(now_ms() + block_ms);
    }
    caller.voice->Stop(0);

    const auto& shares = capture.window_share;
    if (GetEnvironmentVariableW(L"X2W_AUDIO_VALIDATION_DEBUG", nullptr, 0)) {
        const double elapsed = now_ms() - begin;
        std::printf("debug: frames=%llu elapsed=%.1fms implied_rate=%.0f Hz (capture %u Hz)\n",
                    static_cast<unsigned long long>(capture.frames), elapsed,
                    elapsed > 0 ? capture.frames * 1000.0 / elapsed : 0.0, capture.format->nSamplesPerSec);
        for (size_t i = 0; i < shares.size() && i < 40; ++i)
            std::printf("debug: w%03zu t=%.1fms share=%.4f\n", i, capture.window_ms_at[i], shares[i]);
    }
    if (shares.empty()) {
        std::printf("skip: capture produced no complete windows\n");
        return skip_no_endpoint;
    }
    // Threshold from the capture itself: halfway between the observed floor and peak, so no fixed level is
    // encoded and a quieter or louder endpoint still classifies.
    const double floor = *std::min_element(shares.begin(), shares.end());
    const double peak = *std::max_element(shares.begin(), shares.end());
    const double threshold = floor + 0.5 * (peak - floor);
    const auto is_tone = [&](size_t i) { return shares[i] >= threshold; };
    // A transition must be sustained: capture starts mid-stream and a window straddling a boundary can land
    // either side of the threshold, so single-window decisions are noise rather than evidence.
    const auto run_of = [&](size_t from, bool want, size_t length) {
        for (size_t i = from; i + length <= shares.size(); ++i) {
            bool all = true;
            for (size_t j = 0; j < length; ++j)
                if (is_tone(i + j) != want) {
                    all = false;
                    break;
                }
            if (all) return i;
        }
        return shares.size();
    };
    // The gap is the longest sustained silence after the tone, not the first dip: a single low window
    // inside the tone region (packet boundary, endpoint buffering) is noise, and taking it produced runs
    // where the "resumption" was actually the tail of the tone still queued.
    const size_t first_tone = run_of(0, true, 3);
    size_t silence_at = shares.size(), silence_end = shares.size();
    for (size_t i = first_tone; i < shares.size();) {
        if (is_tone(i)) {
            ++i;
            continue;
        }
        const size_t run_start = i;
        while (i < shares.size() && !is_tone(i)) ++i;
        const size_t length = i - run_start;
        if (length >= 3 && (silence_at == shares.size() || length > silence_end - silence_at)) {
            silence_at = run_start;
            silence_end = i;
        }
    }
    const size_t resume_index = silence_end == shares.size() ? shares.size() : run_of(silence_end, true, 3);
    const auto at_ms = [&](size_t i) { return i < capture.window_ms_at.size() ? capture.window_ms_at[i] : -1.0; };
    std::printf("windows=%zu floor=%.4f peak=%.4f threshold=%.4f\n", shares.size(), floor, peak, threshold);
    std::printf("tone_onset=%.1fms silence_at=%.1fms tone_resume=%.1fms silence_ms=%.1f resume_vs_submit=%.1fms\n",
                at_ms(first_tone), at_ms(silence_at), at_ms(resume_index),
                (silence_at != shares.size() && resume_index != shares.size()) ? at_ms(resume_index) - at_ms(silence_at) : -1.0,
                (resume_index != shares.size()) ? at_ms(resume_index) - (resume_at - begin) : -1.0);

    if (first_tone == shares.size()) {
        std::printf("FAIL: endpoint never showed the submitted tone\n");
        return 1;
    }
    if (silence_at == shares.size()) {
        std::printf("FAIL: endpoint never showed the intentional silence\n");
        return 1;
    }
    if (resume_index == shares.size()) {
        std::printf("FAIL: endpoint never resumed the tone after the drained queue\n");
        return 1;
    }
    // The plan must be visible in the measurements, within a latency envelope rather than exact numbers:
    // a shared-mode endpoint plus XAudio2's own buffering adds tens of milliseconds, but the *ordering* and
    // the size of the withheld gap are not optional. These bounds are what the earlier broken version of
    // this fixture failed to reflect, which is why it could not have been evidence.
    const double silence_ms = at_ms(resume_index) - at_ms(silence_at);
    const double onset_ms = at_ms(first_tone);
    const double resume_delay_ms = at_ms(resume_index) - (resume_at - begin);
    // The onset is a diagnostic: it is dominated by how long the endpoint took to start streaming.
    (void)onset_ms;
    // The silence the endpoint must show is the submitted silence phase plus the withheld gap, not the gap
    // alone; the earlier bound compared against the gap only and would have failed a correct run.
    const double silence_expected_ms = plan.silence_blocks * block_ms + plan.withheld_ms;
    if (silence_ms < silence_expected_ms - 600.0 || silence_ms > silence_expected_ms + 600.0) {
        std::printf("FAIL: measured silence %.1f ms does not match the planned %.1f ms "
                    "(silence phase %d ms + withheld %d ms)\n",
                    silence_ms, silence_expected_ms, static_cast<int>(plan.silence_blocks * block_ms),
                    plan.withheld_ms);
        return 1;
    }
    // Same for the resumption delay: reported above, not asserted, for the same reason.
    (void)resume_delay_ms;
    // Isolation: while our own process withheld audio, the only sound on the endpoint was the
    // interference process at 880 Hz. A process-scoped capture must show silence there, and the captured
    // tone must be our own signal rather than the interferer's.
    double gap_level = 0, tone_440 = 0, tone_880 = 0;
    for (size_t i = silence_at; i < resume_index && i < shares.size(); ++i)
        gap_level = (std::max)(gap_level, shares[i]);
    for (size_t i = first_tone; i < silence_at && i < shares.size(); ++i) {
        tone_440 = (std::max)(tone_440, capture.window_440[i]);
        tone_880 = (std::max)(tone_880, capture.window_880[i]);
    }
    std::printf("isolation(%s): gap_level=%.4f tone_440=%.4f tone_880=%.4f\n", capture.isolation.c_str(), gap_level,
                tone_440, tone_880);
    if (gap_level > 0.01) {
        std::printf("FAIL: the interfering process reached the capture (gap level %.4f); the capture is not "
                    "scoped to this process\n", gap_level);
        return 1;
    }
    if (tone_440 < 0.05 || tone_440 <= tone_880) {
        std::printf("FAIL: the captured tone is not the fixture's own signal (440 share %.4f, 880 share %.4f)\n",
                    tone_440, tone_880);
        return 1;
    }
    TerminateProcess(interference.hProcess, 0);
    CloseHandle(interference.hProcess);
    CloseHandle(interference.hThread);
    std::printf("PASS: tone, intentional silence and resumption all observed at the endpoint (isolation=%s)\n",
                capture.isolation.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    if (FAILED(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED))) {
        std::printf("skip: COM initialisation failed\n");
        return skip_no_endpoint;
    }
    if (argc > 1 && std::strcmp(argv[1], "--self-test") == 0) {
        CoUninitialize();
        return classifier_self_test();
    }
    if (argc > 1 && std::strcmp(argv[1], "noise") == 0) {
        // Interference process for the isolation control: a second process on the endpoint playing a
        // distinguishable tone. A capture scoped to the fixture's process tree must not see it.
        Caller noise;
        if (!noise.open().empty()) {
            CoUninitialize();
            return 3;
        }
        std::vector<int16_t> pcm(480);
        for (int i = 0; i < 480; ++i)
            pcm[i] = static_cast<int16_t>(3200.0 * std::sin(2.0 * 3.14159265358979323846 * 880.0 * i / 48000.0));
        noise.voice->Start(0);
        for (int block = 0; block < 2500; ++block) {  // ~25 s, longer than the fixture run
            if (!noise.submit(pcm)) break;
            Sleep(10);
        }
        CoUninitialize();
        return 0;
    }
    const int result = run();
    CoUninitialize();
    return result;
}
