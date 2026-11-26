#pragma once

// The DXIL validator in dxil.dll (IDxcValidator), for tests and the offline batch. Translated shaders do
// not need it: they carry the container hash we compute (container::hash).
//
// The DLL is found at X2W_SHADERCFG_DXIL if set, else the newest Windows SDK's bin\<version>\x64\dxil.dll,
// else wherever LoadLibrary finds it.
//
// Creation returns nullopt and writes an error string on failure. Validation returns the rejection message
// on failure and nullopt on success.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>

namespace x2w::vs::validator {

class Validator {
public:
    // Where dxil.dll is looked for.
    static std::optional<std::filesystem::path> default_path();

    // Loads path's validator (or dxil.dll from the DLL search path when path is nullopt).
    // nullopt on failure, with the reason in *error if error is given.
    static std::optional<Validator> load(const std::optional<std::filesystem::path>& path,
                                         std::string* error = nullptr);

    // Validates container and signs it in place (DxcValidatorFlags_InPlaceEdit: the validator
    // writes its hash into the container's digest); the error text if it is rejected.
    std::optional<std::string> validate(std::span<std::uint8_t> container);

    Validator(Validator&& o) noexcept : object_(o.object_) { o.object_ = nullptr; }
    Validator& operator=(Validator&& o) noexcept;
    Validator(const Validator&) = delete;
    Validator& operator=(const Validator&) = delete;
    ~Validator();

private:
    explicit Validator(void* object) : object_(object) {}
    // The IDxcValidator; we own one reference. The validator object is free-threaded (DXC's validator has no
    // thread affinity); calls are serialised by the caller.
    void* object_ = nullptr;
};

}  // namespace x2w::vs::validator
