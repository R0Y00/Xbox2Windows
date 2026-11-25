#pragma once

// Root descriptor views (CBV, SRV, UAV) a title sets on a command list, for the slots that carry them:
// compute CBV 41, compute SRV 43, compute UAV 45 and graphics UAV 46 (desktop 37, 39, 41, 42 plus four).
// Plain header with no Windows types so the host-only test can build it.
//
// A view is accepted only when a root signature of the matching kind is set, the parameter is below
// root_view_parameters and the root signature declares that parameter with the same view type. Anything
// else is refused and the store is left as it was. Setting a new root signature clears the store.

#include <cstdint>

namespace x2w_startup {

constexpr unsigned root_view_parameters = 16;

// D3D12_ROOT_PARAMETER_TYPE values.
constexpr unsigned root_type_cbv = 2;
constexpr unsigned root_type_srv = 3;
constexpr unsigned root_type_uav = 4;

enum class RootViewResult { stored, no_root, parameter_out_of_range, wrong_type };

inline const char* root_view_reason(RootViewResult result)
{
    switch (result) {
    case RootViewResult::stored: return "stored";
    case RootViewResult::no_root: return "no root signature set";
    case RootViewResult::parameter_out_of_range: return "parameter out of range";
    case RootViewResult::wrong_type: return "parameter declared as another type";
    }
    return "unknown";
}

struct RootViewStore {
    std::uint64_t address[root_view_parameters] = {};  // title address, 0 when never set or the address was unreadable
    std::uint32_t bound = 0;                           // bit per parameter that holds a view

    void clear() { *this = RootViewStore{}; }

    // True when every parameter in required (one bit each) holds a view. A Dispatch that needs these roots
    // must be recorded only when this holds.
    bool covers(std::uint32_t required) const { return (bound & required) == required; }

    // declared: the type the root signature gives parameter, or ~0u when it has no such parameter.
    RootViewResult set(bool root_set, unsigned declared, unsigned wanted, unsigned parameter, std::uint64_t value)
    {
        if (!root_set) return RootViewResult::no_root;
        if (parameter >= root_view_parameters || declared == ~0u) return RootViewResult::parameter_out_of_range;
        if (declared != wanted) return RootViewResult::wrong_type;
        address[parameter] = value;
        if (value) bound |= 1u << parameter;
        else bound &= ~(1u << parameter);
        return RootViewResult::stored;
    }
};

}  // namespace x2w_startup
