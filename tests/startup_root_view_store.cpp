// Host-only check of src/startup_stubs/root_view_store.h: which root view calls a list accepts and what the
// store holds afterwards.
#include "startup_stubs/root_view_store.h"

#include <cstdio>

using namespace x2w_startup;

namespace {
int failures;
void expect(bool ok, const char* what)
{
    if (ok) return;
    ++failures;
    std::printf("FAIL %s\n", what);
}
}  // namespace

int main()
{
    RootViewStore store;
    expect(store.set(false, root_type_cbv, root_type_cbv, 0, 0x1000) == RootViewResult::no_root, "no root refused");
    expect(store.bound == 0 && store.address[0] == 0, "no root leaves the store empty");

    expect(store.set(true, root_type_cbv, root_type_cbv, 3, 0x2000) == RootViewResult::stored, "cbv stored");
    expect(store.address[3] == 0x2000 && store.bound == (1u << 3), "cbv address and bit");

    expect(store.set(true, root_type_srv, root_type_uav, 3, 0x3000) == RootViewResult::wrong_type, "uav on srv parameter refused");
    expect(store.address[3] == 0x2000, "refused call keeps the old address");

    expect(store.set(true, ~0u, root_type_srv, 5, 0x4000) == RootViewResult::parameter_out_of_range, "undeclared parameter refused");
    expect(store.set(true, root_type_srv, root_type_srv, root_view_parameters, 0x4000) == RootViewResult::parameter_out_of_range,
           "parameter 16 refused");
    expect(store.set(true, root_type_srv, root_type_srv, 15, 0x5000) == RootViewResult::stored, "last parameter stored");
    expect(store.bound == ((1u << 3) | (1u << 15)), "two bits set");

    expect(store.set(true, root_type_cbv, root_type_cbv, 3, 0) == RootViewResult::stored, "zero address stored");
    expect(store.bound == (1u << 15) && store.address[3] == 0, "zero address clears the bit");

    store.clear();
    expect(store.bound == 0 && store.address[15] == 0, "clear empties the store");

    if (failures) return 1;
    std::printf("root view store ok\n");
    return 0;
}
