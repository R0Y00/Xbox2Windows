#pragma once
// A translated buffer binding. records is the descriptor NUM_RECORDS (or a created view's element count).
// source is 0 for an alias or any other backing, 1 for a GPU copy this list made, 2 for a snapshot of title
// memory, and 3 for zeros that stand in for a null descriptor.
// Members are assigned by name. A brace initializer put the upload flag in records and narrowed the next
// integer into the bool.
#include <cstdint>

template <class Resource>
struct IndirectBufferViewOf {
    Resource* resource = nullptr;
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
    unsigned stride = 0;
    unsigned records = 0;
    bool upload = false;
    int source = 0;
};

template <class Resource>
IndirectBufferViewOf<Resource> indirect_buffer_view(Resource* resource, std::uint64_t offset, std::uint64_t bytes,
                                                    unsigned stride, unsigned records, bool upload, int source)
{
    IndirectBufferViewOf<Resource> view;
    view.resource = resource;
    view.offset = offset;
    view.bytes = bytes;
    view.stride = stride;
    view.records = records;
    view.upload = upload;
    view.source = source;
    return view;
}
