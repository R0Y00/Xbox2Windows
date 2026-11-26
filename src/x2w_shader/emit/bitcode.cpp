// LLVM bitcode module and writer.

#include "x2w_shader/emit/bitcode.h"

#include <stdexcept>

#include "x2w_shader/emit/census.h"

namespace x2w::vs::bitcode {

void bug(const char* what) { throw std::logic_error(what); }

namespace {

void hash_mix(std::size_t& h, std::uint64_t v) {
    // FNV-style mixing; only lookup speed depends on it.
    h ^= std::size_t(v) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
}

}  // namespace

std::size_t TypeDefHash::operator()(const TypeDef& t) const noexcept {
    std::size_t h = t.kind;
    hash_mix(h, t.n);
    hash_mix(h, t.of.id);
    if (t.name) hash_mix(h, std::hash<std::string>{}(*t.name));
    for (Ty x : t.list) hash_mix(h, x.id);
    return h;
}

std::size_t ConstKeyHash::operator()(const std::pair<Ty, Const>& k) const noexcept {
    std::size_t h = k.first.id;
    hash_mix(h, k.second.kind);
    hash_mix(h, std::uint64_t(k.second.value));
    hash_mix(h, k.second.bits);
    for (std::uint32_t x : k.second.ids) hash_mix(h, x);
    return h;
}

Ty Module::ty(const TypeDef& def) {
    auto it = type_ids.find(def);
    if (it != type_ids.end()) return Ty{it->second};
    auto t = std::uint32_t(types.size());
    types.push_back(def);
    type_ids.emplace(def, t);
    return Ty{t};
}

Ty Module::void_() { return ty(TypeDef{TypeDef::Void}); }
Ty Module::int_(std::uint32_t bits) { return ty(TypeDef{TypeDef::Int, bits}); }
Ty Module::float_() { return ty(TypeDef{TypeDef::Float}); }
Ty Module::double_() { return ty(TypeDef{TypeDef::Double}); }
Ty Module::ptr(Ty to) { return ty(TypeDef{TypeDef::Ptr, 0, to}); }
Ty Module::ptr_in(Ty to, std::uint32_t space) { return ty(TypeDef{TypeDef::Ptr, space, to}); }
Ty Module::array(std::uint32_t n, Ty of) { return ty(TypeDef{TypeDef::Array, n, of}); }
Ty Module::named_struct(std::string_view name, std::vector<Ty> fields) {
    return ty(TypeDef{TypeDef::Struct, 0, Ty{}, std::string(name), std::move(fields)});
}
Ty Module::func_ty(Ty ret, std::vector<Ty> params) {
    return ty(TypeDef{TypeDef::Func, 0, ret, std::nullopt, std::move(params)});
}

bool Module::is_void(Ty t) const { return types[t.id].kind == TypeDef::Void; }

V Module::constant(Ty t, const Const& c) {
    auto key = std::make_pair(t, c);
    auto it = const_ids.find(key);
    if (it != const_ids.end()) return V::constant(it->second);
    auto i = std::uint32_t(consts.size());
    consts.push_back(key);
    const_ids.emplace(std::move(key), i);
    return V::constant(i);
}

V Module::int_const(Ty t, std::int64_t v) {
    Const c;
    c.kind = Const::Int;
    c.value = v;
    return constant(t, c);
}

V Module::i32(std::uint32_t v) {
    Ty t = int_(32);
    return int_const(t, std::int64_t(std::int32_t(v)));
}

V Module::i8(std::uint8_t v) {
    Ty t = int_(8);
    return int_const(t, std::int64_t(std::int8_t(v)));
}

V Module::bool_(bool v) {
    Ty t = int_(1);
    return int_const(t, -std::int64_t(v));
}

V Module::f32(std::uint32_t bits) {
    Ty t = float_();
    Const c;
    c.kind = Const::Float;
    c.bits = bits;
    return constant(t, c);
}

V Module::undef(Ty t) {
    Const c;
    c.kind = Const::Undef;
    return constant(t, c);
}

V Module::aggregate(Ty t, std::span<const V> fields) {
    Const c;
    c.kind = Const::Aggregate;
    c.ids.reserve(fields.size());
    for (V f : fields) {
        if (f.kind != V::Const) bug("aggregate of a non-constant");
        c.ids.push_back(f.i);
    }
    return constant(t, c);
}

std::optional<std::int64_t> Module::int_value(V v) const {
    if (v.kind != V::Const) return std::nullopt;
    const Const& c = consts[v.i].second;
    if (c.kind != Const::Int) return std::nullopt;
    return c.value;
}

void Module::replace_const(V old, V new_) {
    // Replacing the slot contents preserves the existing lookup key.
    if (old.kind == V::Const && new_.kind == V::Const) consts[old.i] = consts[new_.i];
}

V Module::function(std::string_view name, Ty t, std::optional<Attr> attr, bool body) {
    std::string key(name);
    auto it = func_ids.find(key);
    if (it != func_ids.end()) return V::func(it->second);
    auto i = std::uint32_t(funcs.size());
    funcs.push_back(Func{key, t, attr, body});
    func_ids.emplace(std::move(key), i);
    return V::func(i);
}

V Module::global(std::string_view name, Ty t, std::uint32_t space) {
    Ty p = ptr_in(t, space);
    globals.push_back(Global{std::string(name), t, p, space, std::nullopt});
    return V::global(std::uint32_t(globals.size()) - 1);
}

V Module::constant_global(std::string_view name, V init) {
    if (init.kind != V::Const) bug("a global initialised with a non-constant");
    Ty t = consts[init.i].first;
    V g = global(name, t, 0);
    globals[g.i].init = init.i;
    return g;
}

void Module::tag(V v, Tag t) {
    if (v.kind == V::Inst && census::enabled()) tags[v.i] = t;
}

std::size_t Module::new_block() {
    blocks.emplace_back();
    return blocks.size() - 1;
}

Ty Module::type_of(V v) const {
    switch (v.kind) {
    case V::Global:
        return globals[v.i].ptr;
    case V::Const:
        return consts[v.i].first;
    case V::Inst:
        return insts[v.i].ty;
    case V::Func: {
        Ty t = funcs[v.i].ty;
        auto it = type_ids.find(TypeDef{TypeDef::Ptr, 0, t});
        if (it == type_ids.end()) bug("function pointer type");
        return Ty{it->second};
    }
    }
    bug("type_of");
}

V Module::push(std::size_t block, Ty t, Op op) {
    auto i = std::uint32_t(insts.size());
    insts.push_back(Inst{t, std::move(op)});
    // Terminators are kept apart, so a block can still grow at its end.
    blocks[block].body.push_back(i);
    return V::inst(i);
}

V Module::bin(std::size_t block, Bin op, V a, V b) {
    Ty t = type_of(a);
    Op o;
    o.kind = Op::Bin;
    o.code = std::uint8_t(op);
    o.a = a;
    o.b = b;
    return push(block, t, std::move(o));
}

V Module::cast(std::size_t block, Cast op, V v, Ty to) {
    Op o;
    o.kind = Op::Cast;
    o.code = std::uint8_t(op);
    o.a = v;
    return push(block, to, std::move(o));
}

V Module::cmp(std::size_t block, std::uint8_t p, V a, V b) {
    Ty t = int_(1);
    Op o;
    o.kind = Op::Cmp;
    o.code = p;
    o.a = a;
    o.b = b;
    return push(block, t, std::move(o));
}

V Module::select(std::size_t block, V c, V t, V f) {
    Ty ty_ = type_of(t);
    Op o;
    o.kind = Op::Select;
    o.a = c;
    o.b = t;
    o.c = f;
    return push(block, ty_, std::move(o));
}

V Module::extract(std::size_t block, V agg, std::uint32_t index) {
    const TypeDef& d = types[type_of(agg).id];
    Ty t;
    if (d.kind == TypeDef::Struct)
        t = d.list[index];
    else if (d.kind == TypeDef::Array)
        t = d.of;
    else
        bug("extractvalue from a non-aggregate");
    Op o;
    o.kind = Op::Extract;
    o.a = agg;
    o.index = index;
    return push(block, t, std::move(o));
}

V Module::call(std::size_t block, V f, std::vector<V> args) {
    if (f.kind != V::Func) bug("call of a non-function");
    const TypeDef& d = types[funcs[f.i].ty.id];
    if (d.kind != TypeDef::Func) bug("unreachable: function type");
    Ty ret = d.of;
    Op o;
    o.kind = Op::Call;
    o.index = f.i;
    o.list = std::move(args);
    return push(block, ret, std::move(o));
}

V Module::phi(std::size_t block, Ty t) {
    auto i = std::uint32_t(insts.size());
    Op o;
    o.kind = Op::Phi;
    insts.push_back(Inst{t, std::move(o)});
    blocks[block].phis.push_back(i);
    return V::inst(i);
}

V Module::element(std::size_t block, V p, V index) {
    const TypeDef pd = types[type_of(p).id];
    if (pd.kind != TypeDef::Ptr) bug("element of a non-pointer");
    Ty arr = pd.of;
    std::uint32_t space = pd.n;
    const TypeDef& ad = types[arr.id];
    if (ad.kind != TypeDef::Array) bug("element of a non-array");
    Ty elem = ad.of;
    Ty t = ptr_in(elem, space);
    V zero = i32(0);
    Op o;
    o.kind = Op::Gep;
    o.ty = arr;
    o.a = p;
    o.list = {zero, index};
    return push(block, t, std::move(o));
}

V Module::load(std::size_t block, V p) {
    const TypeDef& d = types[type_of(p).id];
    if (d.kind != TypeDef::Ptr) bug("load from a non-pointer");
    Ty t = d.of;
    Op o;
    o.kind = Op::Load;
    o.a = p;
    return push(block, t, std::move(o));
}

void Module::store(std::size_t block, V p, V value) {
    Ty v = void_();
    Op o;
    o.kind = Op::Store;
    o.a = p;
    o.b = value;
    push(block, v, std::move(o));
}

V Module::atomic(std::size_t block, Rmw op, V p, V value) {
    Ty t = type_of(value);
    Op o;
    o.kind = Op::AtomicRmw;
    o.code = std::uint8_t(op);
    o.a = p;
    o.b = value;
    return push(block, t, std::move(o));
}

V Module::cmpxchg(std::size_t block, V p, V expected, V new_) {
    Ty i32t = type_of(new_);
    Ty i1 = int_(1);
    Ty pair = ty(TypeDef{TypeDef::Struct, 0, Ty{}, std::nullopt, {i32t, i1}});
    Op o;
    o.kind = Op::CmpXchg;
    o.a = p;
    o.b = expected;
    o.c = new_;
    V r = push(block, pair, std::move(o));
    return extract(block, r, 0);
}

void Module::set_phi(V phi_, std::vector<std::pair<V, std::size_t>> incoming) {
    if (phi_.kind != V::Inst) bug("set_phi of a non-instruction");
    Op& o = insts[phi_.i].op;
    o = Op{};
    o.kind = Op::Phi;
    o.incoming = std::move(incoming);
}

void Module::terminate(std::size_t block, Term t) { blocks[block].term = t; }

Md Module::md_str(std::string_view s) {
    MdDef d;
    d.kind = MdDef::Str;
    d.str = std::string(s);
    mds.push_back(std::move(d));
    return Md{std::uint32_t(mds.size()) - 1};
}

Md Module::md_value(V v) {
    MdDef d;
    d.kind = MdDef::Value;
    d.ty = type_of(v);
    d.v = v;
    mds.push_back(std::move(d));
    return Md{std::uint32_t(mds.size()) - 1};
}

Md Module::md_node(std::vector<std::optional<Md>> ops) {
    MdDef d;
    d.kind = MdDef::Node;
    d.ops = std::move(ops);
    mds.push_back(std::move(d));
    return Md{std::uint32_t(mds.size()) - 1};
}

void Module::named_md(std::string_view name, std::vector<Md> nodes) {
    named.emplace_back(std::string(name), std::move(nodes));
}

std::vector<bool> Module::live() const {
    std::vector<bool> lv(insts.size(), false);
    std::vector<std::uint32_t> work;
    auto mark = [&](V v) {
        if (v.kind == V::Inst && !lv[v.i]) {
            lv[v.i] = true;
            work.push_back(v.i);
        }
    };
    for (const Block& b : blocks) {
        for (std::uint32_t i : b.body) mark(V::inst(i));
        if (b.term && b.term->kind == Term::CondBr) mark(b.term->cond);
    }
    while (!work.empty()) {
        std::uint32_t i = work.back();
        work.pop_back();
        for_each_operand(insts[i].op, mark);
    }
    return lv;
}

namespace {

constexpr std::uint64_t MODULE_BLOCK = 8;
constexpr std::uint64_t PARAMATTR_BLOCK = 9;
constexpr std::uint64_t PARAMATTR_GROUP_BLOCK = 10;
constexpr std::uint64_t CONSTANTS_BLOCK = 11;
constexpr std::uint64_t FUNCTION_BLOCK = 12;
constexpr std::uint64_t VALUE_SYMTAB_BLOCK = 14;
constexpr std::uint64_t METADATA_BLOCK = 15;
constexpr std::uint64_t TYPE_BLOCK = 17;

std::vector<std::uint64_t> chars(std::string_view s) {
    std::vector<std::uint64_t> r;
    r.reserve(s.size());
    for (unsigned char c : s) r.push_back(c);
    return r;
}

}  // namespace

Succs successors(const std::optional<Term>& term) {
    Succs s;
    if (term && term->kind == Term::Br) {
        s.to[0] = term->t;
        s.len = 1;
    } else if (term && term->kind == Term::CondBr) {
        s.to[0] = term->t;
        s.to[1] = term->f;
        s.len = 2;
    }
    return s;
}

std::uint64_t signed_vbr(std::int64_t v) {
    if (v >= 0) return std::uint64_t(v) << 1;
    std::uint64_t mag = 0 - std::uint64_t(v);
    return (mag << 1) | 1;
}

std::vector<std::uint8_t> Module::write() const {
    Writer w;
    const std::uint8_t magic[4] = {'B', 'C', 0xC0, 0xDE};
    w.bytes(magic, 4);
    w.enter(MODULE_BLOCK, 3);
    w.record(1, {1});  // VERSION: relative value ids

    // Attribute groups 1-4 and one list per group.
    w.enter(PARAMATTR_GROUP_BLOCK, 3);
    {
        const std::vector<std::pair<std::uint64_t, std::vector<std::uint64_t>>> groups = {
            {1, {18, 20}}, {2, {18}}, {3, {18, 21}}, {4, {12, 18}}};
        for (const auto& [id, kinds] : groups) {
            std::vector<std::uint64_t> r = {id, 0xFFFFFFFFull};
            for (std::uint64_t k : kinds) {
                r.push_back(0);
                r.push_back(k);
            }
            w.record(3, r);
        }
    }
    w.exit();
    w.enter(PARAMATTR_BLOCK, 3);
    for (std::uint64_t id = 1; id <= 4; ++id) w.record(2, {id});
    w.exit();

    w.enter(TYPE_BLOCK, 4);
    w.record(1, {std::uint64_t(types.size())});
    for (const TypeDef& t : types) {
        switch (t.kind) {
        case TypeDef::Void:
            w.record(2, {});
            break;
        case TypeDef::Float:
            w.record(3, {});
            break;
        case TypeDef::Double:
            w.record(4, {});
            break;
        case TypeDef::Int:
            w.record(7, {std::uint64_t(t.n)});
            break;
        case TypeDef::Ptr:
            w.record(8, {std::uint64_t(t.of.id), std::uint64_t(t.n)});
            break;
        case TypeDef::Array:
            w.record(11, {std::uint64_t(t.n), std::uint64_t(t.of.id)});
            break;
        case TypeDef::Struct: {
            std::vector<std::uint64_t> r = {0};
            for (Ty f : t.list) r.push_back(f.id);
            if (t.name) {
                w.record(19, chars(*t.name));
                w.record(20, r);
            } else {
                w.record(18, r);
            }
            break;
        }
        case TypeDef::Func: {
            std::vector<std::uint64_t> r = {0, std::uint64_t(t.of.id)};
            for (Ty p : t.list) r.push_back(p.id);
            w.record(21, r);
            break;
        }
        }
    }
    w.exit();

    w.record(2, chars("dxil-ms-dx"));
    w.record(3, chars("e-m:e-p:32:32-i1:32-i8:32-i16:32-i32:32-i64:64-f16:32-f32:32-f64:64-n8:16:32:64"));
    const std::uint64_t first_const = globals.size() + funcs.size();
    for (const Global& g : globals) {
        // [type, address space << 2 | explicit type | constant, init id + 1, linkage (external 0,
        // internal 3), log2 alignment + 1, section]
        std::uint64_t constant = 0, init = 0, linkage = 0;
        if (g.init) {
            constant = 1;
            init = first_const + std::uint64_t(*g.init) + 1;
            linkage = 3;
        }
        w.record(7, {std::uint64_t(g.ty.id), (std::uint64_t(g.space) << 2) | 2 | constant, init, linkage, 3, 0});
    }
    for (const Func& f : funcs) {
        // [type, cc, isproto, linkage, paramattr, align, section, visibility, gc, unnamed_addr,
        // prologue, dllstorage, comdat, prefix]
        std::uint64_t attr = f.attr ? std::uint64_t(*f.attr) : 0;
        w.record(8, {std::uint64_t(f.ty.id), 0, std::uint64_t(!f.body), 0, attr, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    }

    // Value ids: globals, functions, constants, instructions.
    const auto nglobals = std::uint32_t(globals.size());
    const auto nfuncs = nglobals + std::uint32_t(funcs.size());
    auto const_id = [&](std::uint32_t i) { return std::uint64_t(nfuncs + i); };
    w.enter(CONSTANTS_BLOCK, 4);
    {
        std::optional<Ty> current;
        for (const auto& [t, c] : consts) {
            if (current != t) {
                w.record(1, {std::uint64_t(t.id)});
                current = t;
            }
            switch (c.kind) {
            case Const::Undef:
                w.record(3, {});
                break;
            case Const::Int:
                w.record(4, {signed_vbr(c.value)});
                break;
            case Const::Float:
                w.record(6, {std::uint64_t(c.bits)});
                break;
            case Const::Aggregate: {
                std::vector<std::uint64_t> r;
                r.reserve(c.ids.size());
                for (std::uint32_t i : c.ids) r.push_back(const_id(i));
                w.record(7, r);
                break;
            }
            }
        }
    }
    w.exit();

    // Phis nothing reads are left out (one register per phi makes many).
    const std::vector<bool> lv = live();

    // Blocks are laid out in reverse postorder, so every block follows its forward predecessors
    // (drivers take a branch to an earlier block for a loop's back edge).
    auto succs = [&](std::size_t b) { return successors(blocks[b].term); };
    const std::size_t nblocks = blocks.size();
    std::vector<bool> seen(nblocks, false);
    std::vector<std::size_t> post;
    std::vector<std::pair<std::size_t, std::size_t>> stack = {{0, 0}};
    seen[0] = true;
    while (!stack.empty()) {
        auto& top = stack.back();
        auto [b, k] = top;
        top.second += 1;
        Succs s = succs(b);
        if (k < s.size()) {
            std::size_t n = s.to[k];
            if (!seen[n]) {
                seen[n] = true;
                stack.push_back({n, 0});
            }
        } else {
            post.push_back(b);
            stack.pop_back();
        }
    }
    std::vector<std::size_t> rpo(post.rbegin(), post.rend());
    constexpr std::uint64_t NONE = ~std::uint64_t(0);
    std::vector<std::uint64_t> pos(nblocks, NONE);
    for (std::size_t n = 0; n < rpo.size(); ++n) pos[rpo[n]] = n;
    // An empty latch (only a branch back to its loop header) is folded into its one predecessor,
    // so the exiting block is the latch as in DXC's loops; the AMD driver hangs on divergent loops
    // otherwise.
    std::vector<std::vector<std::size_t>> preds(nblocks);
    for (std::size_t b : rpo)
        for (std::size_t s : succs(b)) preds[s].push_back(b);
    std::vector<std::optional<std::pair<std::size_t, std::size_t>>> fold(nblocks);
    for (std::size_t b : rpo) {
        const Block& blk = blocks[b];
        if (!blk.term || blk.term->kind != Term::Br) continue;
        bool empty = blk.body.empty();
        for (std::uint32_t i : blk.phis)
            if (lv[i]) empty = false;
        if (!empty || preds[b].size() != 1) continue;
        std::size_t h = blk.term->t;
        std::size_t p = preds[b][0];
        if (pos[h] < pos[b] && !succs(p).contains(h)) fold[b] = std::make_pair(p, h);
    }
    std::vector<std::size_t> layout;
    for (std::size_t b : rpo)
        if (!fold[b]) layout.push_back(b);
    std::fill(pos.begin(), pos.end(), NONE);
    for (std::size_t n = 0; n < layout.size(); ++n) pos[layout[n]] = n;
    // Branch targets and phi blocks through folded latches.
    auto target = [&](std::size_t t) { return fold[t] ? pos[fold[t]->second] : pos[t]; };
    auto from = [&](std::size_t b) { return fold[b] ? pos[fold[b]->first] : pos[b]; };

    // Instruction value ids, in layout order.
    const std::uint32_t base = nfuncs + std::uint32_t(consts.size());
    std::vector<std::uint32_t> ids(insts.size(), UINT32_MAX);
    {
        std::uint32_t next = base;
        for (std::size_t lb : layout) {
            const Block& b = blocks[lb];
            for (std::uint32_t i : b.phis) {
                if (!lv[i]) continue;
                if (!is_void(insts[i].ty)) ids[i] = next++;
            }
            for (std::uint32_t i : b.body)
                if (!is_void(insts[i].ty)) ids[i] = next++;
        }
    }
    auto abs = [&](V v) -> std::uint32_t {
        switch (v.kind) {
        case V::Global:
            return v.i;
        case V::Func:
            return nglobals + v.i;
        case V::Const:
            return nfuncs + v.i;
        case V::Inst: {
            std::uint32_t id = ids[v.i];
            if (id == UINT32_MAX) bug("use of a void value");
            return id;
        }
        }
        bug("abs");
    };

    w.enter(METADATA_BLOCK, 3);
    for (const MdDef& m : mds) {
        switch (m.kind) {
        case MdDef::Str:
            w.record(1, chars(m.str));
            break;
        case MdDef::Value:
            w.record(2, {std::uint64_t(m.ty.id), std::uint64_t(abs(m.v))});
            break;
        case MdDef::Node: {
            std::vector<std::uint64_t> r;
            r.reserve(m.ops.size());
            for (const auto& o : m.ops) r.push_back(o ? std::uint64_t(o->id) + 1 : 0);
            w.record(3, r);
            break;
        }
        }
    }
    for (const auto& [name, nodes] : named) {
        w.record(4, chars(name));
        std::vector<std::uint64_t> r;
        r.reserve(nodes.size());
        for (Md m : nodes) r.push_back(m.id);
        w.record(10, r);
    }
    w.exit();
    // The metadata kinds an LLVM 3.7 module declares, as DXC writes them.
    w.enter(METADATA_BLOCK, 3);
    {
        static const char* const kinds[] = {"dbg",
                                            "tbaa",
                                            "prof",
                                            "fpmath",
                                            "range",
                                            "tbaa.struct",
                                            "invariant.load",
                                            "alias.scope",
                                            "noalias",
                                            "nontemporal",
                                            "llvm.mem.parallel_loop_access",
                                            "nonnull",
                                            "dereferenceable",
                                            "dereferenceable_or_null"};
        std::uint64_t id = 0;
        for (const char* kind : kinds) {
            std::vector<std::uint64_t> r = {id++};
            for (unsigned char c : std::string_view(kind)) r.push_back(c);
            w.record(6, r);
        }
    }
    w.exit();

    w.enter(VALUE_SYMTAB_BLOCK, 4);
    {
        std::uint64_t i = 0;
        auto sym = [&](const std::string& name) {
            std::vector<std::uint64_t> r = {i++};
            for (unsigned char c : name) r.push_back(c);
            w.record(1, r);
        };
        for (const Global& g : globals) sym(g.name);
        for (const Func& f : funcs) sym(f.name);
    }
    w.exit();

    w.enter(FUNCTION_BLOCK, 4);
    w.record(1, {std::uint64_t(layout.size())});
    std::uint32_t inst_id = base;
    // The operands of variable-length records, reused across them.
    std::vector<std::uint64_t> r;
    auto one = [&](std::uint32_t i) {
        const Inst& inst = insts[i];
        auto rel = [&](V v) -> std::uint64_t {
            std::uint32_t a = abs(v);
            if (!(a < inst_id)) bug("forward reference outside a phi");
            return std::uint64_t(inst_id - a);
        };
        const Op& op = inst.op;
        switch (op.kind) {
        case Op::Bin:
            w.record(2, {rel(op.a), rel(op.b), std::uint64_t(op.code)});
            break;
        case Op::Cast:
            w.record(3, {rel(op.a), std::uint64_t(inst.ty.id), std::uint64_t(op.code)});
            break;
        case Op::Cmp:
            w.record(28, {rel(op.a), rel(op.b), std::uint64_t(op.code)});
            break;
        case Op::Select:
            w.record(29, {rel(op.b), rel(op.c), rel(op.a)});
            break;
        case Op::Extract:
            w.record(26, {rel(op.a), std::uint64_t(op.index)});
            break;
        case Op::Call:
            r.clear();
            r.push_back(0);
            r.push_back(1 << 15);
            r.push_back(funcs[op.index].ty.id);
            r.push_back(rel(V::func(op.index)));
            for (V a : op.list) r.push_back(rel(a));
            w.record(34, r);
            break;
        case Op::Phi:
            r.clear();
            r.push_back(inst.ty.id);
            for (const auto& [v, b] : op.incoming) {
                r.push_back(signed_vbr(std::int64_t(inst_id) - std::int64_t(abs(v))));
                r.push_back(from(b));
            }
            w.record(16, r);
            break;
        case Op::Gep:
            r.clear();
            r.push_back(0);
            r.push_back(op.ty.id);
            r.push_back(rel(op.a));
            for (V x : op.list) r.push_back(rel(x));
            w.record(43, r);
            break;
        // Alignment 4 (log2 + 1), not volatile.
        case Op::Load:
            w.record(20, {rel(op.a), std::uint64_t(inst.ty.id), 3, 0});
            break;
        case Op::Store:
            w.record(44, {rel(op.a), rel(op.b), 3, 0});
            break;
        // Sequentially consistent, across threads.
        case Op::AtomicRmw:
            w.record(38, {rel(op.a), rel(op.b), std::uint64_t(op.code), 0, 6, 1});
            break;
        case Op::CmpXchg:
            w.record(46, {rel(op.a), rel(op.b), rel(op.c), 0, 6, 1, 6, 0});
            break;
        }
        if (!is_void(inst.ty)) inst_id += 1;
    };
    for (std::size_t lb : layout) {
        const Block& b = blocks[lb];
        for (std::uint32_t i : b.phis)
            if (lv[i]) one(i);
        for (std::uint32_t i : b.body) one(i);
        if (!b.term) bug("unterminated block");
        const Term& t = *b.term;
        switch (t.kind) {
        case Term::Ret:
            w.record(10, {});
            break;
        case Term::Br:
            w.record(11, {target(t.t)});
            break;
        case Term::CondBr: {
            std::uint32_t a = abs(t.cond);
            if (!(a < inst_id)) bug("branch condition after its branch");
            w.record(11, {target(t.t), target(t.f), std::uint64_t(inst_id - a)});
            break;
        }
        }
    }
    w.exit();
    w.exit();
    return w.finish();
}

std::uint32_t Writer::width() const { return stack.empty() ? 2 : stack.back().first; }

void Writer::emit(std::uint64_t v, std::uint32_t w) {
    cur |= v << bits;
    bits += w;
    while (bits >= 32) {
        words.push_back(std::uint32_t(cur));
        cur >>= 32;
        bits -= 32;
    }
}

void Writer::vbr(std::uint64_t v, std::uint32_t w) {
    const std::uint64_t hi = std::uint64_t(1) << (w - 1);
    while (v >= hi) {
        emit((v & (hi - 1)) | hi, w);
        v >>= w - 1;
    }
    emit(v, w);
}

void Writer::align() {
    if (bits > 0) {
        words.push_back(std::uint32_t(cur));
        cur = 0;
        bits = 0;
    }
}

void Writer::bytes(const std::uint8_t* b, std::size_t n) {
    for (std::size_t k = 0; k < n; ++k) emit(b[k], 8);
}

void Writer::enter(std::uint64_t id, std::uint32_t w) {
    emit(1, width());  // ENTER_SUBBLOCK
    vbr(id, 8);
    vbr(w, 4);
    align();
    words.push_back(0);
    stack.push_back({w, words.size()});
}

void Writer::exit() {
    emit(0, width());  // END_BLOCK
    align();
    if (stack.empty()) bug("open block");
    std::size_t start = stack.back().second;
    stack.pop_back();
    words[start - 1] = std::uint32_t(words.size() - start);
}

void Writer::record(std::uint64_t code, const std::uint64_t* ops, std::size_t n) {
    emit(3, width());
    vbr(code, 6);
    vbr(n, 6);
    for (std::size_t k = 0; k < n; ++k) vbr(ops[k], 6);
}

std::vector<std::uint8_t> Writer::finish() {
    align();
    std::vector<std::uint8_t> out;
    out.reserve(words.size() * 4);
    for (std::uint32_t x : words) {
        out.push_back(std::uint8_t(x));
        out.push_back(std::uint8_t(x >> 8));
        out.push_back(std::uint8_t(x >> 16));
        out.push_back(std::uint8_t(x >> 24));
    }
    return out;
}

}  // namespace x2w::vs::bitcode
