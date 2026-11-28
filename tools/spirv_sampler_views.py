"""Separate the depth-reference uses of a bindless sampler table for HLSL.

SPIRV-Cross gives a sampler table one HLSL type, so a table that serves both ordinary and
depth-reference samples needs a second view. This pass finds every uniform-constant variable whose
type is an array (fixed or runtime) of samplers, whatever its names and decorations, and adds a
copy of it at descriptor set 3 and the same binding for the depth-reference uses.
"""

CHAINS = ("OpAccessChain", "OpInBoundsAccessChain")
BRIDGE = "%bridge_cmp"
BRIDGE_SET = "3"


def _require_assembly_str(assembly):
    """Reject non-string SPIR-V assembly inputs."""
    if not isinstance(assembly, str):
        raise ValueError(f"assembly must be a str, got {type(assembly).__name__}")


def strip_comments(assembly):
    """Split into lines and drop `;` comments; a quoted string may span lines and may hold a `;`."""
    _require_assembly_str(assembly)
    lines, quoted, escaped = [], False, False
    for line in assembly.splitlines():
        end = len(line)
        for position, char in enumerate(line):
            if quoted and escaped:
                escaped = False
            elif quoted and char == "\\":
                escaped = True
            elif char == '"':
                quoted = not quoted
            elif char == ";" and not quoted:
                end = position
                break
        lines.append(line[:end].rstrip() if end < len(line) else line)
    return lines


def read_module(lines):
    """Collect result definitions, sample instructions and decorations from the assembly lines.

    Besides the samples and gathers, an `OpImageQueryLod` is an ordinary use of its sampled image: HLSL
    `CalculateLevelOfDetail` takes a `SamplerState`, never a comparison sampler.
    """
    module = {"defs": {}, "comparisons": [], "ordinary": [], "decorated": set(), "indirect": False,
              "sets": {}, "bindings": {}}
    for index, line in enumerate(lines):
        words = line.split()
        if words[:1] in (["OpGroupDecorate"], ["OpDecorateId"]):
            module["indirect"] = True
        if words[:1] == ["OpDecorate"] and len(words) >= 3:
            if words[2] == "DescriptorSet":
                module["sets"].setdefault(words[1], []).append(words[3:])
            elif words[2] == "Binding":
                module["bindings"].setdefault(words[1], []).append(words[3:])
            else:
                module["decorated"].add(words[1])
        if len(words) < 3 or words[1] != "=":
            continue
        module["defs"][words[0]] = words[2:]
        if words[2].startswith("OpImage") and "Dref" in words[2]:
            module["comparisons"].append((index, words))
        elif words[2] == "OpImageQueryLod" or words[2].startswith(
                ("OpImageSample", "OpImageGather", "OpImageSparseSample", "OpImageSparseGather")):
            module["ordinary"].append(words)
    return module


def pointee(defs, variable):
    """The type a uniform-constant variable points to, or None."""
    declaration = defs.get(variable, [])
    if len(declaration) < 3 or declaration[0] != "OpVariable" or declaration[2] != "UniformConstant":
        return None
    pointer = defs.get(declaration[1], [])
    if len(pointer) != 3 or pointer[:2] != ["OpTypePointer", "UniformConstant"]:
        return None
    return pointer[2]


def sampler_variables(defs):
    """Return the sampler tables and the plain sampler variables, in declaration order."""
    tables, plains = [], set()
    for name in defs:
        target = pointee(defs, name)
        if target is None:
            continue
        shape = defs.get(target, [])
        if shape == ["OpTypeSampler"]:
            plains.add(name)
        elif shape[:1] in (["OpTypeRuntimeArray"], ["OpTypeArray"]) and defs.get(shape[1]) == ["OpTypeSampler"]:
            tables.append(name)
    return tables, plains


def sampler_source(defs, tables, plains, sampled):
    """Follow a sampled image to its sampler: ("table", chain, load, image) or (kind, ...) otherwise."""
    image = defs.get(sampled, [])
    if len(image) != 4 or image[0] != "OpSampledImage":
        return ("image",)
    load = defs.get(image[3], [])
    if len(load) != 3 or load[0] != "OpLoad" or defs.get(load[1]) != ["OpTypeSampler"]:
        return ("load",)
    if load[2] in plains:
        return ("plain",)
    chain = defs.get(load[2], [])
    if (len(chain) == 4 and chain[0] in CHAINS and chain[2] in tables
            and defs.get(chain[1]) == ["OpTypePointer", "UniformConstant", load[1]]):
        return ("table", chain, load, image, image[3], load[2])
    return ("provenance",)


FAILURES = {"image": "unsupported depth-reference sampled image",
            "load": "unsupported depth-reference sampler load",
            "provenance": "unsupported depth-reference sampler provenance"}


def tables_to_split(module, tables, plains):
    """Tables used by a depth-reference sample and by an ordinary one; also the resolved comparisons."""
    defs = module["defs"]
    ordinary, unresolved = set(), False
    for words in module["ordinary"]:
        source = sampler_source(defs, tables, plains, words[4])
        if source[0] == "table":
            ordinary.add(source[1][2])
        elif source[0] != "plain":
            unresolved = True  # an ordinary sample with an unknown sampler may use any table
    resolved = []
    for index, words in module["comparisons"]:
        source = sampler_source(defs, tables, plains, words[4])
        if source[0] == "table":
            resolved.append((index, words, source))
        elif source[0] != "plain":
            raise ValueError(FAILURES[source[0]])
    return resolved, ordinary, unresolved


def _parse_uint_literal(token):
    """Parse a non-negative ASCII decimal or 0x-prefixed hexadecimal SPIR-V literal, or None.

    A multi-digit decimal token starting with ``0`` is refused with ValueError:
    spirv-as reads such tokens as octal (``010`` is 8, ``08`` is invalid), so
    parsing them as decimal would silently misread the register range.
    """
    if not isinstance(token, str) or not token.isascii():
        return None
    if token.isdigit():
        if len(token) > 1 and token[0] == "0":
            raise ValueError(
                f"leading-zero decimal literal {token!r} is ambiguous: spirv-as reads it as octal; "
                "use a plain decimal or 0x-prefixed hex literal"
            )
        return int(token, 10)
    if len(token) > 2 and token[:2] in ("0x", "0X") and all(ch in "0123456789abcdefABCDEF" for ch in token[2:]):
        return int(token, 16)
    return None


def _parse_set_literal(token):
    """Parse a DescriptorSet literal the way spirv-as does (base-0, optional ``+``), or None.

    Only used to detect a collision with descriptor set 3: ``03`` is octal 3
    and ``+3`` is 3, so both must count as set 3 even though _parse_uint_literal
    refuses the former and does not accept the latter.
    """
    if not isinstance(token, str):
        return None
    if not token.isascii():
        return None
    text = token[1:] if token.startswith("+") else token
    if text.isdigit():
        if len(text) > 1 and text[0] == "0":
            return int(text, 8) if all(ch in "01234567" for ch in text) else None
        return int(text, 10)
    if len(text) > 2 and text[:2] in ("0x", "0X") and all(ch in "0123456789abcdefABCDEF" for ch in text[2:]):
        return int(text, 16)
    return None


def table_slot(module, table):
    """The (set, binding) of a table; it must carry exactly one of each and no other decoration."""
    sets, bindings = module["sets"].get(table, []), module["bindings"].get(table, [])
    if len(sets) != 1 or len(bindings) != 1 or len(sets[0]) != 1 or len(bindings[0]) != 1:
        raise ValueError("sampler table requires one DescriptorSet and one Binding decoration")
    if _parse_uint_literal(sets[0][0]) is None or _parse_uint_literal(bindings[0][0]) is None:
        raise ValueError("sampler table requires literal DescriptorSet and Binding numbers")
    if table in module["decorated"] or len(module["defs"].get(table, [])) != 3:
        raise ValueError("unsupported sampler table declaration")
    return sets[0][0], bindings[0][0]


def register_end(defs, table, binding):
    """The first register after a table's views; a runtime or unresolvable size has no end."""
    decl = defs.get(table, [])
    pointer = defs.get(decl[1], []) if len(decl) >= 2 else []
    shape = defs.get(pointer[2], []) if len(pointer) >= 3 else []
    length = defs.get(shape[2], []) if len(shape) >= 3 and shape[0] == "OpTypeArray" else []
    binding_num = _parse_uint_literal(binding)
    length_num = _parse_uint_literal(length[2]) if len(length) == 3 and length[0] == "OpConstant" else None
    if binding_num is not None and length_num is not None and length_num > 0:
        return binding_num + length_num
    return None


def bridge_names(module, split):
    """Name one bridge variable per split table; the first keeps the plain prefix."""
    names, ranges = {}, []
    for number, table in enumerate(split):
        _, binding = table_slot(module, table)
        binding_num = _parse_uint_literal(binding)
        end = register_end(module["defs"], table, binding)
        for other, start, stop in ranges:
            if (stop is None or binding_num < stop) and (end is None or start < end):
                raise ValueError(f"sampler tables {other} and {table} overlap in the set 3 register range")
        ranges.append((table, binding_num, end))
        names[table] = BRIDGE if number == 0 else f"{BRIDGE}_table_{number}"
    return names


def split_sampler_views(assembly):
    _require_assembly_str(assembly)
    if "Dref" not in assembly:
        return assembly
    lines = strip_comments(assembly)
    module = read_module(lines)
    defs = module["defs"]
    tables, plains = sampler_variables(defs)
    if not tables or not module["comparisons"] or not module["ordinary"]:
        return assembly
    resolved, ordinary, unresolved = tables_to_split(module, tables, plains)
    compared = {source[1][2] for _, _, source in resolved}
    split = [table for table in tables if table in compared and (unresolved or table in ordinary)]
    if not split:
        return assembly
    if module["indirect"]:
        raise ValueError("unsupported indirect descriptor decorations")
    set_three = any(
        values and _parse_set_literal(values[0]) == int(BRIDGE_SET)
        for assigned in module["sets"].values()
        for values in assigned
    )
    if BRIDGE in assembly or set_three:
        raise ValueError("comparison sampler view requires free descriptor set 3 and bridge_cmp IDs")
    names = bridge_names(module, split)
    replacements = {}
    for index, words, (_, chain, load, image, sampled_load, pointer) in resolved:
        if chain[2] not in names:
            continue
        if module["decorated"].intersection((chain[2], words[4], sampled_load, pointer)):
            raise ValueError("unsupported decorated sampler path")
        name = f"{BRIDGE}_{index}"
        words[4] = name + "_image"
        replacements[index] = [
            f"{name}_ptr = {chain[0]} {chain[1]} {names[chain[2]]} {chain[3]}",
            f"{name}_value = OpLoad {load[1]} {name}_ptr",
            f"{name}_image = OpSampledImage {image[1]} {image[2]} {name}_value",
            " ".join(words),
        ]
    out = []
    for i, line in enumerate(lines):
        words = line.split()
        if words[:1] == ["OpEntryPoint"]:
            line += "".join(" " + bridge for table, bridge in names.items() if table in words)
        for table, bridge in names.items():
            if words[:3] == ["OpDecorate", table, "Binding"]:
                out.extend([f"OpDecorate {bridge} DescriptorSet {BRIDGE_SET}",
                            f"OpDecorate {bridge} Binding {table_slot(module, table)[1]}"])
            if words[:2] == [table, "="]:
                out.append(f"{bridge} = " + " ".join(defs[table]))
        out.extend(replacements.get(i, [line]))
    return "\n".join(out) + "\n"
