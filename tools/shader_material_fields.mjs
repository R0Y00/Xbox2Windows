// node tools/shader_material_fields.mjs path/to/dxc.exe path/to/<shader-key>.dxil [...]
// Save stdout to a text file and set X2W_MATERIAL_FIELDS_PROFILE to its path before launch.
// Writes C++ rows for fields that feed resource handles of actual texture-sampling instructions.
import { readFileSync } from "node:fs";
import { basename } from "node:path";
import { execFileSync } from "node:child_process";

function sampleFields(ir) {
    const definitions = new Map();
    for (const m of ir.matchAll(/^  %(\d+) = ([^;\r\n]+)/gm)) definitions.set(m[1], m[2].trim());
    function match(id, pattern) { return pattern.exec(definitions.get(id) ?? ""); }
    function input(id, pattern) { return match(id, pattern)?.[1]; }
    function handle(id) {
        return input(id, /^call %dx.types.Handle @dx\.op\.annotateHandle\(i32 216, %dx.types.Handle %(\d+),/);
    }
    function heapIndex(id) {
        return input(handle(id), /^call %dx.types.Handle @dx\.op\.createHandleFromHeap\(i32 218, i32 %(\d+), i1 false, i1 false\)$/);
    }
    function unclamp(id) {
        return input(id, /^call i32 @dx\.op\.binary\.i32\(i32 40, i32 %(\d+), i32 2097151\)$/);
    }
    function unshift(id) { return input(id, /^lshr i32 %(\d+), 5$/); }
    function rootWord(id) {
        const field = match(id, /^extractvalue %dx.types.CBufRet.i32 %(\d+), (\d+)$/);
        if (!field) return;
        const load = match(field[1], /^call %dx.types.CBufRet.i32 @dx\.op\.cbufferLoadLegacy.i32\(i32 59, %dx.types.Handle %(\d+), i32 (\d+)\)$/);
        if (load && match(handle(load[1]), /^call %dx.types.Handle @dx\.op\.createHandleFromBinding\(i32 217, %dx.types.ResBind \{ i32 0, i32 0, i32 0, i8 2 \}, i32 0, i1 false\)$/))
            return 4 * Number(load[2]) + Number(field[2]);
    }
    function materialOffset(sampleHandle) {
        const shifted = unshift(unclamp(heapIndex(sampleHandle)));
        let word = input(shifted, /^shl i32 %(\d+), 5$/);
        word = input(word, /^bitcast float %(\d+) to i32$/) ?? word;
        const raw = input(word, /^extractvalue %dx.types.ResRet.(?:f32|i32) %(\d+), 0$/);
        const load = match(raw, /^call %dx.types.ResRet.(?:f32|i32) @dx\.op\.rawBufferLoad\.(?:f32|i32)\(i32 139, %dx.types.Handle %(\d+), i32 %(\d+), i32 undef, i8 1, i32 4\)$/);
        if (!load) return;
        const regularIndex = input(heapIndex(load[1]), /^or i32 %(\d+), 2097152$/);
        const tableBase = input(unshift(unclamp(regularIndex)), /^add i32 %(\d+), 2752$/);
        if (rootWord(tableBase) !== 8) return;
        const offset = match(load[2], /^add i32 %(\d+), (\d+)$/);
        if (rootWord(offset ? offset[1] : load[2]) !== 14) return;
        return offset ? Number(offset[2]) : 0;
    }
    let mask = 0;
    for (const expression of definitions.values()) {
        const sample = /@dx\.op\.sample\w*\.f32\(i32 \d+, %dx.types.Handle %(\d+),/.exec(expression);
        if (!sample) continue;
        const offset = materialOffset(sample[1]);
        if (offset >= 0 && offset < 64 && offset % 4 === 0) mask |= 1 << (offset / 4);
    }
    return mask;
}

const [dxc, ...files] = process.argv.slice(2);
if (!dxc || !files.length) throw Error("Expected a DXC executable and one or more keyed DXIL files");
for (const path of files.sort((a, b) => basename(a).localeCompare(basename(b)))) {
    const key = basename(path, ".dxil");
    const code = readFileSync(path);
    if (!/^[0-9a-f]{16}$/.test(key) || code.toString("ascii", 0, 4) !== "DXBC") throw Error(path);
    const ir = execFileSync(dxc, ["-dumpbin", path], { encoding: "utf8", maxBuffer: 32 << 20 });
    const mask = sampleFields(ir);
    if (!mask) continue;
    const hex = (v, width) => v.toString(16).padStart(width, "0");
    console.log([key, hex(code.readBigUInt64LE(4), 16), hex(code.readBigUInt64LE(12), 16), hex(mask, 4)].join(" "));
}
