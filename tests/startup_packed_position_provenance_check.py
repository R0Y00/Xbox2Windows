"""Pixel input masks from the captured stream description of the exact pipeline (#1629).

  (no arguments)                       association controls, pure Python
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       translate the requests below through translate_request, compile the
                                       pixel stage (DXC ps_6_0, and FXC ps_5_1 when given) and draw it with the
                                       #1634 gradient: every pixel must read back its own packed x | (y << 16)

Two source-built requests share one cache directory. Each has a stream description, req_<key>.psin:
  target  exports v4; enable 0x8002 (PERSP_CENTER, POS_FIXED_PT) and address 0x8003 (PERSP_SAMPLE allocated
          but not enabled). The address mask lays out v0..v1 PERSP_SAMPLE, v2..v3 PERSP_CENTER, v4 the packed
          position. Laying out by the enable mask would put the position in v2.
  decoy   exports v2; enable = address = 0x8002, so its position is v2.
Both draw correctly only when each translation takes the masks of its own description, so no fixed register
and no mask taken from the other description passes. The controls also refuse, by name, a description that
names another pipeline, a stale description whose stage blobs changed, a missing enable or address mask,
contradictory masks (an enabled input not allocated) and a malformed line.
"""
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, G, compile_hlsl, root_blob

TARGET = (0x8002, 0x8003)  # (enable, address): the packed position in v4
DECOY = (0x8002, 0x8002)   # the packed position in v2


def ps_words(reg):
    # exp mrt0 vN, vN, vN, vN done vm; s_endpgm (the #1634 fixture's pixel stage)
    return struct.pack('<III', 0xf800180f, reg * 0x01010101, 0xbf810000)


def put_request(cache, ps, description=None, key=None):
    """Write req_<key>.a/.b/.root for VS_WORDS and `ps`, and the stream description line when given."""
    real = G.pipeline_key(VS_WORDS, ps, root_blob())
    key = real if key is None else key
    name = cache / ('req_%016x' % key)
    for ext, blob in (('.b', ps), ('.root', root_blob()), ('.a', VS_WORDS)):
        name.with_suffix(ext).write_bytes(blob)
    if description is not None:
        name.with_suffix('.psin').write_bytes(description.encode('ascii'))
    return key


def describe(key, masks):
    return 'psin1 %016x ena=0x%04x addr=0x%04x\n' % (key, *masks)


def translated_ps(cache, key):
    """translate_request with the manifest-only path (X2W_FXC=''): the pixel HLSL, or the refusal."""
    previous = os.environ.get('X2W_FXC')
    os.environ['X2W_FXC'] = ''
    try:
        result = G.translate_request(str(cache), key)
    finally:
        if previous is None:
            del os.environ['X2W_FXC']
        else:
            os.environ['X2W_FXC'] = previous
    if result != 'ok':
        return None, result
    return (cache / ('%016x.ps.hlsl' % key)).read_text(), result


def packed_line(text):
    found = re.findall(r'\bv(\d+) = \(uint\(i\.pos\.x\) & 0xffffu\)', text)
    return [int(n) for n in found]


def build_pair(cache, target=TARGET, decoy=DECOY):
    target_ps, decoy_ps = ps_words(4), ps_words(2)
    t = G.pipeline_key(VS_WORDS, target_ps, root_blob())
    d = G.pipeline_key(VS_WORDS, decoy_ps, root_blob())
    put_request(cache, target_ps, describe(t, target))
    put_request(cache, decoy_ps, describe(d, decoy))
    return t, d


def controls():
    failures = []

    def check(label, ok):
        print(('ok   ' if ok else 'FAIL ') + label)
        if not ok:
            failures.append(label)

    def refused(label, cache, key, reason):
        text, result = translated_ps(cache, key)
        rows = G.request_refusals(str(cache), key)
        check(label, text is None and reason in result and any(reason in r for r, _ in rows))

    # The translator lays out by the address mask and loads by the enable mask.
    pipe = G.Pipeline(VS_WORDS, ps_words(4), root_blob(), ps_hw_inputs=TARGET)
    check('distinct masks: position after the allocated PERSP_SAMPLE and PERSP_CENTER pairs (v4)',
          pipe.packed_position_vgpr == 4 and pipe.persp_center_vgpr == 2)
    pipe = G.Pipeline(VS_WORDS, ps_words(4), root_blob(), ps_hw_inputs=(0x0002, 0x8002))
    check('position allocated but not enabled is not modelled', pipe.packed_position_vgpr is None)
    for masks in ((0x8003, 0x8002), (0x8100, 0x8000)):
        try:
            G.Pipeline(VS_WORDS, ps_words(4), root_blob(), ps_hw_inputs=masks)
        except G.Unsupported as err:
            check('contradictory masks %04x/%04x refused by name' % masks,
                  'contradictory pixel input enable/address masks' in str(err))
        else:
            check('contradictory masks %04x/%04x refused by name' % masks, False)

    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        t, d = build_pair(cache)
        text, _ = translated_ps(cache, t)
        check('target takes its own description: packed position in v4', text is not None and packed_line(text) == [4])
        text, _ = translated_ps(cache, d)
        check('decoy takes its own description: packed position in v2', text is not None and packed_line(text) == [2])
        check('refusal report sees no refusal for either request',
              G.request_refusals(str(cache), t) == [] and G.request_refusals(str(cache), d) == [])
        # An RTV variant reads its source pipeline's description.
        request = 'rtv1 %d 28' % t
        variant = G.fnv(request.encode('ascii'))
        (cache / ('req_%016x.rtv' % variant)).write_text(request, encoding='ascii')
        text, _ = translated_ps(cache, variant)
        check('RTV variant takes its source pipeline description: v4', text is not None and packed_line(text) == [4])

    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        t = put_request(cache, ps_words(4))
        text, _ = translated_ps(cache, t)
        check('no captured description: no hardware inputs, position not modelled (today)',
              text is not None and packed_line(text) == [])

    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp)
        t, d = build_pair(cache)
        # The decoy's description copied over the target's.
        (cache / ('req_%016x.psin' % t)).write_bytes((cache / ('req_%016x.psin' % d)).read_bytes())
        refused('decoy description under the target name refused', cache, t,
                'pixel input description names another pipeline')
        # A stale description: the line names the key, but the stored pixel stage changed after capture. The
        # translation refuses the request first by its source key (#2595); the refusal report by its description.
        (cache / ('req_%016x.psin' % t)).write_bytes(describe(t, TARGET).encode('ascii'))
        (cache / ('req_%016x.b' % t)).write_bytes(ps_words(2))
        text, result = translated_ps(cache, t)
        rows = G.request_refusals(str(cache), t)
        check('stale description (stage blob changed) refused',
              text is None and 'shader request source key mismatch' in result
              and any('pixel input description names another pipeline' in r for r, _ in rows))

    for label, line, reason in (
        ('missing address mask', 'psin1 {k} ena=0x8002\n', 'pixel input address mask missing from the stream description'),
        ('missing enable mask', 'psin1 {k} addr=0x8003\n', 'pixel input enable mask missing from the stream description'),
        ('no masks', 'psin1 {k}\n', 'pixel input enable mask missing from the stream description'),
        ('contradictory masks', 'psin1 {k} ena=0x8003 addr=0x8002\n', 'contradictory pixel input enable/address masks'),
        ('duplicate mask', 'psin1 {k} ena=0x8002 ena=0x8002\n', 'invalid pixel input description'),
        ('malformed line', 'psin1 {k} ena=8002 addr=8003\n', 'invalid pixel input description'),
        ('unknown version', 'psin2 {k} ena=0x8002 addr=0x8003\n', 'invalid pixel input description'),
    ):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            t = G.pipeline_key(VS_WORDS, ps_words(4), root_blob())
            put_request(cache, ps_words(4), line.format(k='%016x' % t))
            refused(label + ' refused by name', cache, t, reason)

    if failures:
        print('%d control(s) failed' % len(failures))
        return 1
    print('controls: masks from the exact pipeline description; decoy, stale, missing and contradictory refused')
    return 0


def draw(dxc, renderer, adapter, fxc=None):
    compilers = [('dxc', dxc, '6_0')]
    if fxc:
        compilers.append(('fxc', fxc, '5_1'))
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        cache = folder / 'cache'
        cache.mkdir()
        t, d = build_pair(cache)
        wrong = cache / 'wrong'
        wrong.mkdir()
        # Wrong register: the target's code with a description whose masks place the position in v2.
        w = put_request(wrong, ps_words(4), describe(G.pipeline_key(VS_WORDS, ps_words(4), root_blob()), DECOY))
        sources = {}
        for label, where, key in (('target', cache, t), ('decoy', cache, d), ('wrong register', wrong, w)):
            text, result = translated_ps(where, key)
            assert text is not None, (label, result)
            sources[label] = text
        for kind, compiler, profile in compilers:
            vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, 'vs', 'vs_' + profile, folder)
            for label, good in (('target', True), ('decoy', True), ('wrong register', False)):
                ps = compile_hlsl(compiler, kind, sources[label], 'ps', 'ps_' + profile, folder)
                run = subprocess.run([renderer, adapter, '--startup-packed-position', vs, ps],
                                     capture_output=True, text=True)
                if run.returncode == 77:
                    print(run.stdout + run.stderr)
                    return 77
                if good:
                    assert run.returncode == 0 and 'all 91 packed positions matched' in run.stdout, (label, run.stdout, run.stderr)
                else:
                    assert run.returncode != 0 and 'position mismatch (1,0): expected 00000001 actual 00000000' in run.stderr, (label, run.stderr)
                device = [line for line in run.stdout.splitlines() if line.startswith(('backend=', 'adapter=', 'debug_'))]
                print(f'{kind} {adapter} {label}: ' + ('91 pixels matched' if good else 'expected mismatch at (1,0)')
                      + ' [%s]' % '; '.join(device))
    return 0


def main(argv):
    if not argv:
        return controls()
    mode, *args = argv
    if mode == 'draw':
        status = controls()
        return status or draw(*args)
    raise SystemExit(__doc__)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
