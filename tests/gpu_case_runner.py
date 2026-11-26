"""Shared by the GPU checks: one D3D12 device per test and a compiler cache, instead of one process per case.

run(renderer, adapter, args)
    The same as subprocess.run([renderer, adapter, *args], capture_output=True, text=True), but the case goes to a long-lived
    `offscreen_check <adapter> --serve` process of this test (one device for all its cases). The returned
    subprocess.CompletedProcess has the one-shot meaning: returncode 0 pass, 1 failure, 77 skip, 2 bad arguments; stdout and
    stderr are the text the one-shot call would print (the adapter banner comes with the first case only). Anything that goes
    wrong with the server (an older offscreen_check without `--serve`, a process that dies or hangs, X2W_FLAT_SERVER=0) falls
    back to the one-shot call, so a check never depends on the server.

compile_cached(compiler, args_for, source, key_parts, output)
    Content-addressed cache of compiled shader objects in out/hlsl-cache (or $X2W_HLSL_CACHE; X2W_HLSL_CACHE=0 turns it off).
    The key is the source text, the compiler file (path, size, modification time) and the compile arguments, so a changed
    source or a different compiler never hits. Failed compiles are never stored. Entries are published with an atomic rename,
    so parallel ctest processes and the warp/hardware twins of a test can share it.
"""
import atexit
import hashlib
import os
import queue
import shutil
import subprocess
import threading
from pathlib import Path

CASE_TIMEOUT_S = 300
SERVER_OFF = os.environ.get('X2W_FLAT_SERVER') == '0'
_servers = {}


class _Server:
    def __init__(self, renderer, adapter):
        self.out, self.err = queue.Queue(), queue.Queue()
        self.proc = subprocess.Popen([str(renderer), adapter, '--serve'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, bufsize=1)
        self.banner, self.banner_err, self.ready, self.returncode, self.stderr_text = [], '', False, None, ''
        threading.Thread(target=self._pump, args=(self.proc.stdout, self.out, 'X2WCASE '), daemon=True).start()
        threading.Thread(target=self._pump, args=(self.proc.stderr, self.err, 'X2WEND '), daemon=True).start()
        first = self._get(self.out)                  # the banner, closed by the marker pair of case -1
        banner_err = self._get(self.err)
        if first is None or first[1] != 0 or banner_err is None:     # closed first (skip 77, an unknown mode, a crash)
            self.proc.kill()
            self.proc.wait()
            self.returncode = self.proc.returncode
            self.stderr_text = ''.join(self._drain(self.err))
            self.banner = [first[0]] if first else []
            return
        self.banner, self.banner_err, self.ready = [first[0]], banner_err[0], True

    @staticmethod
    def _pump(stream, sink, marker):
        """Split a stream into per-case chunks at its marker line; the stdout marker carries the exit code."""
        chunk = []
        for line in stream:
            if line.startswith(marker):
                code = int(line.split()[2]) if marker == 'X2WCASE ' else None
                sink.put((''.join(chunk), code if code is not None else 0, True))
                chunk = []
            else:
                chunk.append(line)
        if chunk:
            sink.put((''.join(chunk), None, False))
        sink.put(None)

    @staticmethod
    def _get(q):
        try:
            item = q.get(timeout=CASE_TIMEOUT_S)
        except queue.Empty:
            return None
        if item is None:
            return None
        text, code, marked = item
        return (text, code if marked else None)

    @staticmethod
    def _drain(q):
        items = []
        while True:
            try:
                item = q.get(timeout=2)
            except queue.Empty:
                return items
            if item is None:
                return items
            items.append(item[0])

    def case(self, args):
        try:
            self.proc.stdin.write('\t'.join(args) + '\n')
            self.proc.stdin.flush()
        except OSError:
            return None
        out, err = self._get(self.out), self._get(self.err)
        if out is None or err is None or out[1] is None:
            return None
        banner, self.banner = ''.join(self.banner), []
        banner_err, self.banner_err = self.banner_err, ''
        return subprocess.CompletedProcess(self.proc.args, out[1], banner + out[0], banner_err + err[0])

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()


def _close_all():
    for server in _servers.values():
        if server:
            server.close()


atexit.register(_close_all)


def run(renderer, adapter, args):
    args = [str(a) for a in args]
    key = (str(renderer), adapter)
    if not SERVER_OFF and '\t' not in ''.join(args) and '\n' not in ''.join(args):
        for _ in range(2):                          # one restart after a dead server
            server = _servers.get(key)
            if server is None:
                server = _servers[key] = _Server(renderer, adapter)
                if not server.ready:
                    _servers[key] = False
                    if server.returncode == 77:     # no adapter: the skip the one-shot call reports
                        return subprocess.CompletedProcess(server.proc.args, 77, ''.join(server.banner), server.stderr_text)
                    break
            elif server is False:
                break
            result = server.case(args)
            if result is not None:
                return result
            server.proc.kill()
            server.proc.wait()
            _servers[key] = None
        else:
            _servers[key] = False
    return subprocess.run([str(renderer), adapter, *args], capture_output=True, text=True)


def compile_cached(compiler, args_for, source, key_parts, output):
    """Compile `source` to `output` with args_for(output) unless an identical compile is cached; raises like subprocess.run(check=True)."""
    root = os.environ.get('X2W_HLSL_CACHE')
    cache = None if root == '0' else Path(root) if root else Path(__file__).resolve().parent.parent / 'out' / 'hlsl-cache'
    entry = None
    if cache is not None:
        try:
            stat = Path(compiler).stat()
            digest = hashlib.sha256('\0'.join([str(Path(compiler).resolve()), str(stat.st_size), str(stat.st_mtime_ns),
                                                *key_parts, source]).encode('utf-8')).hexdigest()
            entry = cache / digest[:2] / (digest + '.cso')
            if entry.is_file():
                shutil.copyfile(entry, output)
                return
        except OSError:
            entry = None
    subprocess.run(args_for(output), check=True, capture_output=True, text=True)
    if entry is not None:
        try:
            entry.parent.mkdir(parents=True, exist_ok=True)
            temporary = entry.with_name('%s.%d.tmp' % (entry.name, os.getpid()))
            shutil.copyfile(output, temporary)
            os.replace(temporary, entry)
        except OSError:
            pass
