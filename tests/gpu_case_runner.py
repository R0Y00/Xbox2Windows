"""Shared by the GPU checks: one D3D12 device per test instead of one offscreen_check process per case (issue #448).

run(renderer, adapter, args)
    The same as subprocess.run([renderer, adapter, *args], capture_output=True, text=True), but the case goes to a long-lived
    `offscreen_check <adapter> --serve` process of this test (one device for all its cases). The returned
    subprocess.CompletedProcess has the one-shot meaning: returncode 0 pass, 1 failure, 77 skip, 2 bad arguments. Its stdout
    and stderr are the text the one-shot call would print, adapter banner included for EVERY case (so a check that looks for
    "debug_layer=enabled" and "debug_errors=0 debug_warnings=0" sees them in each case, as it does in a one-shot call).

    Only a server that cannot start falls back to the one-shot call (an older offscreen_check without `--serve`, no adapter,
    X2W_FLAT_SERVER=0): no case has been sent yet, so the one-shot call gives the verdict. Once a case has been sent, a
    renderer that dies, hangs past CASE_TIMEOUT_S, or answers with the wrong case number gives a FAILED verdict (returncode 1,
    the reason on stderr) for that case; it is never run again, so a crash cannot turn into a pass. The next case starts a
    new server.
"""
import atexit
import os
import queue
import subprocess
import threading
import time

CASE_TIMEOUT_S = 300
SERVER_OFF = os.environ.get('X2W_FLAT_SERVER') == '0'
DEVICE_REMOVED = 'device removed during case'
_servers = {}


class _Lost(Exception):
    """The server did not answer the case it was sent."""


class _Server:
    def __init__(self, renderer, adapter):
        self.out, self.err = queue.Queue(), queue.Queue()
        self.proc = subprocess.Popen([str(renderer), adapter, '--serve'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, bufsize=1)
        self.banner, self.banner_err, self.ready, self.returncode, self.stderr_text = '', '', False, None, ''
        self.next_case = 0
        self.threads = [threading.Thread(target=self._pump, args=(self.proc.stdout, self.out, 'X2WCASE '), daemon=True),
                        threading.Thread(target=self._pump, args=(self.proc.stderr, self.err, 'X2WEND '), daemon=True)]
        for thread in self.threads:
            thread.start()
        deadline = time.monotonic() + CASE_TIMEOUT_S
        first = self._get(self.out, deadline)        # the banner, closed by the marker pair of case -1
        banner_err = self._get(self.err, deadline)
        if first is None or first[1] != -1 or first[2] != 0 or banner_err is None or banner_err[1] != -1:
            self.proc.kill()                         # closed first (skip 77, an unknown mode, a crash) or a hang
            self.proc.wait()
            self.returncode = self.proc.returncode
            self.stderr_text = ''.join(self._drain(self.err))
            self.banner = first[0] if first else ''
            self._release()
            return
        self.banner, self.banner_err, self.ready = first[0], banner_err[0], True

    @staticmethod
    def _pump(stream, sink, marker):
        """Split a stream into per-case chunks at its marker line "<marker><case index> [exit code]"."""
        chunk = []
        for line in stream:
            parts = line.split()
            if line.startswith(marker) and len(parts) == (3 if marker == 'X2WCASE ' else 2):
                try:
                    index = int(parts[1])
                    code = int(parts[2]) if marker == 'X2WCASE ' else 0
                except ValueError:
                    chunk.append(line)
                    continue
                sink.put((''.join(chunk), index, code))
                chunk = []
            else:
                chunk.append(line)
        if chunk:
            sink.put((''.join(chunk), None, None))
        sink.put(None)

    @staticmethod
    def _get(q, deadline):
        """The next (text, case index, exit code) of a stream, or None at its end or the deadline."""
        try:
            item = q.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty:
            return None
        return item                                 # None at the end of the stream; an unmarked tail has index None

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
        """Send one case; the reply must carry exactly this case's number on both streams, else _Lost."""
        index, self.next_case = self.next_case, self.next_case + 1
        deadline = time.monotonic() + CASE_TIMEOUT_S
        try:
            self.proc.stdin.write('\t'.join(args) + '\n')
            self.proc.stdin.flush()
        except OSError as error:
            raise _Lost('could not send the case: %s' % error)
        out, err = self._get(self.out, deadline), self._get(self.err, deadline)
        if out is None or err is None or out[1] is None or err[1] is None:
            state = 'timed out after %d s' % CASE_TIMEOUT_S if self.proc.poll() is None else 'exited (code %s)' % self.proc.poll()
            raise _Lost('the renderer %s during the case' % state)
        if out[1] != index or err[1] != index:
            raise _Lost('the renderer answered case %s/%s to case %d' % (out[1], err[1], index))
        return subprocess.CompletedProcess(self.proc.args, out[2], self.banner + out[0], self.banner_err + err[0])

    def _release(self):
        """Wait for the process and close every pipe, so nothing is left open after any exit."""
        self.proc.wait()
        for thread in self.threads:                 # the pumps end at the end of the pipes
            thread.join(timeout=5)
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            try:
                stream.close()
            except OSError:
                pass

    def kill(self):
        self.proc.kill()
        self._release()

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()
        self._release()


def _close_all():
    for server in _servers.values():
        if server:
            server.close()


atexit.register(_close_all)


def _failed(server, reason):
    """The FAILED verdict of a case whose server did not answer it."""
    text = ''.join(server._drain(server.err)) if server.proc.poll() is not None else ''
    server.kill()
    return subprocess.CompletedProcess(server.proc.args, 1, '', 'gpu_case_runner: %s\n%s' % (reason, text))


def run(renderer, adapter, args):
    args = [str(a) for a in args]
    key = (str(renderer), adapter)
    if not SERVER_OFF and '\t' not in ''.join(args) and '\n' not in ''.join(args):
        for _ in range(2):                          # a second start only when the first server was already gone before the case
            server = _servers.get(key)
            if server is None:
                server = _servers[key] = _Server(renderer, adapter)
                if not server.ready:                # nothing was sent: the one-shot call gives the verdict
                    _servers[key] = False
                    if server.returncode == 77:     # no adapter: the skip the one-shot call reports
                        return subprocess.CompletedProcess(server.proc.args, 77, server.banner, server.stderr_text)
                    break
            elif server is False:
                break
            if server.proc.poll() is not None:      # it ended after its last answer (device removed): start a new one
                server.kill()
                _servers[key] = None
                continue
            try:
                result = server.case(args)
            except _Lost as lost:
                _servers[key] = None
                return _failed(server, str(lost))
            if DEVICE_REMOVED in result.stderr:
                server.close()
                _servers[key] = None
            return result
        else:
            _servers[key] = False
    return subprocess.run([str(renderer), adapter, *args], capture_output=True, text=True)
