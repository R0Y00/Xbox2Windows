"""Keep the launched executable's own window at its requested initial size and update its caption."""
import ctypes
from ctypes import wintypes as wt
import os
import re
import threading


def follow_window(pid, trap_log):
    user = ctypes.WinDLL('user32', use_last_error=True)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    user.SetThreadDpiAwarenessContext.argtypes = [wt.HANDLE]
    user.SetThreadDpiAwarenessContext(wt.HANDLE(-4))
    user.SendMessageTimeoutW.argtypes = (wt.HWND, wt.UINT, wt.WPARAM, wt.LPCWSTR, wt.UINT, wt.UINT,
                                          ctypes.POINTER(ctypes.c_size_t))
    user.GetWindow.argtypes = (wt.HWND, wt.UINT)
    user.GetWindow.restype = wt.HWND
    user.GetMenu.argtypes = [wt.HWND]
    user.GetMenu.restype = wt.HMENU
    kernel.OpenProcess.restype = wt.HANDLE
    process = kernel.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
    if not process:
        return
    window = None
    caption = ''
    position = 0
    remainder = b''
    previous = None
    status = 'Xbox2Windows'
    displayed = None

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def candidate(hwnd, unused):
        nonlocal window
        owner = wt.DWORD()
        user.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value != pid or not user.IsWindowVisible(hwnd) or user.GetWindow(hwnd, 4):
            return True
        name = ctypes.create_unicode_buffer(256)
        user.GetClassNameW(hwnd, name, len(name))
        if name.value in ('#32770', 'X2WOutputWindow') or user.GetWindowLongW(hwnd, -20) & 0x80:
            return True
        window = hwnd
        return False

    try:
        while kernel.WaitForSingleObject(wt.HANDLE(process), 300) == 258:
            if not window or not user.IsWindow(window):
                window = None
                user.EnumWindows(candidate, 0)
                if not window:
                    continue
                title = ctypes.create_unicode_buffer(512)
                user.GetWindowTextW(window, title, len(title))
                caption = title.value
                displayed = None
                rect = wt.RECT(0, 0, 1280, 720)
                if user.IsZoomed(window):
                    user.ShowWindowAsync(window, 9)
                if user.AdjustWindowRectExForDpi(ctypes.byref(rect), user.GetWindowLongW(window, -16),
                                                bool(user.GetMenu(window)), user.GetWindowLongW(window, -20),
                                                user.GetDpiForWindow(window)):
                    # Size once, keep position, z-order and activation, let the owner process the request.
                    user.SetWindowPos(window, None, 0, 0, rect.right - rect.left, rect.bottom - rect.top, 0x4016)
            try:
                with open(trap_log, 'rb') as log:
                    size = os.fstat(log.fileno()).st_size
                    if position > size or size - position > 2**21:
                        position = max(0, size - 2**21)
                        remainder = b''
                    log.seek(position)
                    chunk = log.read(2**21)
                    position = log.tell()
            except OSError:
                chunk = b''
            if chunk:
                lines = (remainder + chunk).split(b'\n')
                remainder = lines.pop()[-4096:]
                for line in lines:
                    match = re.search(rb'heartbeat: (\d+) frames presented in ([\d.]+) s', line)
                    if match:
                        frame, seconds = int(match[1]), float(match[2])
                        if previous and seconds > previous[1]:
                            fps = (frame - previous[0]) / (seconds - previous[1])
                            status = f'Xbox2Windows | Frame {frame} | {fps:.1f} FPS'
                        previous = frame, seconds
            if displayed != status:
                text = f'{caption} | {status}' if caption else status
                if user.SendMessageTimeoutW(window, 12, 0, text, 2, 150, ctypes.byref(ctypes.c_size_t())):
                    displayed = status
    finally:
        kernel.CloseHandle(wt.HANDLE(process))


def start(pid, trap_log):
    threading.Thread(target=follow_window, args=(pid, trap_log), name='title-window-status', daemon=True).start()
