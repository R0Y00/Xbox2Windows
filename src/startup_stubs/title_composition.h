#pragma once

#include <dcomp.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

namespace x2w_startup {

// Attach the GPU output to the existing client area without taking ownership of the window or its message loop.
struct TitleComposition {
    Microsoft::WRL::ComPtr<IDCompositionDevice> device;
    Microsoft::WRL::ComPtr<IDCompositionTarget> target;
    Microsoft::WRL::ComPtr<IDCompositionVisual> visual;
    HWND window = nullptr;
    SIZE size{};

    HRESULT show(HWND hwnd, IDXGISwapChain1* chain, UINT width, UINT height)
    {
        RECT client{};
        if (!GetClientRect(hwnd, &client)) return HRESULT_FROM_WIN32(ERROR_INVALID_WINDOW_HANDLE);
        if (client.right <= 0 || client.bottom <= 0) return S_FALSE;
        if (!device) {
            HRESULT hr = DCompositionCreateDevice(nullptr, IID_PPV_ARGS(device.GetAddressOf()));
            if (FAILED(hr)) return hr;
        }
        if (window != hwnd || !target) {
            Microsoft::WRL::ComPtr<IDCompositionTarget> next_target;
            Microsoft::WRL::ComPtr<IDCompositionVisual> next_visual;
            HRESULT hr = device->CreateTargetForHwnd(hwnd, TRUE, next_target.GetAddressOf());
            if (SUCCEEDED(hr)) hr = device->CreateVisual(next_visual.GetAddressOf());
            if (SUCCEEDED(hr)) hr = next_visual->SetContent(chain);
            if (SUCCEEDED(hr)) hr = next_target->SetRoot(next_visual.Get());
            if (FAILED(hr)) return hr;
            target = next_target;
            visual = next_visual;
            window = hwnd;
            size = {};
        }
        if (size.cx == client.right && size.cy == client.bottom) return S_OK;
        D2D_MATRIX_3X2_F transform{static_cast<float>(client.right) / width, 0, 0,
                                   static_cast<float>(client.bottom) / height, 0, 0};
        HRESULT hr = visual->SetTransform(transform);
        if (SUCCEEDED(hr)) hr = device->Commit();
        if (SUCCEEDED(hr)) size = {client.right, client.bottom};
        return hr;
    }
};
}
