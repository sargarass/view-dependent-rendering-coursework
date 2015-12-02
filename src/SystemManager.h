#pragma once
#include "Window.h"
#include "GpuStackAllocator.h"
#include "StackAllocator.h"
#include "ViewDependentRender.h"
class SystemManager {
public:
    SystemManager(){
        LibResouces::cudaInit();
        LibResouces::glfwInit();
        LibResouces::glfwInit();
    }

    static SystemManager *getInstance() {
        static SystemManager manager;
        return &manager;
    }

    ~SystemManager(){
        vdRender.deinit();
        gpuStackAllocator.deinit();
        gpuMemoryManager.freeAll();
        window.close();
        LibResouces::deinit();
    }

    VDRender vdRender;
    Window window;
    StackAllocator stackAllocator;
    GpuMemoryAllocator gpuMemoryManager;
    GpuStackAllocator gpuStackAllocator;

};
