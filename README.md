# bitsandbytes

The bitsandbytes is a lightweight wrapper around CUDA custom functions, in particular 8-bit optimizers, matrix multiplication (LLM.int8()), and quantization functions.

For Original `README`, please ref to https://github.com/TimDettmers/bitsandbytes

This repo is for building it on Windows only. Changed some codes to make it work. This readme is to provide a step-by-step guide to build it on Windows.
It is built on Visual Studio 2022 Community, and CUDA 12.1.

**Why** try to build it on Windows? Because my RTX4070 laptop version only has 8G memory. Linux cannot load the large model without enough GPU memory. But Windows has the "Shared GPU Memory" concept to share RAM to GPU, which can make our job continue.

## Comparing to Linux version, what files are changed?
- csrc/ops.cuh: remove the `unistd.h`
- csrc/kernals.cu:
  - fixed the non-strict parameters for `kspmm_coo_very_sparse_naive` 
  - add templates instances for those like below  
    `MAKE_PreconditionOptimizer32bit1State`  
    They should all have 
    - ADAM (0)
    - MOMENTUM (1)
    - RMSPROP (2)
    - LION (4)
    - ADAGRAD (5)
- include/SIMD.h: define the missed `InstrFloatTraits<Scalar, float>`
- Some python files revised by https://github.com/fa0311/bitsandbytes-windows. Thanks for that!

## Prerequisites
- Install CUDA Toolkit
- Install Visual Studio build tools like VS 2022 Pro / Community

## Steps
  Open `x64 Native Tools Command Prompt for VS 2022` and start your journey  
  ```
  set PRJ_HOME=C:\${your-path}\bitsandbytes-windows
  ```
- Prepare the `pthread-win32`
  ```
  mkdir dependencies
  cd dependencies
  wget https://github.com/GerHobbelt/pthread-win32/archive/refs/tags/version-3.1.0-release.zip
  unzip pthread-win32-version-3.1.0-release.zip
  rm pthread-win32-version-3.1.0-release.zip
  mv pthread-win32-version-3.1.0-release pthread-win32
  cd pthread-win32
  nmake VC-static
  ```
- set some other env variables
  ```
  set PYTHON_HOME=C:/miniconda3
  set CUDA_HOME=c:/cuda
  set CUDA_VERSION=121
  set LIB=%LIB%;%CUDA_HOME%/lib/x64;%PYTHON_HOME%/lib;%PRJ_HOME%/dependencies/pthread-win32
  ```
- Try to build
  ```
  cd %PRJ_HOME%
  mkdir build
  nvcc -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -Xcompiler /DYNAMICBASE --use_fast_math -Xptxas=-v -dc %PRJ_HOME%/csrc/ops.cu %PRJ_HOME%/csrc/kernels.cu -I %CUDA_HOME%/include -I %PRJ_HOME%/csrc -I %PYTHON_HOME%/include -I %PRJ_HOME%/include -I %PRJ_HOME%/dependencies/pthread-win32 -L %CUDA_HOME%/lib/x64 -lcudart -lcublas -lcublasLt -lcusparse -lpthreadVC3 -L %PYTHON_HOME%/lib -L %PRJ_HOME%/dependencies/pthread-win32 --output-directory %PRJ_HOME%/build
  nvcc -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -Xcompiler /DYNAMICBASE -dlink %PRJ_HOME%/build/ops.obj %PRJ_HOME%/build/kernels.obj -o %PRJ_HOME%/build/link.obj
  cl /EHsc /std:c++14 /LD /I %CUDA_HOME%/include /I %PRJ_HOME%/csrc /I %PYTHON_HOME%/include /I %PRJ_HOME%/include /I %PRJ_HOME%/dependencies/pthread-win32 %PRJ_HOME%/build/kernels.obj %PRJ_HOME%/build/ops.obj %PRJ_HOME%/build/link.obj ./csrc/common.cpp ./csrc/cpu_ops.cpp ./csrc/pythonInterface.cpp libpthreadVC3.lib cudart.lib cublas.lib cublasLt.lib cusparse.lib /link /out:./bitsandbytes/libbitsandbytes_cuda%CUDA_VERSION%.dll
  ```
- Once build successfully, install it! If your CUDA version is not 12.1, please search below string `libbitsandbytes_cuda121.dll` and replace your cuda version number. (`main.py` it is)
  ```
  python setup.py install
  ```