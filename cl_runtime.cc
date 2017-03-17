/*
 * Copyright (c) 2011-2015 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * For use for simulation and test purposes only
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Authors: Marc Orr
 */


#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include <cassert>
#include <set>

#include "cl_runtime.hh"
#include "hsa_kernel_info.hh"
#include "misc.hh"
#include "qstruct.hh"

#define divCeil(a, b) ((a + b - 1) / b)

volatile uint32_t *dispatcherDoorbell = (uint32_t*)0x10000000;
HsaQueueEntry *hsaTaskPtr = (HsaQueueEntry*)0x10000008;

// Pointer to the portion of the flat address space reserved for LDS memory
static char *ldsSpaceStart = nullptr;

// global variables
platform *theOnlyPlatform = nullptr;
std::set<cl_mem> memTracker;

//object tracker
std::map <cl_context, cl_int> refcontext;
std::map <cl_mem, cl_int> refmem;
std::map <cl_kernel, cl_int> refkernel;
std::map <cl_command_queue, cl_int> refcmdqueue;
std::map <cl_program, cl_int> refprogram;

static HsaDriverSizes hsaDriverSizes;
static HsaKernelInfo *hsaKernelInfo;
static const char *hsaStringTable;
static const uint8_t *hsaCode;
static const uint8_t *hsaReadonly;
static uint32_t numCUs = 0;
static uint32_t VecSize = 0;

cl_uint _cl_device_id::nextID = 0;

// The current version of the compiler adds six implicit arguments to an
// OpenCL Kernel. This was 3 before, and now it's become 6. CLOC or other
// runtimes may not have them.
static const int DEFAULT_OCL_KERN_ARGS = 6;

void
clWarn(const char *s)
{
    DPRINT("%s\n", s);
}

void
clFatal(const char *s)
{
    printf("%s\n", s);
    exit(0);
}

static void
hsaDriverInit()
{
    // skip if already initialized
    if (hsaKernelInfo)
        return;

    DPRINT("hsaDriverInit()");

    int fd = open("/dev/hsa", O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Failed to open /dev/hsa\n");
        exit(1);
    }

    int status;

    status = ioctl(fd, HSA_GET_SIZES, &hsaDriverSizes);
    if (status) {
        fprintf(stderr, "HSA_GET_SIZES failed\n");
        exit(1);
    }

    DPRINT("hsaDriverInit(): found %d kernels\n", hsaDriverSizes.num_kernels);

    hsaKernelInfo = new HsaKernelInfo[hsaDriverSizes.num_kernels];
    hsaStringTable = new char[hsaDriverSizes.string_table_size];

    // The simulator fetches a cache block at a time. This behavior may result
    // in fetching beyond the code region. If this happens, the simulator
    // expects the values fetched to be nullptr. In reality, a more elegant
    // solution should be put into place eventually.
    uint32_t pad = hsaDriverSizes.code_size % 4096;
    pad = (pad == 0) ? 0 : 4096 - pad;
    uint32_t padded_code_size = hsaDriverSizes.code_size + pad;
    int fp = open("/dev/zero", O_RDONLY);
    hsaCode = (uint8_t*)mmap(nullptr, padded_code_size, PROT_EXEC,
                             MAP_PRIVATE, fp, 0);

    if (hsaDriverSizes.readonly_size > 0) {
        pad = hsaDriverSizes.readonly_size % 4096;
        pad = (pad == 0) ? 0 : 4096 - pad;
        uint32_t padded_readonly_size = hsaDriverSizes.readonly_size + pad;
        hsaReadonly = (uint8_t*)mmap(nullptr, padded_readonly_size,
                                     PROT_READ, MAP_PRIVATE, fp, 0);
        fprintf(stderr, "hsaReadonly = %llx\n", hsaReadonly);
    } else {
        hsaReadonly = nullptr;
    }

    DPRINT("\t%d %d %d\n\t%p %p %p\n", hsaDriverSizes.num_kernels,
           hsaDriverSizes.string_table_size, hsaDriverSizes.code_size,
           hsaKernelInfo, hsaStringTable, hsaCode);

    status = ioctl(fd, HSA_GET_KINFO, hsaKernelInfo);
    if (status) {
        fprintf(stderr, "HSA_GET_KINFO failed\n");
        exit(1);
    }
    DPRINT("HSA_GET_KINFO\n");

    status = ioctl(fd, HSA_GET_STRINGS, hsaStringTable);
    if (status) {
        fprintf(stderr, "HSA_GET_STRINGS failed\n");
        exit(1);
    }
    DPRINT("HSA_GET_STRINGS\n");

    status = ioctl(fd, HSA_GET_CODE, hsaCode);
    if (status) {
        fprintf(stderr, "HSA_GET_CODE failed\n");
        exit(1);
    }
    DPRINT("HSA_GET_CODE\n");

    if (hsaDriverSizes.readonly_size > 0) {
        status = ioctl(fd, HSA_GET_READONLY_DATA, hsaReadonly);
        if (status) {
            fprintf(stderr, "HSA_GET_READONLY_DATA failed\n");
            exit(1);
        }
        DPRINT("HSA_GET_READONLY_DATA\n");
    } else {
        hsaReadonly = nullptr;
    }

    status = ioctl(fd, HSA_GET_CU_CNT, (cl_uint*)(&numCUs));
    if (status) {
        fprintf(stderr, "HSA_GET_CU_CNT failed\n");
        exit(1);
    }

    status = ioctl(fd, HSA_GET_VSZ, (cl_uint*)(&VecSize));
    if (status) {
        fprintf(stderr, "HSA_GET_VSZ failed\n");
        exit(1);
    }
}

// opencl api implementation

/* Platform API */
CL_API_ENTRY cl_int CL_API_CALL
clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms,
                 cl_uint *num_platforms)
CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clGetPlatformIDs()\n");

    if (!theOnlyPlatform) {
        theOnlyPlatform = new platform();
    }

    if ((!num_entries && platforms) ||
       (!num_platforms && !platforms)) {
        return CL_INVALID_VALUE;
    } else if ((platforms) && (num_entries > 0)) {
       platforms[0] = theOnlyPlatform->getID();
    } else if (num_platforms) {
        *num_platforms = 1;
    }

    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name,
                  size_t param_value_size, void *param_value,
                  size_t *param_value_size_ret)
CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clGetPlatformInfo()\n");

    if (!platform || platform->ID != theOnlyPlatform->getID()->ID) {
        return CL_INVALID_PLATFORM;
    }

    if (param_value_size_ret) {
        *param_value_size_ret = 0;
    }

    char *ret = (char*)param_value;
    const char *src;

    switch(param_name) {
      case CL_PLATFORM_PROFILE:
        src = "FULL_PROFILE";
        if (param_value_size > strlen(src)) {
            strcpy(ret, src);
            if (param_value_size_ret) {
                *param_value_size_ret = sizeof(src);
            }
        }
        break;
      case CL_PLATFORM_VERSION:
        src = "OpenCL 1.0";
        if (param_value_size > strlen(src)) {
            strcpy(ret, src);
            if (param_value_size_ret) {
                *param_value_size_ret = sizeof(src);
            }
        }
        break;
      case CL_PLATFORM_NAME:
        src = "ATI Stream";
        if (param_value_size > strlen(src)) {
            strcpy(ret, src);
            if (param_value_size_ret) {
                *param_value_size_ret = sizeof(src);
            }
        }
        break;
      case CL_PLATFORM_VENDOR:
        src = "Advanced Micro Devices, Inc.";
        if (param_value_size > strlen(src)) {
            strcpy(ret, src);
            if (param_value_size_ret) {
                *param_value_size_ret = sizeof(src);
            }
        }
        break;
      case CL_PLATFORM_EXTENSIONS:
        src = "";
        if (param_value_size > strlen(src)) {
            strcpy(ret, src);
            if (param_value_size_ret) {
                *param_value_size_ret = sizeof(src);
            }
        }
        break;
      default:
        return CL_INVALID_VALUE;
    }

    return CL_SUCCESS;
}

CL_API_ENTRY cl_context CL_API_CALL
clCreateContextFromType(const cl_context_properties *properties,
                        cl_device_type device_type,
                        void (CL_CALLBACK *pfn_notify)(const char *,
                                                       const void *,
                                                       size_t, void *),
                        void *user_data, cl_int *errcode_ret)
CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clCreateContextFromType()\n");

    cl_int ret = CL_SUCCESS;

    if (!properties) {
        ret = CL_INVALID_PLATFORM;
    } else if (!pfn_notify && user_data) {
        ret = CL_INVALID_VALUE;
    } else if (pfn_notify) {
        clFatal("clCreateContextFromType: pfn_notify not implemented\n");
    }

    int prop_idx = 0;
    while (properties[prop_idx]) {
        _cl_platform_id *prop;
        switch (properties[prop_idx]) {
          case CL_CONTEXT_PLATFORM:
            prop = (_cl_platform_id*)(properties[prop_idx + 1]);
            if (prop->ID != theOnlyPlatform->getID()->ID) {
                ret = CL_INVALID_PLATFORM;
            }
            break;
          default:
            ret = CL_INVALID_VALUE;
        }
        prop_idx += 2;
    }

    if (ret != CL_SUCCESS) {
        if (errcode_ret) {
            *errcode_ret = ret;
        }

        return nullptr;
    }

    _cl_context *context;
    ret = theOnlyPlatform->addContext(device_type, &context);
    if (errcode_ret) {
        *errcode_ret = ret;
    }

    if (ret == CL_SUCCESS) {
        return context;
    } else {
        return nullptr;
    }
}

CL_API_ENTRY cl_context CL_API_CALL
clCreateContext(const cl_context_properties *properties, cl_uint num_devices,
                const cl_device_id *devices,
                void (CL_CALLBACK *pfn_notify)(const char*, const void*,
                                               size_t, void*),
                void *user_data, cl_int *errcode_ret)
CL_API_SUFFIX__VERSION_1_0
{
    if (!properties) {
        _cl_context *context;
        theOnlyPlatform->addContext(CL_DEVICE_TYPE_GPU, &context);

        if (errcode_ret) {
            *errcode_ret = CL_SUCCESS;
        }

        return context;
    }

    return clCreateContextFromType(properties, CL_DEVICE_TYPE_GPU, pfn_notify,
                                   user_data,errcode_ret);
}
CL_API_ENTRY cl_int CL_API_CALL
clGetContextInfo(cl_context context, cl_context_info param_name,
                 size_t param_value_size, void *param_value,
                 size_t *param_value_size_ret)
CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clGetContextInfo()\n");

    if (!theOnlyPlatform->isContextValid(context)) {
        return CL_INVALID_CONTEXT;
    }

    switch (param_name) {
      case CL_CONTEXT_REFERENCE_COUNT:
        if (param_value_size_ret) {
            *param_value_size_ret = sizeof(cl_uint);
        }

        if (param_value) {
            if (param_value_size >= sizeof(cl_uint)) {
                *((cl_uint*)(param_value)) = context->getRefCount();
            } else {
                return CL_INVALID_VALUE;
            }
        }

        break;
      case CL_CONTEXT_DEVICES:
        if (param_value_size_ret) {
           *param_value_size_ret =
               sizeof(cl_device_id) * context->getNumDevices();
        }

        if (param_value) {
            if (param_value_size >=
                sizeof(cl_device_id) * context->getNumDevices()) {
                (*((cl_device_id *)(param_value))) = context->devList;
            } else {
                return CL_INVALID_VALUE;
            }
        }
        break;
      case CL_CONTEXT_PROPERTIES:
        clFatal("clGetContextInfo: CL_CONTEXT_PROPERTIES not "
                "yet implemented\n");
        break;
      default:
        return CL_INVALID_VALUE;
    }

    return CL_SUCCESS;
}

/* Command Queue APIs */
CL_API_ENTRY cl_command_queue CL_API_CALL
clCreateCommandQueue(cl_context context, cl_device_id device,
                     cl_command_queue_properties properties,
                     cl_int *errcode_ret)
CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clCreateCommandQueue()\n");

    if (!theOnlyPlatform->isContextValid(context)) {
        if (errcode_ret) {
            *errcode_ret = CL_INVALID_CONTEXT;
        }

        return nullptr;
    }

    if (device->type != context->getDevType()
        && !context->isValidDev(device)) {
        if (errcode_ret) {
            *errcode_ret = CL_INVALID_DEVICE;
        }

        return nullptr;
    }

    if (properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
        clWarn("clCreateCommandQueue: CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE "
               "not yet implemented\n");
    }
    if (properties & CL_QUEUE_PROFILING_ENABLE) {
        clWarn("clCreateCommandQueue: CL_QUEUE_PROFILING_ENABLE not yet "
               "implemented\n");
    }
    if (properties & ~(CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
        | CL_QUEUE_PROFILING_ENABLE)) {
        if (errcode_ret) {
            *errcode_ret = CL_INVALID_QUEUE_PROPERTIES;
        }

        return nullptr;
    }

    _cl_command_queue *CQ = device->addCQ();
    if (errcode_ret) {
        *errcode_ret = CL_SUCCESS;
    }

    return CQ;
}

CL_API_ENTRY cl_mem CL_API_CALL
clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size,
               void *host_ptr, cl_int *errcode_ret)
CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clCreateBuffer()\n");

    _cl_mem *buf;

    if (flags & CL_MEM_USE_HOST_PTR) {
        buf = (_cl_mem *)host_ptr;
        if (errcode_ret)
            *errcode_ret = buf ? CL_SUCCESS :
                                 CL_MEM_OBJECT_ALLOCATION_FAILURE;
        DPRINT("returning from clCreateBuffer()\n");
        return buf;
    } else {
        // Ensure the allocated buffer is cache block aligned so that
        // coalescing is maximized and bandwidth is reduced.
        posix_memalign((void**)&buf, 64, size);
    }
    if (buf && (flags & CL_MEM_COPY_HOST_PTR)) {
        memcpy(buf, host_ptr, size);
    }
    if (errcode_ret)
        *errcode_ret = buf ? CL_SUCCESS :
                             CL_MEM_OBJECT_ALLOCATION_FAILURE;
    memTracker.insert(buf);
    DPRINT("returning from clCreateBuffer()\n");

    return buf;
}

CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithSource(cl_context context, cl_uint count,
                          const char **strings, const size_t *lengths,
                          cl_int *errcode_ret)
CL_API_SUFFIX__VERSION_1_0
{
    _cl_program *program;
    // addSource sets the program
    cl_int err = context->addSource(count, strings, lengths, &program);
    // we set the error code if the programmer
    // did not pass nullptr for errcode_ret
    if (errcode_ret)
      *errcode_ret = err;

    return program;
}

extern CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithBinary(cl_context, cl_uint, const cl_device_id*,
                          const size_t*, const unsigned char**,
                          cl_int*, cl_int*)
CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clCreateProgramWithBinary()\n");
    clWarn("clCreateProgramWithBinary unimplemented\n");
}

CL_API_ENTRY cl_int CL_API_CALL
clBuildProgram(cl_program program, cl_uint num_devices,
               const cl_device_id *device_list, const char *options,
               void (CL_CALLBACK *pfn_notify)(cl_program program,
                                              void *user_data),
               void *user_data)
CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clBuildProgram()\n");

    clWarn("clBuildProgram unimplemented\n");

    return CL_SUCCESS;
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetProgramInfo(cl_program program, cl_program_info param_name,
                 size_t param_value_size, void *param_value,
                 size_t *param_value_size_ret)
CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clGetProgramInfo()\n");
    clWarn("clGetProgramInfo unimplemented\n");

    switch(param_name) {
      case CL_PROGRAM_REFERENCE_COUNT:
        clFatal("clGetProgramInfo: CL_PROGRAM_REFERENCE_COUNT not "
                "yet implemented\n");
        break;
      case CL_PROGRAM_CONTEXT:
        clFatal("clGetProgramInfo: CL_PROGRAM_CONTEXT not yet "
                "implemented\n");
        break;
      case CL_PROGRAM_NUM_DEVICES:
        clFatal("clGetProgramInfo: CL_PROGRAM_NUM_DEVICES not yet "
                "implemented\n");
        break;
      case CL_PROGRAM_DEVICES:
        clFatal("clGetProgramInfo: CL_PROGRAM_DEVICES not yet implemented\n");
        break;
      case CL_PROGRAM_SOURCE:
        clFatal("clGetProgramInfo: CL_PROGRAM_SOURCE not yet implemented\n");
        break;
      case CL_PROGRAM_BINARY_SIZES:
        clFatal("clGetProgramInfo: CL_PROGRAM_BINARY_SIZES not yet "
                "implemented\n");
        break;
      case CL_PROGRAM_BINARIES:
        clFatal("clGetProgramInfo: CL_PROGRAM_BINARIES not yet "
                "implemented\n");
        break;
      default:
        return CL_INVALID_VALUE;
    }

    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetProgramBuildInfo(cl_program, cl_device_id, cl_program_build_info,
                      size_t, void*, size_t*)
CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clGetProgramBuildInfo()\n");
    clWarn("clGetProgramBuildInfo unimplemented\n");
}

/* Kernel Object APIs */
CL_API_ENTRY cl_kernel CL_API_CALL
clCreateKernel(cl_program program, const char *kernel_name,
               cl_int *errcode_ret)
CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clCreateKernel()\n");

    // init driver in case this is the first call.  this call will
    // return if the driver is already initialized.
    hsaDriverInit();

    const int MAX_KERNEL_NAME_LENGTH = 256; // totally arbitrary
    char long_kernel_name[MAX_KERNEL_NAME_LENGTH];
    snprintf(long_kernel_name, MAX_KERNEL_NAME_LENGTH,
             "__OpenCL_%s_kernel", kernel_name);

    DPRINT("clCreateKernel() %s\n", long_kernel_name);

    _cl_kernel *kernel = nullptr;
    for (int i = 0; i < hsaDriverSizes.num_kernels; ++i) {
        const HsaKernelInfo *kinfo = &hsaKernelInfo[i];
        const char *chk_name = &hsaStringTable[kinfo->name_offs];
        if (!strcmp(long_kernel_name, chk_name)) {
            kernel = new _cl_kernel(kernel_name, &hsaCode[kinfo->code_offs],
            kinfo->sRegCount, kinfo->dRegCount,
            kinfo->cRegCount, kinfo->private_mem_size,
            kinfo->spill_mem_size,
            kinfo->static_lds_size);
            break;
         }
    }

    if (!kernel) {
        fprintf(stderr, "Can't find kernel %s\n", kernel_name);
        if (errcode_ret)
            *errcode_ret = CL_INVALID_KERNEL_NAME;

        return nullptr;
    }

    cl_int tmp = program->addFunction(kernel);

    if (tmp == CL_SUCCESS) {
        if (errcode_ret) {
            *errcode_ret = tmp;
        }
        return kernel;
    } else {
        if (errcode_ret)
            *errcode_ret = CL_BUILD_ERROR;

        return nullptr;
    }
}

/* Device APIs */
CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type,
               cl_uint num_entries, cl_device_id *devices,
               cl_uint *num_devices)
CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clGetDeviceIDs()\n");

    if (!platform || platform->ID != theOnlyPlatform->getID()->ID) {
        return CL_INVALID_PLATFORM;
    }

    if (device_type & ~(CL_DEVICE_TYPE_ALL | CL_DEVICE_TYPE_CPU |
        CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR |
        CL_DEVICE_TYPE_DEFAULT) ) {
        return CL_INVALID_DEVICE_TYPE;
    }

    if ((!num_entries && devices) || (!num_devices && !devices)) {
        return CL_INVALID_VALUE;
    }

    cl_uint max_dev = 0;
    if (num_devices) {
        if (device_type & CL_DEVICE_TYPE_ALL) {
            *num_devices = theOnlyPlatform->numGPUs() +
                            theOnlyPlatform->numCPUs();
        } else {
            *num_devices = 0;
            if (device_type & CL_DEVICE_TYPE_CPU) {
                *num_devices += theOnlyPlatform->numGPUs();
            }
            if (device_type & CL_DEVICE_TYPE_GPU) {
                *num_devices += theOnlyPlatform->numCPUs();
            }
        }
        max_dev = *num_devices;
    }

    cl_uint num_dev_returned = 0;
    if (devices) {
        num_dev_returned += theOnlyPlatform->getGPUs(devices, num_entries);
        num_dev_returned +=
            theOnlyPlatform->getCPUs(devices, num_entries - num_dev_returned);
    }

    if (!(max_dev || num_dev_returned)) {
        return CL_DEVICE_NOT_FOUND;
    }

    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceInfo(cl_device_id device, cl_device_info param_name,
                size_t param_value_size, void *param_value,
                size_t *param_value_size_ret)
CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clGetDeviceInfo()\n");

    // init driver in case this is the first call.  this call will
    // return if the driver is already initialized.
    hsaDriverInit();

    int vector_width = 0;

    if (!theOnlyPlatform->isValidDev(device)) {
        return CL_INVALID_DEVICE;
    }

    if (param_value_size_ret) {
        *param_value_size_ret = 0;
    }

    char *strRet = (char*)param_value;
    const char *strSrc;
    switch (param_name) {
      case CL_DEVICE_TYPE:
        if (param_value_size_ret) {
            *param_value_size_ret = sizeof(cl_device_type);
        }

        if (param_value) {
            if (param_value_size >= sizeof(cl_device_type)) {
               *((cl_device_type*)(param_value)) = CL_DEVICE_TYPE_GPU;
            } else {
               return CL_INVALID_VALUE;
            }
        }
        break;
      case CL_DEVICE_VENDOR_ID:
        if (param_value_size_ret) {
            *param_value_size_ret = sizeof(cl_uint);
        }

        if (param_value) {
            if (param_value_size >= sizeof(cl_uint)) {
               *((cl_uint*)(param_value)) = 0;
            } else {
               return CL_INVALID_VALUE;
            }
        }
        break;
      case CL_DEVICE_MAX_COMPUTE_UNITS:
        if (param_value_size_ret) {
            *param_value_size_ret = sizeof(cl_uint);
        }

        if (param_value) {
            if (param_value_size >= sizeof(cl_uint)) {
                *((cl_uint*)(param_value)) = numCUs;
            } else {
               return CL_INVALID_VALUE;
            }
        }
        break;
      case CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:
        if (param_value_size_ret) {
            *param_value_size_ret = sizeof(cl_uint);
        }

        if (param_value) {
            if (param_value_size >= sizeof(cl_uint)) {
               *((cl_uint*)(param_value)) = MAX_WI_DIM;
            } else {
               return CL_INVALID_VALUE;
            }
        }
        break;
      case CL_DEVICE_MAX_WORK_ITEM_SIZES:
        if (param_value_size_ret) {
            *param_value_size_ret = sizeof(size_t) * 3;
        }

        if (param_value) {
            if (param_value_size >= sizeof(cl_uint)) {
                size_t *ret = (size_t*)(param_value);
                ret[0] = MAX_WI_DIM0;
                ret[1] = MAX_WI_DIM1;
                ret[2] = MAX_WI_DIM2;
            } else {
               return CL_INVALID_VALUE;
            }
        }
        break;
      case CL_DEVICE_MAX_WORK_GROUP_SIZE:
        if (param_value_size_ret) {
            *param_value_size_ret = sizeof(size_t);
        }

        if (param_value) {
            if (param_value_size >= sizeof(cl_uint)) {
               *((size_t*)(param_value)) = MAX_WG_SIZE;
            } else {
               return CL_INVALID_VALUE;
            }
        }
        break;
      case CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR:
        vector_width = !vector_width ?
            VecSize * sizeof(double) / sizeof(char) : vector_width;
      case CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF:
        vector_width = !vector_width ?
            VecSize * sizeof(double) / sizeof(cl_half) : vector_width;
      case CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT:
        vector_width = !vector_width ?
            VecSize * sizeof(double) / sizeof(short) : vector_width;
      case CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT:
        vector_width = !vector_width ?
            VecSize * sizeof(double) / sizeof(int) : vector_width;
      case CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG:
        vector_width = !vector_width ?
            VecSize * sizeof(double) / sizeof(long) : vector_width;
      case CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT:
        vector_width = !vector_width ?
            VecSize * sizeof(double) / sizeof(float) : vector_width;
      case CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE:
        vector_width = !vector_width ?
            VecSize * sizeof(double) / sizeof(double) : vector_width;
        DPRINT("vector_width = %d\n", vector_width);

        if (param_value_size_ret) {
            *param_value_size_ret = sizeof(cl_uint);
        }

        if (param_value) {
            if (param_value_size >= sizeof(cl_uint)) {
               *((cl_uint*)(param_value)) = vector_width;
            } else {
               return CL_INVALID_VALUE;
            }
        }
        break;
      case CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR:
        vector_width = !vector_width ?
            VecSize * sizeof(double) / sizeof(char) : vector_width;
      case CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF:
        vector_width = !vector_width ?
            VecSize * sizeof(double) / sizeof(cl_half) : vector_width;
      case CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT:
        vector_width = !vector_width ?
            VecSize * sizeof(double) / sizeof(short) : vector_width;
      case CL_DEVICE_NATIVE_VECTOR_WIDTH_INT:
        vector_width = !vector_width ?
            VecSize * sizeof(double) / sizeof(int) : vector_width;
      case CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG:
        vector_width = !vector_width ?
            VecSize * sizeof(double) / sizeof(long) : vector_width;
      case CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT:
        vector_width = !vector_width ?
            VecSize * sizeof(double) / sizeof(float) : vector_width;
      case CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE:
        vector_width = !vector_width ?
            VecSize * sizeof(double) / sizeof(double) : vector_width;
        DPRINT("vector_width = %d\n", vector_width);

        if (param_value_size_ret) {
            *param_value_size_ret = sizeof(cl_uint);
        }

        if (param_value) {
            if (param_value_size >= sizeof(cl_uint)) {
               *((cl_uint*)(param_value)) = vector_width;
            } else {
               return CL_INVALID_VALUE;
            }
        }
        break;
      case CL_DEVICE_MAX_CLOCK_FREQUENCY:
        clWarn("clGetDeviceInfo: ignoring request for "
               "CL_DEVICE_MAX_CLOCK_FREQUENCY...\n");
        break;
      case CL_DEVICE_ADDRESS_BITS:
        if (param_value_size_ret) {
            *param_value_size_ret = sizeof(cl_uint);
        }

        if (param_value) {
            if (param_value_size >= sizeof(cl_uint)) {
               *((cl_uint*)(param_value)) = sizeof(void*) * 8;
            } else {
               return CL_INVALID_VALUE;
            }
        }
        break;
        clWarn("clGetDeviceInfo: ignoring request for "
               "CL_DEVICE_ADDRESS_BITS...\n");
        break;
      case CL_DEVICE_MAX_MEM_ALLOC_SIZE:
        clWarn("clGetDeviceInfo: ignoring request for "
               "CL_DEVICE_MAX_MEM_ALLOC_SIZE...\n");
        break;
      case CL_DEVICE_IMAGE_SUPPORT:
        clWarn("clGetDeviceInfo: ignoring request for "
               "CL_DEVICE_IMAGE_SUPPORT...\n");
        break;
      case CL_DEVICE_MAX_READ_IMAGE_ARGS:
        clWarn("clGetDeviceInfo: ignoring request for "
               "CL_DEVICE_MAX_READ_IMAGE_ARGS...\n");
        break;
      case CL_DEVICE_MAX_WRITE_IMAGE_ARGS:
        clWarn("clGetDeviceInfo: CL_DEVICE_MAX_WRITE_IMAGE_ARGS not "
               "implemented\n");
        break;
      case CL_DEVICE_IMAGE2D_MAX_WIDTH:
        clWarn("clGetDeviceInfo: CL_DEVICE_IMAGE2D_MAX_WIDTH not "
               "implemented\n");
        break;
      case CL_DEVICE_IMAGE2D_MAX_HEIGHT:
        clWarn("clGetDeviceInfo: CL_DEVICE_IMAGE2D_MAX_HEIGHT not "
               "implemented\n");
        break;
      case CL_DEVICE_IMAGE3D_MAX_WIDTH:
        clWarn("clGetDeviceInfo: CL_DEVICE_IMAGE3D_MAX_WIDTH not "
               "implemented\n");
        break;
      case CL_DEVICE_IMAGE3D_MAX_HEIGHT:
        clWarn("clGetDeviceInfo: CL_DEVICE_IMAGE3D_MAX_HEIGHT not "
               "implemented\n");
        break;
     case CL_DEVICE_IMAGE3D_MAX_DEPTH:
        clWarn("clGetDeviceInfo: CL_DEVICE_IMAGE3D_MAX_DEPTH not "
               "implemented\n");
        break;
      case CL_DEVICE_MAX_SAMPLERS:
        clWarn("clGetDeviceInfo: CL_DEVICE_MAX_SAMPLERS not implemented\n");
        break;
      case CL_DEVICE_MAX_PARAMETER_SIZE:
        clWarn("clGetDeviceInfo: CL_DEVICE_MAX_PARAMETER_SIZE not "
               "implemented\n");
        break;
      case CL_DEVICE_MEM_BASE_ADDR_ALIGN:
        clWarn("clGetDeviceInfo: CL_DEVICE_MEM_BASE_ADDR_ALIGN not "
               "implemented\n");
        break;
      case CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE:
        clWarn("clGetDeviceInfo: CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE not "
               "implemented\n");
        break;
      case CL_DEVICE_SINGLE_FP_CONFIG:
        clWarn("clGetDeviceInfo: CL_DEVICE_SINGLE_FP_CONFIG not "
               "implemented\n");
        break;
      case CL_DEVICE_DOUBLE_FP_CONFIG:
        clWarn("clGetDeviceInfo: CL_DEVICE_DOUBLE_FP_CONFIG not "
               "implemented\n");
        break;
      case CL_DEVICE_GLOBAL_MEM_CACHE_TYPE:
        clWarn("clGetDeviceInfo: CL_DEVICE_GLOBAL_MEM_CACHE_TYPE not "
               "implemented\n");
        break;
      case CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE:
        clWarn("clGetDeviceInfo: CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE not "
               "implemented\n");
        break;
      case CL_DEVICE_GLOBAL_MEM_CACHE_SIZE:
        clWarn("clGetDeviceInfo: CL_DEVICE_GLOBAL_MEM_CACHE_SIZE not "
               "implemented\n");
        break;
      case CL_DEVICE_GLOBAL_MEM_SIZE:
        clWarn("clGetDeviceInfo: CL_DEVICE_GLOBAL_MEM_SIZE not "
               "implemented\n");
        break;
      case CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:
        clWarn("clGetDeviceInfo: CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE not "
               "implemented\n");
        break;
      case CL_DEVICE_MAX_CONSTANT_ARGS:
        clWarn("clGetDeviceInfo: CL_DEVICE_MAX_CONSTANT_ARGS not "
               "implemented\n");
        break;
      case CL_DEVICE_LOCAL_MEM_TYPE:
        clWarn("clGetDeviceInfo: CL_DEVICE_LOCAL_MEM_TYPE not implemented\n");
        break;
      case CL_DEVICE_LOCAL_MEM_SIZE:
        if (param_value_size_ret) {
            *param_value_size_ret = sizeof(unsigned long long);
        }

        if (param_value) {
            if (param_value_size >= sizeof(unsigned long long)) {
               *((unsigned long long*)(param_value)) = MAX_LDS_SIZE;
            } else {
               return CL_INVALID_VALUE;
            }
        }
        break;
      case CL_DEVICE_ERROR_CORRECTION_SUPPORT:
        clWarn("clGetDeviceInfo: CL_DEVICE_ERROR_CORRECTION_SUPPORT not "
               "implemented\n");
        break;
      case CL_DEVICE_PROFILING_TIMER_RESOLUTION:
        clWarn("clGetDeviceInfo: CL_DEVICE_PROFILING_TIMER_RESOLUTION not "
               "implemented\n");
        break;
      case CL_DEVICE_ENDIAN_LITTLE:
        clWarn("clGetDeviceInfo: CL_DEVICE_ENDIAN_LITTLE not implemented\n");
        break;
      case CL_DEVICE_HOST_UNIFIED_MEMORY:
        clWarn("clGetDeviceInfo: CL_DEVICE_HOST_UNIFIED_MEMORY not "
               "implemented\n");
        break;
      case CL_DEVICE_AVAILABLE:
        clWarn("clGetDeviceInfo: CL_DEVICE_AVAILABLE not implemented\n");
        break;
      case CL_DEVICE_COMPILER_AVAILABLE:
        clWarn("clGetDeviceInfo: CL_DEVICE_COMPILER_AVAILABLE not "
               "implemented\n");
        break;
      case CL_DEVICE_EXECUTION_CAPABILITIES:
        clWarn("clGetDeviceInfo: CL_DEVICE_EXECUTION_CAPABILITIES not "
               "implemented\n");
        break;
      case CL_DEVICE_QUEUE_PROPERTIES:
        clWarn("clGetDeviceInfo: CL_DEVICE_QUEUE_PROPERTIES not "
               "implemented\n");
        break;
      case CL_DEVICE_PLATFORM:
        clWarn("clGetDeviceInfo: CL_DEVICE_PLATFORM not implemented\n");
        break;
      case CL_DEVICE_NAME:
        strSrc = "HSAIL-GPU";
        if (param_value_size_ret) {
            *param_value_size_ret = strlen(strSrc) + 1;
        }

        if (param_value) {
            if (param_value_size >= strlen(strSrc) + 1) {
               strcpy (strRet, strSrc);
            } else {
               return CL_INVALID_VALUE;
            }
        }
        break;
      case CL_DEVICE_VENDOR:
        clWarn("clGetDeviceInfo: CL_DEVICE_VENDOR not implemented\n");
        break;
      case CL_DRIVER_VERSION:
        clWarn("clGetDeviceInfo: CL_DRIVER_VERSION not implemented\n");
        break;
      case CL_DEVICE_OPENCL_C_VERSION:
        clWarn("clGetDeviceInfo: CL_DEVICE_OPENCL_C_VERSION not "
               "implemented\n");
        break;
      case CL_DEVICE_PROFILE:
        clWarn("clGetDeviceInfo: CL_DEVICE_PROFILE not implemented\n");
        break;
      case CL_DEVICE_VERSION:
        clWarn("clGetDeviceInfo: CL_DEVICE_VERSION not implemented\n");
        break;
      case CL_DEVICE_EXTENSIONS:
        strSrc = "cl_khr_byte_addressable_store";
        if (param_value_size_ret) {
            *param_value_size_ret = strlen(strSrc) + 1;
        }

        if (param_value) {
            if (param_value_size >= strlen(strSrc) + 1) {
               strcpy (strRet, strSrc);
            } else {
               return CL_INVALID_VALUE;
            }
        }
        break;
      default:
        return CL_INVALID_VALUE;
    }

    return CL_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////

CL_API_ENTRY cl_int CL_API_CALL
clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size,
               const void *arg_value)
CL_API_SUFFIX__VERSION_1_0
{
    if (!arg_value) {
        DPRINT("clSetKernelArg(%p, %d, %d, nullptr)\n",
               (void*)kernel, arg_index, (int)arg_size);
    } else if (arg_size == 4) {
        DPRINT("clSetKernelArg(%p, %d, %d, %#x)\n",
               (void*)kernel, arg_index, (int)arg_size,
               *(uint32_t*)arg_value);
    } else if (arg_size == 8) {
        DPRINT("clSetKernelArg(%p, %d, %d, %#llx)\n",
               (void*)kernel, arg_index, (int)arg_size,
               *(uint64_t*)arg_value);
    } else {
        DPRINT("clSetKernelArg(%p, %d, %d, %p)\n",
               (void*)kernel, arg_index, (int)arg_size, arg_value);
    }

    kernel->addArg(arg_index + DEFAULT_OCL_KERN_ARGS, arg_size, arg_value);

    return CL_SUCCESS;
}

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueUnmapMemObject(cl_command_queue, cl_mem, void*, cl_uint,
                        const cl_event*, cl_event*)
CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clEnqueueUnmapMemObject()\n");
    clWarn("clEnqueueUnmapMemObject unimplemented\n");

    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel,
                       cl_uint work_dim, const size_t *global_work_offset,
                       const size_t *global_work_size,
                       const size_t *local_work_size,
                       cl_uint num_events_in_wait_list,
                       const cl_event *event_wait_list, cl_event *event)
CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clEnqueueNDRangeKernel()\n");
    HsaQueueEntry *hsa_task = (HsaQueueEntry*)malloc(sizeof(HsaQueueEntry));
    HostState *host_state = (HostState*)malloc(sizeof(HostState));

    // the current version of the compiler adds 6 implicit arguments to an
    // OpenCL kernel
    if (!global_work_offset) {
        kernel->addOCLKernelOffsetArgs(0, 0);
        kernel->addOCLKernelOffsetArgs(1, 0);
        kernel->addOCLKernelOffsetArgs(2, 0);
    } else {
        kernel->addOCLKernelOffsetArgs(0, &global_work_offset[0]);
        kernel->addOCLKernelOffsetArgs(1, &global_work_offset[1]);
        kernel->addOCLKernelOffsetArgs(2, &global_work_offset[2]);
    }

    kernel->addOCLKernelOffsetArgs(3, 0);
    kernel->addOCLKernelOffsetArgs(4, 0);
    kernel->addOCLKernelOffsetArgs(5, 0);

    if (event) {
        *event = new _cl_event();
    }

    hsa_task->depends = (uint64_t)host_state;
    host_state->event = event ? (uint64_t)(*event) : 0;

    if (work_dim < 1 || work_dim > 3) {
        return CL_INVALID_WORK_DIMENSION;
    }

    for (cl_uint i = 0; i < work_dim; ++i) {
        hsa_task->gdSize[i] = global_work_size[i];
        if (local_work_size) {
            hsa_task->wgSize[i] = (local_work_size[i] > global_work_size[i]) ?
                                   global_work_size[i] : local_work_size[i];
        } else {
            hsa_task->wgSize[i] = global_work_size[i] > 256 ?
                256 : global_work_size[i];
            // only set 1 dimension automatically
            for (++i; i < work_dim; ++i)
                hsa_task->wgSize[i] = 1;
            break;
        }
    }
    for (cl_uint i = work_dim; i < 3; ++i) {
        hsa_task->gdSize[i] = 1;
        hsa_task->wgSize[i] = 1;
    }

    // allocate space for WF context
    int numWgTotal = 1;
    int wfPerWgTotal = 1;
    for (int i = 0; i < 3; ++i) {
        numWgTotal *= divCeil(hsa_task->gdSize[i], hsa_task->wgSize[i]);
        wfPerWgTotal *= (hsa_task->wgSize[i] + VecSize - 1) / VecSize;
    }
    int numCtx = numWgTotal * wfPerWgTotal;

    //////////////////////////////////////
    hsa_task->code_ptr = (uint64_t)kernel->code;
    DPRINT("launching %s\n", kernel->name);

    // setup arguments
    hsa_task->num_args = kernel->maxArgIdx + 1;
    int offset = 0;

    for (cl_uint i = 0; i <= kernel->maxArgIdx; i++) {
        // copy argument to argument buffer in allocated HSA queue entry
        DPRINT("HSA runtime: Offset %d\n", offset);
        hsa_task->offsets[i] = offset;
        if (kernel->argList[i].contents) {
            memcpy(hsa_task->args + offset, kernel->argList[i].contents,
                   kernel->argList[i].size);
        } else {
            // argument is __local pointer, i.e., LDS offset
            // must use groupMemOffset instead of contents
            *(uint64_t*)(hsa_task->args + offset) =
            (uint64_t)(kernel->argList[i].groupMemOffset);
        }

        offset += kernel->argList[i].size;

        if (i != kernel->maxArgIdx) {
            int pad = offset % kernel->argList[i + 1].size;
            pad = !pad ? 0 : kernel->argList[i + 1].size - pad;
            offset += pad;
        }
    }

    // Point the dispatcher to counter variable (tracking # of dispatches)
    // polled by runtime
    hsa_task->numDispLeft = (uint64_t)command_queue->numDispLeft;

    hsa_task->sRegCount = kernel->sRegCount;
    hsa_task->dRegCount = kernel->dRegCount;
    hsa_task->cRegCount = kernel->cRegCount;

    DPRINT("regs: s %d d %d c %d\n", hsa_task->sRegCount,
           hsa_task->dRegCount, hsa_task->cRegCount);

    // Allocate private memory for the kernel, the gdSize already
    // accounts for all the blocks
    int numWorkItems = hsa_task->gdSize[0] *
                       hsa_task->gdSize[1] *
                       hsa_task->gdSize[2];
    numWorkItems = numWorkItems % VecSize ? numWorkItems + VecSize -
            (numWorkItems % VecSize) : numWorkItems;

    hsa_task ->privMemPerItem = kernel->privateMemSize;
    hsa_task ->spillMemPerItem = kernel->spillMemSize;

    // Total of privMem and spillMem should be calculated with
    // Total number of wavefronts must be:
    // (Wavefronts * WavefrontSize)* privMemPerItem/spillMemPerItem.
    hsa_task->privMemTotal = (kernel->privateMemSize * numWorkItems);
    hsa_task->spillMemTotal = (kernel->spillMemSize * numWorkItems);

    // FIXME: There is a mismatch regarding the allcation and use of both
    // privateMemory and spliiMemory. On this (HSA_runtime), the total memory
    // for each buufer (spill or priv) is calucalted by muptiplying
    // total_number of workitems with spill/priv size per workitem.
    // On hsail GPU side, the accesses to spill/priv memory assumes
    // there is (VSZ * spill/priv per workitem) for each wavefront.
    // This becomes a problem when the (total workitems less than number
    // of wavefronts * VSZ). The multiplication by 8 below is a hack.
    // should be fixed. The multiply with 8 (below) is to make the allocated
    // memory for spill buffer large enough. Multipying with 64 would be
    // the most conservative approach.
    hsa_task->spillMemTotal = hsa_task->spillMemTotal * 8;

    hsa_task->privMemStart = hsa_task->privMemTotal > 0 ?
        (uint64_t)malloc(hsa_task->privMemTotal) : 0;
    DPRINT("hsa_task->privMemTotal=%d\n", hsa_task->privMemTotal);
    DPRINT("hsa_task->privMemStart=%p\n", (void*)hsa_task->privMemStart);

    hsa_task->spillMemStart = hsa_task->spillMemTotal > 0 ?
        (uint64_t)malloc(hsa_task->spillMemTotal) : 0;
    DPRINT("hsa_task->spillMemTotal=%d\n", hsa_task->spillMemTotal);
    DPRINT("hsa_task->spillMemStart=%p\n", (void*)hsa_task->spillMemStart);

    hsa_task->roMemTotal = hsaDriverSizes.readonly_size;
    hsa_task->roMemStart = (uint64_t)hsaReadonly;

    // initialize read-only memory

    // LDS storage will be allocated by CU
    hsa_task->ldsSize = kernel->groupMemSize;
    DPRINT("hsa_task->ldsSize=%d\n", hsa_task->ldsSize);

    // Point the dispatcher to done variables polled by runtime
    if (event) {
        hsa_task->addrToNotify = (uint64_t)&((*event)->done);
        (*event)->hsaTaskPtr = hsa_task;
    } else {
        hsa_task->addrToNotify = 0;
        hsa_task->depends = 0;
    }
    memcpy(hsaTaskPtr, hsa_task, sizeof(HsaQueueEntry));

    // notify the dispatch engine that the task params are complete
    *dispatcherDoorbell = 0;

    return CL_SUCCESS;
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetKernelWorkGroupInfo(cl_kernel kernel, cl_device_id device,
                         cl_kernel_work_group_info param_name,
                         size_t param_value_size, void *param_value,
                         size_t *param_value_size_ret)
CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clGetKernelWorkGroupInfo()\n");

    switch (param_name) {
      case CL_KERNEL_WORK_GROUP_SIZE:
        if (param_value_size_ret) {
            *param_value_size_ret = sizeof(size_t);
        }

        if (param_value) {
            if (param_value_size >= sizeof(size_t)) {
                *((size_t*)(param_value)) = MAX_WG_SIZE;
            } else {
                return CL_INVALID_VALUE;
            }
        }
        break;
      case CL_KERNEL_COMPILE_WORK_GROUP_SIZE:
        if(param_value_size_ret) {
            *param_value_size_ret = sizeof(size_t) * 3;
        }

        if (param_value) {
            if (param_value_size >= sizeof(size_t) * 3) {
                size_t *dest = (size_t*)param_value;
                dest[0] = 0;
                dest[1] = 0;
                dest[2] = 0;
            } else {
                return CL_INVALID_VALUE;
            }
        }
        break;
      case CL_KERNEL_LOCAL_MEM_SIZE:
        if (param_value_size_ret) {
            *param_value_size_ret = sizeof(cl_ulong);
        }

        if (param_value) {
            if (param_value_size >= sizeof(cl_ulong)) {
                *((cl_ulong*)(param_value)) = MAX_LDS_SIZE;
            } else {
                return CL_INVALID_VALUE;
            }
        }
        break;
      default:
        return CL_INVALID_VALUE;
    }

    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
clWaitForEvents(cl_uint num_events, const cl_event *event_list)
CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clWaitForEvents()\n");

    for (cl_uint i = 0; i < num_events; ++i) {
        while (!event_list[i]->done) {
            asm("hlt");
        }
    }
    return CL_SUCCESS;
}

extern CL_API_ENTRY cl_int CL_API_CALL
clGetEventInfo(cl_event event, cl_event_info param_name,
               size_t param_value_size, void *param_value,
               size_t *param_value_size_ret)
CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clGetEventInfo()\n");

    switch (param_name) {
      case CL_EVENT_COMMAND_QUEUE:
        clFatal("CL_EVENT_COMMAND_QUEUE: clGetEventInfo not yet "
                "implemented\n");
        break;
      case CL_EVENT_COMMAND_TYPE:
        clFatal("CL_EVENT_COMMAND_TYPE: clGetEventInfo not yet "
                "implemented\n");
        break;
      case CL_EVENT_COMMAND_EXECUTION_STATUS:
        if (param_value_size_ret) {
            *param_value_size_ret = sizeof(cl_int);
        }

        if (param_value) {
            if (param_value_size >= sizeof(cl_int)) {
                if(event->done) {
                    *((cl_int*)(param_value)) = CL_COMPLETE;
                } else {
                    *((cl_int*)(param_value)) = CL_RUNNING;
                }
            } else {
                return CL_INVALID_VALUE;
            }
        }
        break;
      case CL_EVENT_REFERENCE_COUNT:
        clFatal("CL_EVENT_REFERENCE_COUNT: clGetEventInfo not yet "
                "implemented\n");
        break;
      default:
        return CL_INVALID_VALUE;
    }

    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseEvent(cl_event event)
CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clReleaseEvent()\n");

    if (event->hsaTaskPtr) {
        if (event->hsaTaskPtr->privMemStart) {
            free((void*)(event->hsaTaskPtr->privMemStart));
        }
        if (event->hsaTaskPtr->spillMemStart) {
            free((char*)(event->hsaTaskPtr->spillMemStart));
        }
        free(event->hsaTaskPtr);
    }
    delete event;
    return CL_SUCCESS;
}

/* Profiling APIs */
extern CL_API_ENTRY cl_int CL_API_CALL
clGetEventProfilingInfo(cl_event event, cl_profiling_info param_name,
                        size_t param_value_size, void *param_value,
                        size_t *param_value_size_ret)
CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clGetEventProfilingInfo()\n");

    if (param_value_size_ret) {
        *param_value_size_ret = 0;
    }

    switch (param_name) {
      case CL_PROFILING_COMMAND_START:
        assert(param_value_size >= sizeof(uint64_t));
        *((uint64_t*)param_value) = event->start;
        if (param_value_size_ret) {
            *param_value_size_ret = sizeof(uint64_t);
        }
        break;
      case CL_PROFILING_COMMAND_END:
        assert(param_value_size >= sizeof(uint64_t));
        *((uint64_t*)param_value) = event->end;
        if (param_value_size_ret) {
            *param_value_size_ret = sizeof(uint64_t);
        }
        break;
      default:
        return CL_INVALID_VALUE;
    }
    return CL_SUCCESS;
}

/* Flush and Finish APIs */
extern CL_API_ENTRY cl_int CL_API_CALL
clFlush(cl_command_queue command_queue)
CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clFlush()\n");
    while (*(command_queue->numDispLeft) > 0);
    // asm("hlt") does not work here because there
    // is a race if the dispatcher called cpu->wakeup()
    // when the CPU is awake and hlt is the next CPU instruction
    // to be executed
    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
clFinish(cl_command_queue  command_queue) CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clFinish()\n");
    while (*(command_queue->numDispLeft) > 0);
    // asm("hlt") does not work here because there
    // is a race if the dispatcher called cpu->wakeup()
    // when the CPU is awake and hlt is the next CPU instruction
    // to be executed
    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer,
                    cl_bool blocking_read, size_t offset, size_t size,
                    void *ptr, cl_uint num_events_in_wait_list,
                    const cl_event *event_wait_list, cl_event *event)
CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clEnqueueReadBuffer()\n");
    // make sure buffer and ptr are not nullptr pointers
    if (!(buffer && ptr)) {
        return CL_INVALID_VALUE;
    }

    if (event) {
        *event = new _cl_event();
    }

    if ((!event_wait_list && num_events_in_wait_list > 0) ||
        (event_wait_list && !num_events_in_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    clWaitForEvents(num_events_in_wait_list, event_wait_list);
    clFinish(command_queue);
    if ((void*)buffer != ptr) {
        memcpy(ptr, (char*)buffer+offset, size);
    }

    if (event) {
        (*event)->done = true;
    }
    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer,
                     cl_bool blocking_write, size_t offset, size_t size,
                     const void *ptr, cl_uint num_events_in_wait_list,
                     const cl_event *event_wait_list, cl_event *event)
CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clEnqueueWriteBuffer()\n");

    if (!(buffer && ptr)) {
        return CL_INVALID_VALUE;
    }

    if (event) {
        *event = new _cl_event();
    }

    if ((!event_wait_list && num_events_in_wait_list > 0) ||
       (event_wait_list && !num_events_in_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    clWaitForEvents(num_events_in_wait_list, event_wait_list);
    if ((void*)buffer != ptr) {
        memcpy((char *)buffer+offset, ptr, size);
    }

    if (event) {
        (*event)->done = true;
    }

    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyBuffer(cl_command_queue command_queue, cl_mem src_buffer,
                    cl_mem dst_buffer, size_t src_offset, size_t dst_offset,
                    size_t size, cl_uint num_events_in_wait_list,
                    const cl_event *event_wait_list, cl_event *event)
CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clEnqueueCopyBuffer()\n");

    if (!src_buffer) {
        return CL_INVALID_VALUE;
    }

    if (!dst_buffer) {
        return CL_INVALID_VALUE;
    }

    if (event) {
        *event = new _cl_event();
    }

    if ((!event_wait_list && num_events_in_wait_list > 0) ||
       (event_wait_list && !num_events_in_wait_list)) {
        return CL_INVALID_EVENT_WAIT_LIST;
    }

    clWaitForEvents(num_events_in_wait_list, event_wait_list);

    if (((char*)(dst_buffer)+dst_offset) != (char*)(src_buffer)) {
        memcpy(((char*)(dst_buffer)+dst_offset), (char*)src_buffer, size);
    }

    if (event) {
        (*event)->done = true;
    }

    return CL_SUCCESS;
}

CL_API_ENTRY void * CL_API_CALL
clEnqueueMapBuffer(cl_command_queue command_queue, cl_mem buffer,
                   cl_bool blocking_map, cl_map_flags map_flags,
                   size_t offset, size_t size,
                   cl_uint num_events_in_wait_list,
                   const cl_event *event_wait_list, cl_event *event,
                   cl_int *errcode_ret)
CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clEnqueueMapBuffer()\n");

    if (event) {
        *event = new _cl_event();
    }

    if ((!event_wait_list && num_events_in_wait_list > 0) ||
        (event_wait_list && !num_events_in_wait_list)) {
        if (errcode_ret) {
            *errcode_ret = CL_INVALID_EVENT_WAIT_LIST;
        }

        return nullptr;
    }

    clWaitForEvents(num_events_in_wait_list, event_wait_list);

    if (event) {
        (*event)->done = true;
    }

    if (errcode_ret) {
        *errcode_ret = CL_SUCCESS;
    }

    return (void*)((char*)buffer + offset);
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseKernel(cl_kernel kernel) CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clReleaseKernel()\n");
    cl_int refcnt = refkernel[kernel];

    if (!refcnt)
       delete kernel;
    else if (refcnt > 0)
       refkernel[kernel]--;

    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseProgram(cl_program program) CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clReleaseProgram()\n");
    cl_int refcnt = refprogram[program];

    if (!refcnt)
        delete program;
    else if (refcnt > 0)
        refprogram[program]--;

    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseMemObject(cl_mem memobj) CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clReleaseMemObject()\n");

    if (memTracker.count(memobj)) {
        free(memobj);
    }

    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseCommandQueue(cl_command_queue command_queue)
CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clReleaseCommandQueue()\n");
    cl_int refcnt = refcmdqueue[command_queue];

    if (!refcnt)
        delete command_queue;
    else if (refcnt > 0)
        refcmdqueue[command_queue]--;

    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseContext(cl_context context) CL_API_SUFFIX__VERSION_1_0
{
    DPRINT("clReleaseContext()\n");
    cl_int refcnt = refcontext[context];

    if (!refcnt)
         delete context;
    else if (refcnt > 0)
        refcontext[context]--;

    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
clRetainContext(cl_context context) CL_API_SUFFIX__VERSION_1_0
{
    ++refcontext[context];
    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
clRetainMemObject(cl_mem memobj) CL_API_SUFFIX__VERSION_1_0
{
    ++refmem[memobj];
    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
clRetainKernel(cl_kernel kernel) CL_API_SUFFIX__VERSION_1_0
{
    ++refkernel[kernel];
    return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
clRetainCommandQueue(cl_command_queue command_queue)
CL_API_SUFFIX__VERSION_1_0
{
      ++refcmdqueue[command_queue];
      return CL_SUCCESS;

}

CL_API_ENTRY cl_int CL_API_CALL
clRetainProgram(cl_program program) CL_API_SUFFIX__VERSION_1_0
{
      ++refprogram[program];
      return CL_SUCCESS;

}
