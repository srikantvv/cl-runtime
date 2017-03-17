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

#ifndef __CL_RUNTIME_HH__
#define __CL_RUNTIME_HH__

#ifdef DEBUG
#define DPRINT(format, ...) fprintf(stderr, format, ## __VA_ARGS__)
#else
#define DPRINT(str, ...)
#endif

#include <algorithm>
#include <cstdio>
#include <map>

#include "CL/cl_platform.h"
#include "CL/cl.hpp"
#include "cl_event.h"
#include "cl_command_queue.h"

static const int MAX_WG_SIZE = 1024;

static const int MAX_WI_DIM = 3;
static const int MAX_WI_DIM0 = 1024;
static const int MAX_WI_DIM1 = 1024;
static const int MAX_WI_DIM2 = 64;

// Assume a maximum LDS space of 64k
static const int MAX_LDS_SIZE = 64 * 1024;

// maximum number of kernels per OpenCL binary
static const int MAX_FUNCTIONS_PER_BINARY = 32;
static const int MAX_ARGS_FOR_KERNELS = 40;

// Used in qstruct.h
typedef uint64_t Addr;

// general stuff
void clWarn(const char *s);
void clFatal(const char *s);

// opencl "built-in" types
struct _cl_platform_id {
    cl_uint ID;
};

class _cl_device_id {
  public:
    _cl_device_id() { }

    _cl_device_id(cl_device_type _type) : type(_type), ID(nextID++) { }

    ~_cl_device_id() { }

    _cl_command_queue *addCQ()
    {
        _cl_command_queue *CQ = new _cl_command_queue();
        cqList.push_back(CQ);
        return CQ;
    }

    cl_uint ID;
    cl_device_type type;

  private:
    std::vector<_cl_command_queue *> cqList;
    static cl_uint nextID;
};

struct source_desc
{
    char *str;
    size_t len;
};

struct argDesc {
    size_t size;
    void *contents;
    int groupMemOffset;
};

class _cl_kernel {
  public:
    _cl_kernel(const char *_name, const void *_code, unsigned sregs,
               unsigned dregs, unsigned cregs, unsigned privmem,
               unsigned spillmem, unsigned static_lds_size) :
        name(_name), code(_code), sRegCount(sregs), dRegCount(dregs),
        cRegCount(cregs), privateMemSize(privmem), spillMemSize(spillmem),
        groupMemSize(static_lds_size), maxArgIdx(0)
    {
        memset(&argList, 0, sizeof(argList));
    }

    ~_cl_kernel()
    {
        for (int i = 0; i < MAX_ARGS_FOR_KERNELS; i++)
            if (argList[i].contents)
                free(argList[i].contents);
    }

    void addArg(cl_uint arg_index, size_t arg_size, const void *arg_value)
    {
        assert(arg_index < MAX_ARGS_FOR_KERNELS);

        if (arg_value != nullptr) {
            argList[arg_index].size = arg_size;
            if (argList[arg_index].contents == nullptr) {
                argList[arg_index].contents =
                    (void*)malloc(argList[arg_index].size);
            }
            memcpy(argList[arg_index].contents, arg_value, arg_size);
        } else {
            argList[arg_index].size = sizeof(uint64_t);
            // assume a null pointer value means it's group memory
            // that needs to be dynamically allocated
            argList[arg_index].contents = nullptr;
            argList[arg_index].groupMemOffset = groupMemSize;
            groupMemSize += (arg_size + 7) & ~7; // force 8 byte alignment
        }

        if (maxArgIdx < arg_index) {
            maxArgIdx = arg_index;
        }
    }

    void addOCLKernelOffsetArgs(cl_uint arg_index, const void *arg_value)
    {
        assert(arg_index < MAX_ARGS_FOR_KERNELS);
        argList[arg_index].size = sizeof(uint64_t);
        argList[arg_index].contents = nullptr;

        if (arg_value) {
            if (argList[arg_index].contents == nullptr) {
                argList[arg_index].contents =
                    (void*)malloc(argList[arg_index].size);
            }

            memcpy(argList[arg_index].contents, arg_value,
                   argList[arg_index].size);
        }

        if (maxArgIdx < arg_index) {
            maxArgIdx = arg_index;
        }
    }

    const char *name;
    const void *code;

    unsigned int privateMemSize;
    unsigned int spillMemSize;
    unsigned int groupMemSize;
    unsigned int sRegCount; // Number of s registers
    unsigned int dRegCount; // Number of d registers
    unsigned int cRegCount; // Number of c registers

    cl_uint maxArgIdx;
    argDesc argList[MAX_ARGS_FOR_KERNELS];
};

class _cl_program {
  public:
    _cl_program() : numFunctions(0)
    {
        kernList =
           (_cl_kernel**)malloc(sizeof(_cl_kernel*)*MAX_FUNCTIONS_PER_BINARY);
    }

    ~_cl_program()
    {
        /*free(kernList);
        for(cl_uint i=0; i<numSrcStrings; i++) {
            free(srcStrings[i]->str);
            free(srcStrings[i]);
        }
        free(srcStrings);*/
    }

    source_desc **srcStrings;
    cl_uint numSrcStrings;
    _cl_kernel **kernList;

    cl_int addFunction(_cl_kernel *kernel)
    {
        if (numFunctions == MAX_FUNCTIONS_PER_BINARY) {
            printf("Unable to register function %s (increase "
                   "MAX_FUNCTIONS_PER_BINARY)\n", kernel->name);
            return CL_OUT_OF_HOST_MEMORY;
        }

        kernList[numFunctions] = kernel;
        numFunctions++;

        return CL_SUCCESS;
    }

  private:
    cl_uint numFunctions;
    char *symbolTable;
    int numAllocatedSymbols;
};

class _cl_context {
  public:
    _cl_context(std::vector<_cl_device_id *> &dev_list,
                cl_device_type device_type) : deviceType(device_type)
    {
        refCount = 1;

        // create device list
        numDevices = dev_list.size();
        devList = new _cl_device_id[numDevices];

        //enumerate devices
        cl_uint devCount = 0;
        for (devCount = 0; devCount < numDevices; devCount++) {
            devList[devCount].ID = dev_list[devCount]->ID;
            devList[devCount].type = device_type;
        }
    }

    ~_cl_context()
    {
        //delete devList;
    }

    cl_int addSource(cl_uint count, const char **strings,
                     const size_t *lengths, _cl_program **program)
    {
        if (count == 0 || strings == nullptr || *strings == nullptr) {
            _cl_program *_program = new _cl_program();
            *program = _program;
            return CL_SUCCESS;
        }

        _cl_program *_program = new _cl_program();
        _program->srcStrings = (source_desc**)malloc(sizeof(source_desc*));
        _program->numSrcStrings = count;

        for (cl_uint i; i < count; i++) {
            if (strings[i] == nullptr) {
                return CL_INVALID_VALUE;
            }

            source_desc *cur_src = (source_desc *)malloc(sizeof(source_desc));

            if (lengths == nullptr || lengths[i] == 0) {
                size_t bytes = strlen(strings[i]) + 1;
                cur_src->str = (char*)malloc(bytes);
                cur_src->len = 0;
                strcpy(cur_src->str, strings[i]);
                cur_src->str[strlen(strings[i])] = '\0';
            } else {
                cur_src->str = (char *)malloc(lengths[i]+1);
                cur_src->len = lengths[i];
                memcpy(cur_src->str, strings[i], lengths[i]);
                cur_src->str[lengths[i]] = '\0';
            }

            sourceList.push_back(cur_src);
            _program->srcStrings[i] = cur_src;
        }

        *program = _program;

        return CL_SUCCESS;
    }

    bool isValidDev(_cl_device_id *dev)
    {
        uint64_t dev_u64 = (uint64_t)dev;
        uint64_t devList_u64 = (uint64_t)devList;

        if (dev_u64 >= devList_u64 && dev_u64 <= devList_u64 + numDevices
            && (dev_u64 - devList_u64) % sizeof(_cl_device_id) == 0 ) {
            return true;
        }

        return false;
    }

    cl_uint getRefCount() { return refCount; }
    cl_uint getNumDevices() { return numDevices; }
    cl_device_type getDevType() { return deviceType; }

    _cl_device_id *devList;

  private:
    cl_uint refCount;
    cl_uint numDevices;
    cl_device_type deviceType;

    std::vector<source_desc *> sourceList;
};

class platform {
  public:
    platform()
    {
        clID.ID = 0;
        gpuDevList.push_back(new _cl_device_id(CL_DEVICE_TYPE_GPU));
    }

    _cl_platform_id *getID() { return &clID; }

    cl_int addContext(cl_device_type device_type, _cl_context **context)
    {
        std::vector<_cl_device_id *> emptyGpuDevList;
        std::vector<_cl_device_id *> emptyCpuDevList;

        if (device_type == CL_DEVICE_TYPE_GPU) {
            *context = new _cl_context(gpuDevList, device_type);
        } else if (device_type == CL_DEVICE_TYPE_CPU) {
            *context = new _cl_context(cpuDevList, device_type);
        } else {
            return CL_INVALID_DEVICE_TYPE;
        }

        if ((*context)->getNumDevices() < 1) {
            delete *context;
            *context = nullptr;

            return CL_DEVICE_NOT_AVAILABLE;
        }

        contextList.push_back(*context);
        return CL_SUCCESS;
    }

    bool isContextValid(_cl_context *context)
    {
        auto it = find (contextList.begin(), contextList.end(), context);

        if (it == contextList.end()) {
            return false;
        }

        return true;
    }

    bool isValidDev(_cl_device_id *dev)
    {
        for (cl_uint i = 0; i < gpuDevList.size(); ++i) {
            if (dev->ID == gpuDevList[i]->ID) {
                return true;
            }
        }

        for (cl_uint i = 0; i < cpuDevList.size(); ++i) {
            if (dev->ID == cpuDevList[i]->ID) {
                return true;
            }
        }
        return false;
    }

    cl_uint numGPUs() { return gpuDevList.size(); }
    cl_uint numCPUs() { return cpuDevList.size(); }

    cl_uint getGPUs(_cl_device_id **dev_list, cl_uint num_dev)
    {
        cl_uint num_ret = 0;

        for(num_ret = 0; num_ret < gpuDevList.size(); num_ret++) {
            if(num_ret+1 > num_dev) {
                break;
            }
            dev_list[num_ret] = gpuDevList[num_ret];
        }

        return num_ret;
    }

    cl_uint getCPUs(_cl_device_id **dev_list, cl_uint num_dev)
    {
        cl_uint num_ret = 0;

        for(; num_ret<cpuDevList.size(); ++num_ret) {
            if (num_ret + 1 > num_dev) {
                break;
            }
            dev_list[num_ret] = cpuDevList[num_ret];
        }

        return num_ret;
    }

  private:
    std::vector<_cl_context *> contextList;
    std::vector<_cl_device_id *> gpuDevList;
    std::vector<_cl_device_id *> cpuDevList;
    _cl_platform_id clID;
};

#endif // __CL_RUNTIME_HH__
