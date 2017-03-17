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

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <assert.h>

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))

// #include "cuda_runtime.h"

void usage()
{
  printf("usage: ./<binary_name> <sample_program>\n");
}

int main(int argc, char *argv[])
{
//     std::string sample_prog;
//
//   const char *code_ptrs[] = {"code0.txt", "code1.txt", "code2.txt", "code3.txt","code4.txt","code5.txt"};
//   int code_id = -1;
//
//   dim3 gridDim;
//   dim3 blockDim;
//
//   if (argc == 2)
//   {
//       sample_prog = argv[1];
//   }
//   else if (argc > 2)
//   {
//     usage();
//     exit(-1);
//   }
//
//   __cudaRegisterFatBinary(NULL);
//
//   // doing this so that runtime code remains the same for both CUDA apps and the test
//   // FSAIL kernels
//   if (!sample_prog.compare("code0.txt")) {
//     code_id = 0;
//   } else if (!sample_prog.compare("code1.txt")) {
//     code_id = 1;
//   } else if (!sample_prog.compare("code2.txt")) {
//     code_id = 2;
//   } else if (!sample_prog.compare("code3.txt")) {
//     code_id = 3;
//   } else if (!sample_prog.compare("code4.txt")) {
//     code_id = 4;
//   } else if (!sample_prog.compare("code5.txt")) {
//     code_id = 5;
//   } else {
//     assert(0);
//   }
//
//   __cudaRegisterFunction(NULL,
//       code_ptrs[code_id], // for an actual kernel this is a function pointer cast to char *
//       (char *)sample_prog.c_str(),
//       NULL,
//       -1,
//       NULL,
//       NULL,
//       NULL,
//       NULL,
//       NULL);
//
//   if (!sample_prog.compare("code0.txt")) {
//
//             int array_size = 64;
//             int size = array_size * sizeof(float);
//
//             float *host_a = (float *)malloc(size);
//             float *host_b = (float *)malloc(size);
//             float *host_c = (float *)malloc(size);
//
//             float *dev_a;
//             float *dev_b;
//             float *dev_c;
//
//             if (!host_a || !host_b || !host_c)
//             {
//                 exit(-1);
//             }
//
//             if (cudaMalloc((void **)&dev_a, size) != cudaSuccess)
//             {
//                 exit(-1);
//             }
//
//             if (cudaMalloc((void **)&dev_b, size) != cudaSuccess)
//             {
//                 cudaFree(dev_a);
//                 exit(-1);
//             }
//
//             if (cudaMalloc((void **)&dev_c, size) != cudaSuccess)
//             {
//                 cudaFree(dev_a);
//                 cudaFree(dev_b);
//                 exit(-1);
//             }
//
//             for (int i = 0; i < array_size; i++)
//             {
//                 host_a[i] = i;
//                 host_b[i] = i + 2;
//             }
//
//             cudaMemcpy(dev_a, host_a, size, cudaMemcpyHostToDevice);
//             cudaMemcpy(dev_b, host_b, size, cudaMemcpyHostToDevice);
//
//
//             gridDim.x = 1;
//             gridDim.y = 1;
//             gridDim.z = 1;
//
//             blockDim.x = array_size;
//             blockDim.y = 1;
//             blockDim.z = 1;
//
//             cudaConfigureCall(gridDim, blockDim);
//
//             if (cudaSetupArgument(&dev_a, sizeof(void *), 0) != cudaSuccess)
//             {
//                 cudaFree(dev_a);
//                 cudaFree(dev_b);
//                 cudaFree(dev_c);
//                 exit(-1);
//             }
//
//             if (cudaSetupArgument(&dev_b, sizeof(void *), 1 * sizeof(void *)) != cudaSuccess)
//             {
//                 cudaFree(dev_a);
//                 cudaFree(dev_b);
//                 cudaFree(dev_c);
//                 exit(-1);
//             }
//
//             if (cudaSetupArgument(&dev_c, sizeof(void *), 2 * sizeof(void *)) != cudaSuccess)
//             {
//                 cudaFree(dev_a);
//                 cudaFree(dev_b);
//                 cudaFree(dev_c);
//                 exit(-1);
//             }
//
//             if (cudaSetupArgument(&array_size, sizeof(int), 3 * sizeof(void *)) != cudaSuccess)
//             {
//                 cudaFree(dev_a);
//                 cudaFree(dev_b);
//                 cudaFree(dev_c);
//                 exit(-1);
//             }
//
//             cudaLaunch(code_ptrs[code_id]);
//
//             cudaGetLastError();
//
//             cudaMemcpy(host_c, dev_c, size, cudaMemcpyDeviceToHost);
//
//             // Error Checking
//             for (int i = 0; i < array_size; i++) {
//                 if (host_c[i] != (host_a[i] + host_b[i])) {
//                     printf("Incorrect Execution for i %d: %.0f != %.0f + %.0f\n",
//                             i, host_c[i], host_a[i], host_b[i]);
//                     exit(-1);
//                 }
//             }
//
//             cudaFree(dev_a);
//             cudaFree(dev_b);
//             cudaFree(dev_c);
//
//             free(host_a);
//             free(host_b);
//             free(host_c);
//
//   } else if (!sample_prog.compare("code1.txt")) {
//
//             int width = 16;
//             int height = 16;
//
//             int array_size = width * height;
//             int size = array_size * sizeof(float);
//
//             float *host_in = (float *)malloc(size);
//             float *host_out = (float *)malloc(size);
//
//             float *dev_in;
//             float *dev_out;
//
//             int index;
//
//             if (!host_in || !host_out)
//             {
//                 printf("failed to allocate memory on host\n");
//                 exit(-1);
//             }
//
//             if (cudaMalloc((void **)&dev_in, size) != cudaSuccess)
//             {
//                 printf("failed to allocate memory on device\n");
//                 exit(-1);
//             }
//
//             if (cudaMalloc((void **)&dev_out, size) != cudaSuccess)
//             {
//                 cudaFree(dev_in);
//
//                 printf("failed to allocate memory on device\n");
//                 exit(-1);
//             }
//
//             for (int i = 0; i < height; ++i)
//             {
//                 for (int j = 0; j < width; ++j)
//                 {
//                     index = i * width + j;
//                     host_in[index] = index;
//                 }
//             }
//
//             cudaMemcpy(dev_in, host_in, size, cudaMemcpyHostToDevice);
//
//             gridDim.x = 1;
//             gridDim.y = 1;
//             gridDim.z = 1;
//
//             blockDim.x = width;
//             blockDim.y = height;
//             blockDim.z = 1;
//
//             cudaConfigureCall(gridDim, blockDim);
//
//             if (cudaSetupArgument(&dev_in, sizeof(void *), 0) != cudaSuccess)
//             {
//                 cudaFree(dev_in);
//                 cudaFree(dev_out);
//                 exit(-1);
//             }
//
//             if (cudaSetupArgument(&dev_out, sizeof(void *), 1 * sizeof(void *)) != cudaSuccess)
//             {
//                 cudaFree(dev_in);
//                 cudaFree(dev_out);
//                 exit(-1);
//             }
//
//             if (cudaSetupArgument(&width, sizeof(int), 2 * sizeof(void *)) != cudaSuccess)
//             {
//                 cudaFree(dev_in);
//                 cudaFree(dev_out);
//                 exit(-1);
//             }
//
//             if (cudaSetupArgument(&height, sizeof(int), 2 * sizeof(void *) + sizeof(int)) != cudaSuccess)
//             {
//                 cudaFree(dev_in);
//                 cudaFree(dev_out);
//                 exit(-1);
//             }
//
//             cudaLaunch(code_ptrs[code_id]);
//
//             cudaGetLastError();
//
//             cudaMemcpy(host_out, dev_out, size, cudaMemcpyDeviceToHost);
//
//             // Error Checking
//             for (int j = 0; j < height; j++) {
//                 for (int i = 0; i < width; i++) {
//                     int index_in = i + j*width;
//                     int index_out = j + (i*height);
//                     if ((host_in[index_in] != host_out[index_out]) ||
//                             (host_in[index_in] != dev_in[index_in])) {
//                         printf("Incorrect Execution for i %d and j %d: %5.0f != %5.0f or %5.0f \n",
//                             i, j, host_in[index_in], host_out[index_out], dev_in[index_in]);
//                         exit(-1);
//                     }
//                 }
//             }
//
//             cudaFree(dev_in);
//             cudaFree(dev_out);
//
//             free(host_in);
//             free(host_out);
//
// } else if (!sample_prog.compare("code2.txt")) {
//
//             int array_size = 64;
//             int size = array_size * sizeof(unsigned int);
//
//             unsigned int *host_in = (unsigned int*)malloc(size);
//             unsigned int *host_out = (unsigned int*)malloc(size);
//
//             unsigned int *dev_in;
//             unsigned int *dev_out;
//
//             int index;
//
//             if (!host_in || !host_out)
//             {
//                 printf("failed to allocate memory on host\n");
//                 exit(-1);
//             }
//
//             if (cudaMalloc((void **)&dev_in, size) != cudaSuccess)
//             {
//                 printf("failed to allocate memory on device\n");
//                 exit(-1);
//             }
//
//             if (cudaMalloc((void **)&dev_out, size) != cudaSuccess)
//             {
//                 cudaFree(dev_in);
//
//                 printf("failed to allocate memory on device\n");
//                 exit(-1);
//             }
//
//             for (int i = 0; i < array_size; ++i)
//             {
//                 host_in[i] = i;
//             }
//
//             cudaMemcpy(dev_in, host_in, size, cudaMemcpyHostToDevice);
//
//             gridDim.x = 1;
//             gridDim.y = 1;
//             gridDim.z = 1;
//
//             blockDim.x = array_size;
//             blockDim.y = 1;
//             blockDim.z = 1;
//
//             cudaConfigureCall(gridDim, blockDim);
//
//             if (cudaSetupArgument(&dev_in, sizeof(void *), 0) != cudaSuccess)
//             {
//                 cudaFree(dev_in);
//                 cudaFree(dev_out);
//                 exit(-1);
//             }
//
//             if (cudaSetupArgument(&dev_out, sizeof(void *), 1 * sizeof(void *)) != cudaSuccess)
//             {
//                 cudaFree(dev_in);
//                 cudaFree(dev_out);
//                 exit(-1);
//             }
//
//             cudaLaunch(code_ptrs[code_id]);
//
//             cudaGetLastError();
//
//             cudaMemcpy(host_out, dev_out, size, cudaMemcpyDeviceToHost);
//
//             cudaFree(dev_in);
//             cudaFree(dev_out);
//
//             free(host_in);
//             free(host_out);
//
//   } else if (!sample_prog.compare("code3.txt")) {
//
//             int width = 64;
//             int height = 64;
//
//             int array_size = width * height;
//             int size = array_size * sizeof(unsigned int);
//
//             unsigned int *host_in = (unsigned int*)malloc(size);
//             unsigned int *host_out = (unsigned int*)malloc(size);
//
//             unsigned int *dev_in;
//             unsigned int *dev_out;
//
//             int index;
//
//             if (!host_in || !host_out)
//             {
//                 printf("failed to allocate memory on host\n");
//                 exit(-1);
//             }
//
//             if (cudaMalloc((void **)&dev_in, size) != cudaSuccess)
//             {
//                 printf("failed to allocate memory on device\n");
//                 exit(-1);
//             }
//
//             if (cudaMalloc((void **)&dev_out, size) != cudaSuccess)
//             {
//                 cudaFree(dev_in);
//
//                 printf("failed to allocate memory on device\n");
//                 exit(-1);
//             }
//
//             for (int i = 0; i < height; i++) {
//                 for (int j = 0; j < width; j++) {
//                     index = i * width + j;
//                     // this simple initialization algorithm makes
//                     // error checking easier
//                     host_in[index] = j;
//                 }
//             }
//
//             cudaMemcpy(dev_in, host_in, size, cudaMemcpyHostToDevice);
//
//
//             gridDim.x = 2;
//             gridDim.y = 2;
//             gridDim.z = 1;
//
//             blockDim.x = width / 2;
//             blockDim.y = height / 2;
//             blockDim.z = 1;
//
//             cudaConfigureCall(gridDim, blockDim);
//
//             if (cudaSetupArgument(&dev_in, sizeof(void *), 0) != cudaSuccess)
//             {
//                 cudaFree(dev_in);
//                 cudaFree(dev_out);
//                 exit(-1);
//             }
//
//             if (cudaSetupArgument(&dev_out, sizeof(void *), 1 * sizeof(void *)) != cudaSuccess)
//             {
//                 cudaFree(dev_in);
//                 cudaFree(dev_out);
//                 exit(-1);
//             }
//
//             cudaLaunch(code_ptrs[code_id]);
//
//             cudaGetLastError();
//
//             cudaMemcpy(host_out, dev_out, size, cudaMemcpyDeviceToHost);
//
//             //error checking
//             //
//             // The following error checking is valid only
//             // for this specific hard-coded initialization
//             // algorithm where host_in[i*width+j] = j.
//             //
//             unsigned int sum;
//             for (int j = 0; j < width; j++) {
//                 if (j <= 1)
//                     sum = (j + 1)*15;
//                 else if (j >= width - 2)
//                     sum = 615 + 15*j;
//                 else
//                    sum = j*25;
//                 for (int i = 0; i < height; i++) {
//                     unsigned int f1 = host_out[i*width + j];
//                     if (sum != f1) {
//                         printf("Incorrect Execution for i %d and j %d: %u != %u\n",
//                             i, j, sum, f1);
//                         exit(-1);
//                     }
//                 }
//             }
//
//             cudaFree(dev_in);
//             cudaFree(dev_out);
//
//             free(host_in);
//             free(host_out);
//         }
// 	else if (!sample_prog.compare("code4.txt")) {
//
//             //int array_size0 = 2560*4;
//             //int array_size1 = 128;
// 	    int n_ops = 9;
//             int array_size0 = 1024;
//             int array_size1 = 64;
//             int size0 = array_size0 * sizeof(unsigned int);
//             int size1 = array_size1 * sizeof(unsigned int) * n_ops;
//
//             unsigned int *host_a = (unsigned int *)malloc(size0);
//             unsigned int *host_b = (unsigned int *)malloc(size0);
//             unsigned int *host_c = (unsigned int *)malloc(size1);
//             unsigned int *host_c_check = (unsigned int *)malloc(size1);
//
//             unsigned int *dev_a = 0;
//             unsigned int *dev_b = 0;
//             unsigned int *dev_c = 0;
//
// 	    try
// 	    {
// 		if (!host_a || !host_b || !host_c || !host_c_check)
// 		    throw("Cannot malloc buffers");
// 		if (cudaMalloc((void **)&dev_a, size0) != cudaSuccess)
// 		    throw("Cannot cudaMalloc buffers");
// 		if (cudaMalloc((void **)&dev_b, size0) != cudaSuccess)
// 		    throw("Cannot cudaMalloc buffers");
// 		if (cudaMalloc((void **)&dev_c, size1) != cudaSuccess)
// 		    throw("Cannot cudaMalloc buffers");
//
// 		for (int i = 0; i < array_size0; i++)
// 		    host_a[i] = random() % array_size1;
// 		for (int i = 0; i < array_size0; i++)
// 		    host_b[i] = random();
// 		for (int i = 0; i < array_size1*n_ops; i++)
// 		    host_c[i] = random();
// 		for (int i = 0; i < array_size1*n_ops; i++)
// 		    host_c_check[i] = host_c[i];
//
// 		for (int i = 0; i < array_size0; i++)
// 		    host_c_check[host_a[i]]+= host_b[i];
// 		for (int i = 0; i < array_size0; i++)
// 		    host_c_check[array_size1+host_a[i]]-= host_b[i];
// 		for (int i = 0; i < array_size0; i++)
// 		    host_c_check[array_size1*2+host_a[i]]&= host_b[i];
// 		for (int i = 0; i < array_size0; i++)
// 		    host_c_check[array_size1*3+host_a[i]]|= host_b[i];
// 		for (int i = 0; i < array_size0; i++)
// 		    host_c_check[array_size1*4+host_a[i]]^= host_b[i];
// 		for (int i = 0; i < array_size0; i++)
// 		    host_c_check[array_size1*5+host_a[i]]=
// 			MAX(host_c_check[array_size1*5+host_a[i]],host_b[i]);
// 		for (int i = 0; i < array_size0; i++)
// 		    host_c_check[array_size1*6+host_a[i]]=
// 			MIN(host_c_check[array_size1*6+host_a[i]],host_b[i]);
// 		for (int i = 0; i < array_size0; i++)
// 		    host_c_check[array_size1*7+host_a[i]]+= 1;
// 		for (int i = 0; i < array_size0; i++)
// 		    host_c_check[array_size1*8+host_a[i]]-= 1;
//
// 		cudaMemcpy(dev_a, host_a, size0, cudaMemcpyHostToDevice);
// 		cudaMemcpy(dev_b, host_b, size0, cudaMemcpyHostToDevice);
// 		cudaMemcpy(dev_c, host_c, size1, cudaMemcpyHostToDevice);
//
// 		gridDim.x = array_size0/1024;
// 		gridDim.y = 1;
// 		gridDim.z = 1;
//
// 		blockDim.x = 1024;
// 		blockDim.y = 1;
// 		blockDim.z = 1;
//
// 		cudaConfigureCall(gridDim, blockDim);
//
// 		if (cudaSetupArgument(&dev_a, sizeof(void *), 0) != cudaSuccess)
// 		    throw("Cannot setup cuda argument A");
//
// 		if (cudaSetupArgument(&dev_b, sizeof(void *), 1 * sizeof(void *)) != cudaSuccess)
// 		    throw("Cannot setup cuda argument B");
//
// 		if (cudaSetupArgument(&dev_c, sizeof(void *), 2 * sizeof(void *)) != cudaSuccess)
// 		    throw("Cannot setup cuda argument C");
//
// 		if (cudaSetupArgument(&array_size0, sizeof(int), 3 * sizeof(void *)) != cudaSuccess)
// 		    throw("Cannot setup cuda argument N");
// 		if (cudaSetupArgument(&array_size1, sizeof(int), 3 * sizeof(void *) + sizeof(int)) != cudaSuccess)
// 		    throw("Cannot setup cuda argument N1");
//
// 		cudaLaunch(code_ptrs[code_id]);
//
// 		cudaGetLastError();
//
// 		cudaMemcpy(host_c, dev_c, size1, cudaMemcpyDeviceToHost);
//
// 		int err = 0;
// 		for (int i = 0; i < array_size1*n_ops; i++)
// 		    if(host_c[i]!=host_c_check[i])
// 		    {
// 			printf("Error at %d: Simulated %08x should be %08x\n",i,host_c[i],host_c_check[i]);
// 			err++;
// 		    }
// 		printf("%d errors out of %d entries\n",err,array_size0);
//
//
// 		cudaFree(dev_a);
// 		cudaFree(dev_b);
// 		cudaFree(dev_c);
//
// 		free(host_a);
// 		free(host_b);
// 		free(host_c);
// 	     }
// 	     catch (char *msg)
// 	     {
// 		printf("ERROR: %s\n",msg);
// 		if(dev_a) cudaFree(dev_a);
// 		if(dev_b) cudaFree(dev_b);
// 		if(dev_c) cudaFree(dev_c);
// 	     }
// 	  }
// 	else if (!sample_prog.compare("code5.txt")) {
//
//             int n_vtx = 2048;
//             int n_tri = 2048;
//
//             int n_vtx_grp = 256;
//             int n_tri_grp = 256;
//
// 	    int size_a = n_vtx*sizeof(float)*6;
// 	    int size_b = n_tri*sizeof(int)*3;
// 	    int size_c = n_tri*sizeof(float)*6*3;
//
//             float *host_a = (float *)malloc(size_a);
//             unsigned int *host_b = (unsigned int *)malloc(size_b);
//             float *host_c = (float *)malloc(size_c);
//             float *host_c_check = (float *)malloc(size_c);
//
//             unsigned int *dev_a = 0;
//             unsigned int *dev_b = 0;
//             unsigned int *dev_c = 0;
//
// 	    try
// 	    {
// 		if (!host_a || !host_b || !host_c)
// 		    throw("Cannot malloc buffers");
// 		if (cudaMalloc((void **)&dev_a, size_a) != cudaSuccess)
// 		    throw("Cannot cudaMalloc buffers");
// 		if (cudaMalloc((void **)&dev_b, size_b) != cudaSuccess)
// 		    throw("Cannot cudaMalloc buffers");
// 		if (cudaMalloc((void **)&dev_c, size_c) != cudaSuccess)
// 		    throw("Cannot cudaMalloc buffers");
//
// 		for (int i = 0; i < n_vtx*6; i++)
// 		    host_a[i] = (float)(random()&0xffff);
// 		for (int i = 0; i < n_tri*3; i++)
// 		    host_b[i] = (random()%n_vtx_grp);
// 		for (int i = 0; i < n_tri*6*3; i++)
// 		    host_c[i] = 0;
//
// 		cudaMemcpy(dev_a, host_a, size_a, cudaMemcpyHostToDevice);
// 		cudaMemcpy(dev_b, host_b, size_b, cudaMemcpyHostToDevice);
// 		cudaMemcpy(dev_c, host_c, size_c, cudaMemcpyHostToDevice);
//
// 		for(int i=0;i<n_tri*6*3;i++)
// 		{
// 		    int tt = i/6;
// 		    tt = host_b[tt];
// 		    tt += n_vtx_grp * (i/(n_tri_grp*6*3));
// 		    tt = tt*6+(i%6);
// 		    host_c_check[i] = host_a[tt];
// 		}
//
// 		gridDim.x = n_vtx/n_vtx_grp;
// 		gridDim.y = 1;
// 		gridDim.z = 1;
//
// 		blockDim.x = n_vtx_grp;
// 		blockDim.y = 1;
// 		blockDim.z = 1;
//
// 		cudaConfigureCall(gridDim, blockDim);
//
// 		if (cudaSetupArgument(&dev_a, sizeof(void *), 0) != cudaSuccess)
// 		    throw("Cannot setup cuda argument A");
//
// 		if (cudaSetupArgument(&dev_b, sizeof(void *), 1 * sizeof(void *)) != cudaSuccess)
// 		    throw("Cannot setup cuda argument B");
//
// 		if (cudaSetupArgument(&dev_c, sizeof(void *), 2 * sizeof(void *)) != cudaSuccess)
// 		    throw("Cannot setup cuda argument C");
//
// 		if (cudaSetupArgument(&n_vtx_grp, sizeof(int), 3 * sizeof(void *)) != cudaSuccess)
// 		    throw("Cannot setup cuda argument N_VTX");
// 		if (cudaSetupArgument(&n_tri_grp, sizeof(int), 3 * sizeof(void *) + sizeof(int)) != cudaSuccess)
// 		    throw("Cannot setup cuda argument N_TRI");
//
// 		cudaLaunch(code_ptrs[code_id]);
//
// 		cudaGetLastError();
//
// 		cudaMemcpy(host_c, dev_c, size_c, cudaMemcpyDeviceToHost);
//
// 		int err = 0;
//
//
// /*
// 		for(int i=0;i<n_vtx;i++)
// 		    printf("vtx %d = %f\n",i,host_a[i*6]);
// 		for(int i=0;i<n_tri;i++)
// 		    printf("tri %d = %d %d %d\n",i,host_b[i*3+0],host_b[i*3+1],host_b[i*3+2]);
// 		for(int i=0;i<n_tri;i++)
// 		    printf("tri %d = %f %f %f\n",i,host_c[i*18],host_c[i*18+6],host_c[i*18+12]);
// 		for(int i=0;i<n_tri;i++)
// 		    printf("tri %d = %f %f %f\n",i,host_c_check[i*18],host_c_check[i*18+6],host_c_check[i*18+12]);
// */
//
//
// 		for(int i=0;i<n_tri*18;i++)
// 		    if(host_c[i]!=host_c_check[i])
// 			err++;
// 		printf("%d errors\n",err);
//
// 		cudaFree(dev_a);
// 		cudaFree(dev_b);
// 		cudaFree(dev_c);
//
// 		free(host_a);
// 		free(host_b);
// 		free(host_c);
// 	     }
// 	     catch (char *msg)
// 	     {
// 		printf("ERROR: %s\n",msg);
// 		if(dev_a) cudaFree(dev_a);
// 		if(dev_b) cudaFree(dev_b);
// 		if(dev_c) cudaFree(dev_c);
// 	     }
// 	  }
//
//   __cudaUnregisterFatBinary(NULL);

  return 0;
}
