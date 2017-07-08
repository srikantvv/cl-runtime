#
# Copyright © 1997,2000 Paul D. Smith
# Verbatim copying and distribution is permitted in any medium, provided this
# notice is preserved.
# See http://make.paulandlesley.org/multi-arch.html#advanced
#

TGTDIR = $(notdir $(CURDIR))

ifeq (64, $(findstring 64, $(TGTDIR)))
    BITS := 64
else ifeq (32, $(findstring 32, $(TGTDIR)))
    BITS := 32
else
    $(error Target directory name does not indicate BITS)
endif

ifeq (dbg, $(findstring dbg, $(TGTDIR)))
    MODE := dbg
else
    MODE :=
endif

VPATH = ..

HSAIL_GPU ?= ../../gem5/src/gpu-compute
GEM5_BASE ?= ../../gem5/src
RUNTIME_SRCS = cl_runtime.cc
HEADERS = cl_runtime.hh \
		$(HSAIL_GPU)/hsa_kernel_info.hh $(HSAIL_GPU)/qstruct.hh
CFLAGS = -D BUILD_CL_RUNTIME -msse3

ifeq ($(MODE), dbg)
    CFLAGS += -g -DDEBUG
else
    CFLAGS += -O3
endif

ifeq ($(BITS), 32)
    CFLAGS += -m32
endif

CXXFLAGS = $(CFLAGS) -std=c++11

CPPFLAGS = -I.. -I$(HSAIL_GPU) -I$(GEM5_BASE)

all: libOpenCL.a

RUNTIME_OBJS = $(RUNTIME_SRCS:.cc=.o)

$(RUNTIME_OBJS): $(HEADERS)

libOpenCL.a: $(RUNTIME_OBJS)
	ar rc libOpenCL.a cl_runtime.o

clean:
	rm -f libOpenCL.a cl_runtime.o hsa_test
