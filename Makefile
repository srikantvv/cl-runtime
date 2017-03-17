#
# Copyright © 1997,2000 Paul D. Smith
# Verbatim copying and distribution is permitted in any medium, provided this
# notice is preserved.
# See http://make.paulandlesley.org/multi-arch.html#advanced
#

TGTDIRS = 64b 32b dbg64b dbg32b

all: $(TGTDIRS)

NEWGOALS = $(filter-out $(TGTDIRS), $(MAKECMDGOALS))

.SUFFIXES:

MAKETARGET = $(MAKE) --no-print-directory -C $@ -f ../lib.mk $(NEWGOALS)

.PHONY: $(TGTDIRS)
$(TGTDIRS):
	+@echo "Building in $@"
	+@[ -d $@ ] || mkdir -p $@
	+@$(MAKETARGET)

Makefile : ;
%.mk :: ;

% :: $(TGTDIRS) ;

.PHONY: clean
clean:
	rm -rf $(TGTDIRS)
