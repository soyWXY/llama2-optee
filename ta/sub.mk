global-incdirs-y += include
srcs-y += llama_ta.c

# ----------------------------------------------------------------------------
# openlibm settings

DEP_DIST_DIR := ../dist

# $CURDIR will be prepended to this path, so $DEP_DIST_DIR should be relative path
incdirs-llama_ta.c-y += $(DEP_DIST_DIR)/include/openlibm

# -fno-builtin: prevent GCC using builtin math implementation
# -D... : turn off OS-dependent features
cflags-llama_ta.c-y += -fno-builtin -D__BSD_VISIBLE=0 -D__XSI_VISIBLE=0

# Adds the static library <openlibm> to the list of the linker directive -l<openlibm>.
libnames += openlibm

# Adds the directory path to the libraries pathes list. Archive file
# lib<openlibm>.a is expected in this directory.
libdirs += $(DEP_DIST_DIR)/lib

# Adds the static library binary to the TA build dependencies.
libdeps += $(DEP_DIST_DIR)/lib/libopenlibm.a
