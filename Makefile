export V ?= 0

# user-defined necessary variables
OPTEE_PATH ?= $(HOME)/optee
CROSS_COMPILE = $(OPTEE_PATH)/toolchains/aarch64/bin/aarch64-linux-gnu-
TA_OUT = out
BR_ROOT_PATH = $(OPTEE_PATH)/out-br/target
HOST_FLAGS = TEEC_EXPORT=$(OPTEE_PATH)/out-br/per-package/optee_client_ext/target/usr
TA_FLAGS = TA_DEV_KIT_DIR=$(OPTEE_PATH)/optee_os/out/arm/export-ta_arm64 O=$(TA_OUT)
DEP = $(wildcard dep/*)

# If _HOST or _TA specific compilers are not specified, then use CROSS_COMPILE
HOST_CROSS_COMPILE ?= $(CROSS_COMPILE)
TA_CROSS_COMPILE ?= $(CROSS_COMPILE)

.PHONY: all
all: $(DEP) build install

.PHONY: build
build:
	$(MAKE) -C host CROSS_COMPILE="$(HOST_CROSS_COMPILE)" --no-builtin-variables $(HOST_FLAGS)
	$(MAKE) -C ta CROSS_COMPILE="$(TA_CROSS_COMPILE)" LDFLAGS="" $(TA_FLAGS)

.PHONY: clean
clean:
	$(MAKE) -C host clean
	$(MAKE) -C ta clean $(TA_FLAGS)
	
# user-defined rules
.PHONY: dep/openlibm
dep/openlibm:
	$(MAKE) -C $@ ARCH="aarch64" TRIPLE="aarch64-linux-gnu" TOOLPREFIX="$(CROSS_COMPILE)" -j
	$(MAKE) -C $@ install-static install-headers DESTDIR="$(CURDIR)/dist" prefix=""

.PHONY: install
install:
	$(MAKE) -C host install INSTALL_DIR=$(BR_ROOT_PATH)/usr/bin
	$(MAKE) -C ta install O=$(TA_OUT) INSTALL_DIR=$(BR_ROOT_PATH)/lib/optee_armtz
	
.PHONY: uninstall
uninstall:
	-$(MAKE) -C host uninstall INSTALL_DIR=$(BR_ROOT_PATH)/usr/bin
	-$(MAKE) -C ta uninstall INSTALL_DIR=$(BR_ROOT_PATH)/lib/optee_armtz
