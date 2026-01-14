CFG_TEE_TA_LOG_LEVEL ?= 4
CFG_TA_OPTEE_CORE_API_COMPAT_1_1=y

# The UUID for the Trusted Application
BINARY=aed564f1-960e-4117-8bb4-417b4e79c130

-include $(TA_DEV_KIT_DIR)/mk/ta_dev_kit.mk

ifeq ($(wildcard $(TA_DEV_KIT_DIR)/mk/ta_dev_kit.mk), )
clean:
	@echo 'Note: $$(TA_DEV_KIT_DIR)/mk/ta_dev_kit.mk not found, cannot clean TA'
	@echo 'Note: TA_DEV_KIT_DIR=$(TA_DEV_KIT_DIR)'
endif

# user-defined rules
.PHONY: install
install:
	cp $(O)/$(BINARY).ta $(INSTALL_DIR)
	
.PHONY: uninstall
uninstall:
	rm $(INSTALL_DIR)/$(BINARY).ta