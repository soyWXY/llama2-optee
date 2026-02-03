# llama2-optee: Secure Inference of Llama 2 on OP-TEE

## Overview

This project is a port of [llama2.c](https://github.com/karpathy/llama2.c) to OP-TEE (Open Portable Trusted Execution Environment). It enables Llama 2 model inference within a Trusted Execution Environment (TEE), leveraging the isolation provided by ARM TrustZone technology.

## Prerequisites

* **OP-TEE Build Environment**: A working QEMUv8 build of OP-TEE.
    * Reference: [OP-TEE QEMUv8 Documentation](https://optee.readthedocs.io/en/latest/building/devices/qemu.html#qemu-v8)
    * Custom OP-TEE build instructions

    ```bash
    mkdir optee
    cd optee
    repo init -u https://github.com/soyWXY/manifest -b llama2 -m qemu_v8.xml
    repo sync -j$(nproc)
    cd build
    make toolchains -j$(nproc)
    make run -j$(nproc)
    ```

## Build Instructions

### 1. Configuration

Both the Client Application (CA) and Trusted Application (TA) must be compiled using the toolchain provided by OP-TEE. Set the `OPTEE_PATH` variable in the `Makefile` to your OP-TEE root directory.

```makefile
OPTEE_PATH ?= <PATH_TO_OPTEE>
```

### 2. Compilation

Execute `make` to compile the CA, TA, and dependencies (e.g., `openlibm`).

```bash
make
```

Note: the build process automatically copies the generated CA and TA binaries to the OP-TEE buildroot overlay directory, you may need to rebuild the rootfs image.

## Deployment & Execution (QEMU)

### 1. Prepare Host Environment

QEMU's VirtFS (9p) is used to share model files between the host and the QEMU guest.

Navigate to your OP-TEE build directory (usually `<PATH_TO_OPTEE>/build`) and launch QEMU with the following environment variables:

```bash
# Set the project folder containing model/tokenizer binaries
export QEMU_VIRTFS_ENABLE=y
export QEMU_USERNET_ENABLE=y
export QEMU_VIRTFS_HOST_DIR=<path_to_this_project_root>

# Launch QEMU
make run -j$(nproc)
```

### 2. Mount Host Directory in Guest

Once QEMU boots:

1. Type `c` and press `Enter` in the QEMU console to continue the boot process.
2. Proceed to the 'Normal World' console and log in as `root`.
3. Create a mount point and mount the host directory using 9p:

```bash
mkdir -p llama
mount -t 9p -o trans=virtio host llama
```

### 3. Execution

Run the Client Application (`llama_ca`) to initiate inference.

**Usage:**

```bash
llama_ca <model_path> -z <tokenizer_path>
```

**Example:**
Running the `stories260K` model:

```bash
# Ensure paths correspond to your mounted directory structure
llama_ca llama/model/stories260K/stories260K.bin -z llama/model/stories260K/tok512.bin
```
