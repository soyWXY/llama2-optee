#pragma once

#define TA_LLAMA_UUID \
    { 0xaed564f1, 0x960e, 0x4117, \
        { 0x8b, 0xb4, 0x41, 0x7b, 0x4e, 0x79, 0xc1, 0x30} }

typedef struct {
    float temperature;
    float topp;
    unsigned long long rng_seed;
} SamplerConfig;

static void __attribute__((unused)) build_sampler_config(SamplerConfig* config, float temperature,
                                                         float topp, unsigned long long rng_seed) {
    config->temperature = temperature;
    config->topp = topp;
    config->rng_seed = rng_seed;
}

/*
 * TA_LLAMA_CMD_GENERATE - Generate tokens
 * param[0] (memref) out: generated tokens
 * param[1] (memref) in: prompt tokens
 * param[2] unused
 * param[3] unused
 */
#define TA_LLAMA_CMD_GENERATE 0

/*
 * TA_LLAMA_CMD_MODEL_STORAGE_CREATE - Create storage object
 * param[0] (memref) in: secure storage id
 * param[1] unused
 * param[2] unused
 * param[3] unused
 */
#define TA_LLAMA_CMD_MODEL_STORAGE_CREATE 1

/*
 * TA_LLAMA_CMD_MODEL_STORAGE_APPEND - append to storage object
 * param[0] (memref) in: secure storage id
 * param[1] (memref) in: data to append to object
 * param[2] unused
 * param[3] unused
 */
#define TA_LLAMA_CMD_MODEL_STORAGE_APPEND 2

/*
 * TA_LLAMA_CMD_INIT_MODEL_WITH_STORAGE - read model from storage
 * param[0] (memref) in: secure storage id
 * param[1] (memref) in: SamplerConfig
 * param[2] (value.a) out: return transformer.config.vocab_size
 * param[3] unused
 */
#define TA_LLAMA_CMD_INIT_MODEL_WITH_STORAGE 3

/*
 * TA_LLAMA_CMD_MODEL_MEM_CREATE - allocate memory buffer
 * param[0] (value.a) in: required buffer size
 * param[1] unused
 * param[2] unused
 * param[3] unused
 */
#define TA_LLAMA_CMD_MODEL_MEM_CREATE 4

/*
 * TA_LLAMA_CMD_INIT_MODEL_WITH_MEM - read model from memory buffer
 * param[0] (memref) in: SamplerConfig
 * param[1] (value.a) out: return transformer.config.vocab_size
 * param[2] unused
 * param[3] unused
 */
#define TA_LLAMA_CMD_INIT_MODEL_WITH_MEM 6

/*
 * TA_LLAMA_CMD_MODEL_MEM_WRITE_AT - write to memory buffer allocated by TA_LLAMA_CMD_MODEL_MEM_CREATE
 * param[0] (memref) in: data to append to object
 * param[1] (value.a) in: memory offset to write at
 * param[2] unused
 * param[3] unused
 */
#define TA_LLAMA_CMD_MODEL_MEM_WRITE_AT 7

/*
 * TA_LLAMA_CMD_DECRYPT - decrypt designated region of memory
 * param[0] (value.a) in: memory offset of the region
 *          (value.b) in: size of the region
 * param[1] (memref) in: tag of the ciphertext
 * param[2] unused
 * param[3] unused
 */
#define TA_LLAMA_CMD_DECRYPT 8