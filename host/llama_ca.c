/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <string.h>
#include <fcntl.h>
#include <assert.h>
#include <omp.h>
#include <sys/param.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif
#include <tee_client_api.h>
#include <llama_ta.h>
// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(TEEC_Context *ctx, TEEC_Session *sess, Tokenizer *tokenizer, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }
    if (num_prompt_tokens - 1 >= steps) {
        fprintf(stderr, "prompt is too long for the number of steps (%d >= %d)\n",
                num_prompt_tokens - 1, steps);
        exit(EXIT_FAILURE);
    }

    int* generated_tokens = (int*)malloc(steps * sizeof(int));
    TEEC_Operation op = {
        .params[0].tmpref.buffer = generated_tokens,
        .params[0].tmpref.size = steps * sizeof(int),
        .params[1].tmpref.buffer = prompt_tokens,
        .params[1].tmpref.size = num_prompt_tokens * sizeof(int),
        .paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_OUTPUT, TEEC_MEMREF_TEMP_INPUT,
                                       TEEC_NONE, TEEC_NONE)
    };
    uint32_t err_origin;
	TEEC_Result res = TEEC_InvokeCommand(sess, TA_LLAMA_CMD_GENERATE, &op, &err_origin);
	if (res != TEEC_SUCCESS) {
		fprintf(stderr, "TEEC_InvokeCommand failed with code 0x%x origin 0x%x\n", res, err_origin);
        exit(EXIT_FAILURE);
    }

    steps = op.params[0].tmpref.size / sizeof(int);
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    for (int pos = 0; pos < steps; ++pos) {
        int next = generated_tokens[pos];
        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;
    }
    printf("\n");

    free(generated_tokens);
    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

struct ModelBinaryHeader {
    uint32_t file_size;
    uint32_t nblock;
    uint32_t intervals[];
};

void create_storage(TEEC_Session *sess, char *model_id) {
    TEEC_Operation op = {
        .params[0].tmpref.buffer = model_id,
        .params[0].tmpref.size = strlen(model_id),
        .paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_NONE,
                                       TEEC_NONE, TEEC_NONE)
    };
    uint32_t err_origin;
    TEEC_Result res = TEEC_InvokeCommand(sess, TA_LLAMA_CMD_MODEL_STORAGE_CREATE, &op, &err_origin);
    if (res != TEEC_SUCCESS) {
        fprintf(stderr, "TA_LLAMA_CMD_MODEL_STORAGE_CREATE failed with code 0x%x origin 0x%x\n", res, err_origin);
        exit(EXIT_FAILURE);
    }
}

void batch_write_storage(TEEC_Context *ctx, TEEC_Session *sess, char *model_id, FILE *file) {
    // if SHM_MAX_SIZE equals to TEEC_CONFIG_SHAREDMEM_MAX_SIZE, TEEC_AllocateSharedMemory will return out
    // -of-memory when writing stories15M.bin model 
    const size_t SHM_MAX_SIZE = 0x40000;
    assert(SHM_MAX_SIZE <= TEEC_CONFIG_SHAREDMEM_MAX_SIZE);

    fseek(file, 0, SEEK_END); // move file pointer to end of file
    ssize_t file_size = ftell(file); // get the file size, in bytes
    if (file_size < 0) { fprintf(stderr, "ftell failed!\n"); exit(EXIT_FAILURE); }
    size_t remain_size = file_size;
    fseek(file, 0, SEEK_SET); // move file pointer to begin of file

    size_t batch_size = MIN(remain_size, SHM_MAX_SIZE);
    TEEC_SharedMemory shm;
    shm.size = batch_size;
    shm.flags = TEEC_MEM_INPUT;
    TEEC_Result res = TEEC_AllocateSharedMemory(ctx, &shm);
	if (res != TEEC_SUCCESS) {
		fprintf(stderr, "TEEC_AllocateSharedMemory failed with code 0x%x\n", res);
        exit(EXIT_FAILURE);
    }

    while (remain_size) {
        fread(shm.buffer, 1, batch_size, file);
        TEEC_Operation op = {
            .params[0].tmpref.buffer = model_id,
            .params[0].tmpref.size = strlen(model_id),
            .params[1].memref.parent = &shm,
            .params[1].memref.offset = 0,
            .params[1].memref.size = batch_size,
            .paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_MEMREF_WHOLE,
                                           TEEC_NONE, TEEC_NONE)
        };
        uint32_t err_origin;
        res = TEEC_InvokeCommand(sess, TA_LLAMA_CMD_MODEL_STORAGE_APPEND, &op, &err_origin);
        if (res != TEEC_SUCCESS) {
            fprintf(stderr, "TA_LLAMA_CMD_MODEL_STORAGE_APPEND failed with code 0x%x origin 0x%x\n", res, err_origin);
            exit(EXIT_FAILURE);
        }
        remain_size -= batch_size;
        batch_size = MIN(remain_size, SHM_MAX_SIZE);
    }

    TEEC_ReleaseSharedMemory(&shm);
}

void create_mem(TEEC_Session *sess, size_t file_size) {
    TEEC_Operation op = {
        .params[0].value.a = file_size,
        .paramTypes = TEEC_PARAM_TYPES(TEEC_VALUE_INPUT, TEEC_NONE,
                                       TEEC_NONE, TEEC_NONE)
    };
    uint32_t err_origin;
    TEEC_Result res = TEEC_InvokeCommand(sess, TA_LLAMA_CMD_MODEL_MEM_CREATE, &op, &err_origin);
    if (res != TEEC_SUCCESS) {
        fprintf(stderr, "TA_LLAMA_CMD_MODEL_MEM_CREATE failed with code 0x%x origin 0x%x\n", res, err_origin);
        exit(EXIT_FAILURE);
    }
}

void batch_transfer_mem(int tid, TEEC_Context *ctx, TEEC_Session *sess, struct ModelBinaryHeader *header, void *payload) {
    // if SHM_MAX_SIZE equals to TEEC_CONFIG_SHAREDMEM_MAX_SIZE, TEEC_AllocateSharedMemory will return out
    // -of-memory when writing stories15M.bin model 
    const size_t SHM_MAX_SIZE = 0x40000;
    assert(SHM_MAX_SIZE <= TEEC_CONFIG_SHAREDMEM_MAX_SIZE);

    const size_t TAG_SZ = 16;

    uint32_t cipher_begin = header->intervals[tid];
    uint32_t cipher_end = header->intervals[tid + 1] - TAG_SZ;
    const uint32_t plain_offset = cipher_begin - TAG_SZ * tid;
    const uint32_t cipher_sz = cipher_end - cipher_begin;
    void * const tag = payload + cipher_end;
    uint32_t dst_offset = plain_offset;

    size_t batch_size = MIN(cipher_end - cipher_begin, SHM_MAX_SIZE);
    TEEC_SharedMemory shm;
    shm.size = batch_size;
    shm.flags = TEEC_MEM_INPUT;
    TEEC_Result res = TEEC_AllocateSharedMemory(ctx, &shm);
    if (res != TEEC_SUCCESS) {
        fprintf(stderr, "TEEC_AllocateSharedMemory failed with code 0x%x\n", res);
        exit(EXIT_FAILURE);
    }

    while (cipher_begin < cipher_end) {
        memcpy(shm.buffer, payload + cipher_begin, batch_size);
        TEEC_Operation op = {
            .params[0].memref.parent = &shm,
            .params[0].memref.offset = 0,
            .params[0].memref.size = batch_size,
            .params[1].value.a = dst_offset,
            .paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_PARTIAL_INPUT, TEEC_VALUE_INPUT,
                                            TEEC_NONE, TEEC_NONE)
        };
        uint32_t err_origin;
        res = TEEC_InvokeCommand(sess, TA_LLAMA_CMD_MODEL_MEM_WRITE_AT, &op, &err_origin);
        if (res != TEEC_SUCCESS) {
            fprintf(stderr, "TA_LLAMA_CMD_MODEL_MEM_WRITE_AT failed with code 0x%x origin 0x%x\n", res, err_origin);
            exit(EXIT_FAILURE);
        }
        dst_offset += batch_size;
        cipher_begin += batch_size;
        batch_size = MIN(cipher_end - cipher_begin, SHM_MAX_SIZE);
    }

    TEEC_ReleaseSharedMemory(&shm);

    TEEC_Operation op = {
        .params[0].value.a = plain_offset,
        .params[0].value.b = cipher_sz,
        .params[1].tmpref.buffer = tag,
        .params[1].tmpref.size = TAG_SZ,
        .paramTypes = TEEC_PARAM_TYPES(TEEC_VALUE_INPUT, TEEC_MEMREF_TEMP_INPUT,
                                        TEEC_NONE, TEEC_NONE)
    };
    uint32_t err_origin;
    res = TEEC_InvokeCommand(sess, TA_LLAMA_CMD_DECRYPT, &op, &err_origin);
    if (res != TEEC_SUCCESS) {
        fprintf(stderr, "TA_LLAMA_CMD_DECRYPT failed with code 0x%x origin 0x%x\n", res, err_origin);
        exit(EXIT_FAILURE);
    }
}

void *mmap_checkpoint(char *checkpoint_path, ssize_t *file_size) {
    // query file size
    FILE *file = fopen(checkpoint_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint_path); exit(EXIT_FAILURE); }
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    if (*file_size < 0) { fprintf(stderr, "ftell failed!\n"); exit(EXIT_FAILURE); }
    fclose(file);

    // mmap file with file size
    int fd = open(checkpoint_path, O_RDONLY); // open in read only mode
    if (fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    void *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    close(fd);
    return data;
}

int init_generate_ctx(TEEC_Session *sess, SamplerConfig *config) {
    TEEC_Operation op = {
        .params[0].tmpref.buffer = config,
        .params[0].tmpref.size = sizeof(*config),
        .paramTypes = TEEC_PARAM_TYPES(TEEC_MEMREF_TEMP_INPUT, TEEC_VALUE_OUTPUT,
                                       TEEC_NONE, TEEC_NONE)
    };
    uint32_t err_origin;
    TEEC_Result res = TEEC_InvokeCommand(sess, TA_LLAMA_CMD_INIT_MODEL_WITH_MEM, &op, &err_origin);
    if (res != TEEC_SUCCESS) {
        fprintf(stderr, "TA_LLAMA_CMD_INIT_MODEL_WITH_MEM failed with code 0x%x origin 0x%x\n", res, err_origin);
        exit(EXIT_FAILURE);
    }
    return op.params[1].value.a;
}

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the SamplerConfig
    SamplerConfig sampler_config;
    build_sampler_config(&sampler_config, temperature, topp, rng_seed);

    // create TEE context
	TEEC_Context ctx;
    TEEC_Result res = TEEC_InitializeContext(NULL, &ctx);
	if (res != TEEC_SUCCESS) {
		fprintf(stderr, "TEEC_InitializeContext failed with code 0x%x\n", res);
        exit(EXIT_FAILURE);
    }

    ssize_t file_size;
    void *data = mmap_checkpoint(checkpoint_path, &file_size);
    struct ModelBinaryHeader *header = (struct ModelBinaryHeader*)data;
    #pragma omp parallel num_threads(header->nblock)
    {
        TEEC_Session sess;
        TEEC_UUID uuid = TA_LLAMA_UUID;
        uint32_t err_origin;
        res = TEEC_OpenSession(&ctx, &sess, &uuid, TEEC_LOGIN_PUBLIC, NULL, NULL, &err_origin);
        if (res != TEEC_SUCCESS) {
            fprintf(stderr, "TEEC_Opensession failed with code 0x%x origin 0x%x\n", res, err_origin);
            exit(EXIT_FAILURE);
        }

        #pragma omp single
        create_mem(&sess, header->file_size);

        // transfer file in batches
        int tid = omp_get_thread_num();
        void *payload = (char*)data + sizeof(struct ModelBinaryHeader) + (header->nblock + 1) * sizeof(uint32_t);
        batch_transfer_mem(tid, &ctx, &sess, header, payload);
        #pragma omp barrier

        #pragma omp single nowait
        {
            int vocab_size = init_generate_ctx(&sess, &sampler_config);

            // build the Tokenizer via the tokenizer .bin file
            Tokenizer tokenizer;
            build_tokenizer(&tokenizer, tokenizer_path, vocab_size);

            // run!
            generate(&ctx, &sess, &tokenizer, prompt, steps);

            // clean up handles
            free_tokenizer(&tokenizer);
        }

        TEEC_CloseSession(&sess);
    }

    munmap(data, file_size);

    // destroy TEE context
    TEEC_FinalizeContext(&ctx);

    return 0;
}
#endif
