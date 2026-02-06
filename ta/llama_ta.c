#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <openlibm.h>

#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>
#include <pta_system.h>
#include <llama_ta.h>

// ----------------------------------------------------------------------------
// pta helper functions

static TEE_Result invoke_system_pta(uint32_t cmd_id, uint32_t param_types, TEE_Param params[TEE_NUM_PARAMS]) {
	static TEE_TASessionHandle sess = TEE_HANDLE_NULL;
	static const TEE_UUID uuid = PTA_SYSTEM_UUID;

	if (sess == TEE_HANDLE_NULL) {
		TEE_Result res = TEE_OpenTASession(&uuid, TEE_TIMEOUT_INFINITE,
						   0, NULL, &sess, NULL);

		if (res)
			return res;
	}

	return TEE_InvokeTACommand(sess, TEE_TIMEOUT_INFINITE, cmd_id,
				   param_types, params, NULL);
}

static void* system_alloc(size_t nbytes) {
    assert(nbytes % PTA_SYSTEM_PROTMEM_ALLOC_ALIGNMENT == 0);
    TEE_Param params[TEE_NUM_PARAMS];
	params[0].value.a = nbytes;
	params[0].value.b = 0;
	uint32_t param_types = TEE_PARAM_TYPES(
        TEE_PARAM_TYPE_VALUE_INPUT, TEE_PARAM_TYPE_VALUE_OUTPUT,
        TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE);
	TEE_Result res = invoke_system_pta(PTA_SYSTEM_PROTMEM_ALLOC, param_types, params);
	if (res != TEE_SUCCESS) {
		EMSG("Failed to allocate protmem: 0x%x", res);
		return NULL;
	}
    return (void*)reg_pair_to_64(params[1].value.a, params[1].value.b);
}

static void system_free(void *va, size_t nbytes) {
    // scrub the memory before freeing
    memset(va, 0, nbytes);

    TEE_Param params[TEE_NUM_PARAMS];
    reg_pair_from_64((uint64_t)va, &params[0].value.a, &params[0].value.b);
    uint32_t param_types = TEE_PARAM_TYPES(
        TEE_PARAM_TYPE_VALUE_INPUT, TEE_PARAM_TYPE_NONE,
        TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE);
	TEE_Result res = invoke_system_pta(PTA_SYSTEM_PROTMEM_FREE, param_types, params);
	if (res != TEE_SUCCESS) {
		EMSG("Failed to free protmem: 0x%x", res);
	}
}

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
    // some more state needed to properly clean up the ephemeral memory
    void*  va;
    size_t nbytes;
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the ephemeral memory
    void*  data;
    size_t nbytes;
} Transformer;

static size_t get_run_state_nbytes(Config* p) {
    size_t ret = 0;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    ret += p->dim * sizeof(float);
    ret += p->dim * sizeof(float);
    ret += p->dim * sizeof(float);
    ret += p->hidden_dim * sizeof(float);
    ret += p->hidden_dim * sizeof(float);
    ret += p->dim * sizeof(float);
    ret += p->n_layers * p->seq_len * kv_dim * sizeof(float);
    ret += p->n_layers * p->seq_len * kv_dim * sizeof(float);
    ret += p->n_heads * p->seq_len * sizeof(float);
    ret += p->vocab_size * sizeof(float);
    return ret;
}

static void malloc_run_state(RunState* s, Config* p) {
    size_t nbytes = get_run_state_nbytes(p);
    nbytes = ROUNDUP(nbytes, PTA_SYSTEM_PROTMEM_ALLOC_ALIGNMENT);
    void* va = system_alloc(nbytes);
    s->va = va;
    s->nbytes = nbytes;
    memset(va, 0, nbytes);

    float *flat_buf = va;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = flat_buf; flat_buf += p->dim;
    s->xb = flat_buf; flat_buf += p->dim;
    s->xb2 = flat_buf; flat_buf += p->dim;
    s->hb = flat_buf; flat_buf += p->hidden_dim;
    s->hb2 = flat_buf; flat_buf += p->hidden_dim;
    s->q = flat_buf; flat_buf += p->dim;
    s->key_cache = flat_buf; flat_buf += p->n_layers * p->seq_len * kv_dim;
    s->value_cache = flat_buf; flat_buf += p->n_layers * p->seq_len * kv_dim;
    s->att = flat_buf; flat_buf += p->n_heads * p->seq_len;
    s->logits = flat_buf; flat_buf += p->vocab_size;
}

static void free_run_state(RunState* s) {
    system_free(s->va, s->nbytes);
}

static void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

static void read_checkpoint(Transformer *t, float* data, size_t data_sz) {
    Config* config = &t->config;
    t->data = data;
    t->nbytes = data_sz;
    // read in the config header
    memcpy(config, data, sizeof(Config));
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    float* weights_ptr = data + sizeof(Config)/sizeof(float);
    memory_map_weights(&t->weights, config, weights_ptr, shared_weights);
}

static void build_transformer(Transformer *t, void* data, size_t data_sz) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(t, data, data_sz);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

static void free_transformer(Transformer* t) {
    // free the TransformerWeights buffers
    system_free(t->data, t->nbytes);
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

static void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

static void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

static void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

static float* forward(Transformer* transformer, int token, int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim*sizeof(*x));

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention. iterate over all heads
        int h;
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

static int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

static int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

static int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

static int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

static void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

static void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

static unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
static float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

static int sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// generation loop

static TEE_Result generate(Transformer *transformer, Sampler *sampler,
                           uint32_t param_types, TEE_Param params[4]) {
    const uint32_t expected_pt = TEE_PARAM_TYPES(
        TEE_PARAM_TYPE_MEMREF_OUTPUT, TEE_PARAM_TYPE_MEMREF_INPUT,
        TEE_PARAM_TYPE_NONE, TEE_PARAM_TYPE_NONE);
    if (param_types != expected_pt) { return TEE_ERROR_BAD_PARAMETERS; }
    
    // unpack params
    int* generated_tokens = (int*)params[0].memref.buffer;
    int steps = params[0].memref.size / sizeof(int);
    if (steps > transformer->config.seq_len) steps = transformer->config.seq_len; // override to ~max length
    int* prompt_tokens = (int*)params[1].memref.buffer;
    int num_prompt_tokens = params[1].memref.size / sizeof(int);

    // start the main loop
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            generated_tokens[pos] = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            generated_tokens[pos] = sample(sampler, logits);
        }
        token = generated_tokens[pos];
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (token == 1) { break; }
    }
    params[0].memref.size = pos * sizeof(int);

    return TEE_SUCCESS;
}

// ----------------------------------------------------------------------------
// TA

typedef struct {
    Transformer transformer;
    Sampler sampler;
} LlamaData;

// use @id and @bin to save model to secure storage, and return ephemeral buffer with @data_p and @data_sz_p
static TEE_Result create_secure_storage(TEE_Param id, TEE_Param bin, void **data_p, size_t *data_sz_p) {
    size_t obj_id_sz = id.memref.size;
    char *obj_id = malloc(obj_id_sz);
    if (!obj_id) return TEE_ERROR_OUT_OF_MEMORY;
    memcpy(obj_id, id.memref.buffer, obj_id_sz);

    size_t data_sz = ROUNDUP(bin.memref.size, PTA_SYSTEM_PROTMEM_ALLOC_ALIGNMENT);
    void *data = system_alloc(data_sz);
    TEE_Result res;
    if (!data) {
        res = TEE_ERROR_OUT_OF_MEMORY;
        goto free_obj;
    }
    memcpy(data, bin.memref.buffer, bin.memref.size);

	uint32_t obj_data_flag = TEE_DATA_FLAG_ACCESS_READ |	// we can later read the oject
			TEE_DATA_FLAG_ACCESS_WRITE |		// we can later write into the object
			TEE_DATA_FLAG_ACCESS_WRITE_META |	// we can later destroy or rename the object
			TEE_DATA_FLAG_OVERWRITE;		// destroy existing object of same ID
	TEE_ObjectHandle object = TEE_HANDLE_NULL;
	res = TEE_CreatePersistentObject(TEE_STORAGE_PRIVATE,
					obj_id, obj_id_sz,
					obj_data_flag,
					TEE_HANDLE_NULL,
					data, data_sz,
					&object);
	if (res != TEE_SUCCESS) {
		EMSG("TEE_CreatePersistentObject failed 0x%08x", res);
		goto free_data;
	}

    *data_p = data;
    *data_sz_p = data_sz;
    TEE_CloseObject(object);
    free(obj_id);
    return TEE_SUCCESS;
free_data:
    system_free(data, data_sz);
free_obj:
    free(obj_id);
    return res;
}

// use @id to read model from secure storage, and return ephemeral buffer with @data_p and @data_sz_p
static TEE_Result read_secure_storage(TEE_Param id, void **data_p, size_t *data_sz_p) {
	size_t obj_id_sz = id.memref.size;
    char *obj_id = malloc(obj_id_sz);
    if (!obj_id) return TEE_ERROR_OUT_OF_MEMORY;

    memcpy(obj_id, id.memref.buffer, obj_id_sz);

	// Check the object exist and can be dumped into output buffer
	// then dump it.
	TEE_ObjectHandle object = TEE_HANDLE_NULL;
	TEE_Result res = TEE_OpenPersistentObject(TEE_STORAGE_PRIVATE,
					obj_id, obj_id_sz,
					TEE_DATA_FLAG_ACCESS_READ |
					TEE_DATA_FLAG_SHARE_READ,
					&object);
	if (res != TEE_SUCCESS) {
		EMSG("Failed to open persistent object, res=0x%08x", res);
		goto free_id;
	}

	TEE_ObjectInfo object_info = { };
	res = TEE_GetObjectInfo1(object, &object_info);
	if (res != TEE_SUCCESS) {
		EMSG("Failed to get object info, res=0x%08x", res);
		goto close_obj;
	}

    size_t data_sz = ROUNDUP(object_info.dataSize, PTA_SYSTEM_PROTMEM_ALLOC_ALIGNMENT);
    char *data = system_alloc(data_sz);
    if (!data) {
        res = TEE_ERROR_OUT_OF_MEMORY;
        goto close_obj;
    }

	uint32_t read_bytes = 0;
	res = TEE_ReadObjectData(object, data, object_info.dataSize, &read_bytes);
	if (res != TEE_SUCCESS || read_bytes != object_info.dataSize) {
		EMSG("TEE_ReadObjectData failed 0x%08x, read %" PRIu32 " over %u",
				res, read_bytes, object_info.dataSize);
        system_free(data, data_sz);
		goto close_obj;
	}
    *data_p = data;
    *data_sz_p = data_sz;
close_obj:
	TEE_CloseObject(object);
free_id:    
    free(obj_id);
    return res;
}

TEE_Result TA_InvokeCommandEntryPoint(void *session, uint32_t cmd, uint32_t param_types, TEE_Param params[4]) {
    LlamaData *priv = (LlamaData *)session;
    switch (cmd) {
	case TA_LLAMA_CMD_GENERATE:
		return generate(&priv->transformer, &priv->sampler, param_types, params);
	default:
		EMSG("Command ID 0x%x is not supported", cmd);
		return TEE_ERROR_NOT_SUPPORTED;
	}
}

TEE_Result TA_CreateEntryPoint(void) {
	/* Nothing to do */
	return TEE_SUCCESS;
}

void TA_DestroyEntryPoint(void) {
	/* Nothing to do */
}

TEE_Result TA_OpenSessionEntryPoint(uint32_t param_types, TEE_Param params[4], void **session) {
    const uint32_t expected_pt = TEE_PARAM_TYPES(
        TEE_PARAM_TYPE_MEMREF_INPUT, TEE_PARAM_TYPE_MEMREF_INPUT,
        TEE_PARAM_TYPE_VALUE_OUTPUT, TEE_PARAM_TYPE_MEMREF_INPUT);
    if (param_types != expected_pt) { return TEE_ERROR_BAD_PARAMETERS; }
    if (params[1].memref.size != sizeof(SamplerConfig)) { return TEE_ERROR_BAD_PARAMETERS; }

    // start allocating session data
    LlamaData *priv = malloc(sizeof(LlamaData));
    if (priv == NULL) { return TEE_ERROR_OUT_OF_MEMORY; }
    *session = priv;

    // load model .bin file into @data and @data_sz
    TEE_Result res;
    void *data = NULL;
    size_t data_sz = 0;
    if (params[0].memref.size > 0) {
        res = create_secure_storage(params[3], params[0], &data, &data_sz);
    } else {
        res = read_secure_storage(params[3], &data, &data_sz);
    }
    if (res != TEE_SUCCESS) {
        free(priv);
        return res;
    }
    // build the Transformer via @data and @data_sz
    Transformer *transformer = &priv->transformer;
    build_transformer(transformer, data, data_sz);

    // build the Sampler
    SamplerConfig *config = (SamplerConfig *)params[1].memref.buffer;
    build_sampler(&priv->sampler, transformer->config.vocab_size, config->temperature, config->topp, config->rng_seed);

    params[2].value.a = transformer->config.vocab_size;
    return TEE_SUCCESS;
}

void TA_CloseSessionEntryPoint(void *session) {
    assert(session != NULL);
    LlamaData *priv = (LlamaData *)session;
    free_sampler(&priv->sampler);
    free_transformer(&priv->transformer);
    free(session);
    DMSG("LGTM!!!");
}