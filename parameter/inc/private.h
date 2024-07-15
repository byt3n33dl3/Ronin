/*
 * libclamma - llama2 C library derived from llama2.c
 *
 * See https://github.com/karpathy/llama2.c for MIT-licensed original
 *
 * Changes Copyright (C) 2023 Andy Green <andy@warmcat.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#if !defined(__LLC_PRIVATE_H__)
#define __LLC_PRIVATE_H__

/* Inference for Llama-2 txf_t model in pure C */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#if defined _WIN32
#include "win.h"
#else
#include <unistd.h>
#include <sys/mman.h>
#endif

#include "clamma.h"

/*
 * The supported types of quantized model files
 */

enum {
	CLAMMA_MODEL_VERSION1_FLOAT,
	CLAMMA_MODEL_VERSION2_INT8_80,
};

typedef int8_t cq_t;

typedef struct {
	uint32_t	dim; /* model dimensions */
	uint32_t	hidden_dim; /* for ffn layers */
	uint32_t	n_layers; /* model layers */
	uint32_t	n_heads; /* number of query heads */
	uint32_t	n_kv_heads; /* number of key/value heads
				     * (can be < query heads because
				     * of multiquery) */
	uint32_t	vocab_size; /* vocabulary size */
	uint32_t	seq_len; /* max sequence length */

	uint32_t	group_size;
	uint8_t		shared_classifier;
	uint8_t		version;

} txf_config_t;

typedef struct {
	int8_t		*q;    /* quantized values */
	float		*s;    /* scaling factors */
} qt_t;

typedef struct {
	qt_t		*q_tokens; // (size, dim)
	float		*token_embedding_table;    // (vocab_size, dim)
	// weights for neur_rmsnorms
	float		*rms_att_weight; // (layer, dim) neur_rmsnorm weights
	float		*rms_ffn_weight; // (layer, dim)

	/* in float quant mode, we simply cast and use these as float * */

	qt_t		*wq; // (layer, dim, n_heads * head_size)
	qt_t		*wk; // (layer, dim, n_kv_heads * head_size)
	qt_t		*wv; // (layer, dim, n_kv_heads * head_size)
	qt_t		*wo; // (layer, n_heads * head_size, dim)
	// weights for ffn
	qt_t		*w1; // (layer, hidden_dim, dim)
	qt_t		*w2; // (layer, dim, hidden_dim)
	qt_t		*w3; // (layer, hidden_dim, dim)
	// (optional) classifier weights for the logits, on the last layer
	qt_t		*wcls;

	// weights for neur_matmuls. note dim == n_heads * head_size

	// final neur_rmsnorm
	float		*rms_final_weight; // (dim,)

} txf_weights_t;

/*
 * These are written during per-layer processing in the forward operation.
 * We will parallelize each layer's worth of operations into its own thread
 */

struct txf;

typedef struct {
	const struct txf	*t;

	float		*xb; // activation at current time stamp  (dim,)
	float		*xb2; // an additional buffer just for convenience (dim,)
	float		*hb; // buffer for hidden dimension in the ffn (hidden_dim,)
	float		*hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
	qt_t		xq; // quantized x (dim,)
	qt_t		hq; // quantized hb (hidden_dim,)
	float		*q; // query (dim,)
	float		*k; // key (dim,)
	float		*v; // value (dim,)
	float		*att; // buffer for scores/attention values (n_heads, seq_len)

#if defined(LIBCLAMMA_SMP)
	clamma_sem_t	sem_done;
	int		queued;
#endif
} txf_session_state_t;

typedef struct {
	// current wave of activations
	float		*x; // activation at current time stamp (dim,)
	// kv cache
	float		*key_cache;   // (layer, seq_len, dim)
	float		*value_cache; // (layer, seq_len, dim)
	float		*logits; // output logits

	unsigned int	count_sessions;

	/* the session-local part of the state */
	txf_session_state_t tss;
} txf_state_t;

typedef struct cwc {
	struct cwc	*next;
	uint64_t	offset;
	size_t		len;
	unsigned int	count;
} cwc_t;

typedef struct cwc_state {
	cwc_t		*cwc_head;
	int		cwc_created;
	uint64_t	cwc_fetched;
	uint64_t	cwc_touched;
	uint64_t	cwc_alloced;

#if defined(LIBCLAMMA_SMP)
	clamma_mutex_t mut_cwc;
#endif
} cwc_state_t;

typedef struct {
	float		prob;
	int		index;
} pidx_t; // struct used when sorting probabilities during top-p sampling

typedef struct txf_sampler {
	size_t		size;
	pidx_t		*probindex; // buffer used in top-p sampling
	float		temperature;
	float		topp;
	uint64_t	rng_state;
} txf_sampler_t;

typedef struct txf_session {
	const struct txf *t;
	struct txf_session *next;

	txf_state_t	s;
	txf_sampler_t   sampler;

	size_t		pos;
	size_t		limit;
	size_t		ct;
	tok_id_t	token;
	tok_id_t	tnext;
	tok_id_t	*tokens;
	uint64_t	token_count;
	uint64_t	start;

	issue_cb_t	issue_cb;
	void		*opaque_user_pointer;
	void		**null_on_destroy;
	char		client_gone;
} txf_session_t;

typedef struct tidx {
	char		*str;
	tok_id_t	id;
} tidx_t;

typedef struct vocab {
	char		**vocab;
	float		*scores;
	tidx_t		*sorted_vocab;
	size_t		size;
	size_t		storage_size;
	uint32_t	max_token_length;
	char		utf8[16]; /* for <0xAB[CD]> format conversion */
} txf_vocab_t;

typedef struct txf {
	txf_config_t	c;
	txf_weights_t	w;
	txf_vocab_t	v;

	clamma_model_access_t model_access;
	clamma_model_type_t model_type;
	void		*model_base;
	size_t		model_size;
	size_t		cache_limit;

	unsigned int	max_sessions;
	char		name[33];
	struct txf	*next;

	int		fd;
	float		*data;
	unsigned int	d_ofs;
	ssize_t		file_size;
} txf_t;

int
_session_matmul(txf_session_state_t *tss,    float *xout, const float *x,
		const float *w1, int i, int dlim, int n, int d);

int
_session_matmul_qt(txf_session_state_t *tss, float *xout, const qt_t *x,
		   const qt_t *w1, int i, int dlim, int n, int d);

#if defined(LIBCLAMMA_SMP)

typedef enum {
	CLAMMA_JOB_MATMUL,
	CLAMMA_JOB_MATMUL_QT
} clamma_job_type_t;

typedef struct job {
	txf_session_state_t	*tss;
	clamma_job_type_t	type;

	float			*xout;
	const float		*x;
	const float		*w1;
	const qt_t		*qt_x;
	const qt_t		*qt_w;
	int			i;
	int			n;
	int			d;
	int			dlim;
} job_t;

typedef struct work {
	clamma_mutex_t mut_job;

	job_t		job_ring[LIBCLAMMA_MAX_THREAD_JOB_QUEUE];

	int		job_head;
	int		job_tail;
} work_t;

typedef struct work_threads {
	pthread_t	pt;
	clamma_sem_t	sem_start;
	char		running;
	char		exiting;
} work_threads_t;


extern work_threads_t *work_threads;
extern work_t work;
extern unsigned int count_threads, thread_init_refcount;
extern clamma_mutex_t          mut_sessions;

void
clamma_smp_deinit(void);

int
clamma_smp_init(unsigned int threads);

int
session_matmul(txf_session_state_t *tss, float *xout, const float *x,
	       const float *w1, int n, int d);

int
session_matmul_qt(txf_session_state_t *tss, float *xout, const qt_t *x, const qt_t *w,
		int n, int d);

void
clamma_smp_sync_point(txf_session_state_t *tss);

int
clamma_smp_tss_init(txf_session_state_t *tss);

void
clamma_smp_tss_deinit(txf_session_state_t *tss);

void *
clamma_session_worker(void *tp);

#else
static inline int
session_matmul(txf_session_state_t *tss, float *xout, const float *x, const float *w1,
		int n, int d)
{
	return _session_matmul(tss, xout, x, w1, 0, d, n, d);
}

static inline int
session_matmul_qt(txf_session_state_t *tss, float *xout, const qt_t *x, const qt_t *w,
		  int n, int d)
{
	return _session_matmul_qt(tss, xout, x, w, 0, d, n, d);
}

static inline void
clamma_smp_sync_point(txf_session_state_t *tss)
{
	(void)tss;
	do { } while(0);
}

static inline int
clamma_smp_init(unsigned int threads)
{
	(void)threads;
	do { } while(0);
	return 0;
}

static inline void
clamma_smp_deinit(void)
{
	do { } while(0);
}

static inline int
clamma_smp_tss_init(txf_session_state_t *tss)
{
	(void)tss;
	do { } while(0);
	return 0;
}

static inline void
clamma_smp_tss_deinit(txf_session_state_t *tss)
{
	(void)tss;
	do { } while(0);
}
#endif

void
session_softmax(float *x, int size);

const void *
clamma_weight_cache(const txf_t *t, const void *weight, size_t size);

void
clamma_weight_cache_clear(void);

int
clamma_sampler_sample(txf_sampler_t *sampler, float *logits);

int
clamma_vocab_construct(struct txf *t, const char *tokenizer_path);

void
clamma_vocab_destroy(struct txf *t);

int
clamma_session_issue(const struct txf_session *t, const char *piece);

tok_id_t *
clamma_vocab_encode(const struct txf *t, const char *text, int8_t bos, int8_t eos,
		    size_t *n_tokens);

const char *
clamma_vocab_decode(const struct txf *t, int prev_token, int token);

tok_id_t
clamma_session_forward(txf_session_t *ts, int is_prompt, int token, int pos);

uint64_t
clamma_timestamp_ns(void);

#endif
