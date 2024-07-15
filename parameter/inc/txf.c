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

#include "private.h"

static txf_t		*txf_head;
static txf_session_t	*sess_head;
#if defined(LIBCLAMMA_SMP)
clamma_mutex_t          mut_sessions;
#endif

static void
dequantize(txf_t *t, qt_t *qx, float *x, int n)
{
	const cq_t *w_q = clamma_weight_cache(t, qx->q, n);
	const float *w_s = clamma_weight_cache(t, qx->s,
				(n / t->c.group_size) * sizeof(*w_s));

	for (int i = 0; i < n; i++)
		x[i] = (float)w_q[i] * w_s[i / t->c.group_size];
}

static qt_t *
init_quantized_tensors(txf_t *t, void **ptr, int n, int size_each)
{
	qt_t *res = malloc(n * sizeof(qt_t));
	void *p = *ptr;

	if (!res)
		return NULL;

	for (int i = 0; i < n; i++) {
		res[i].q = (cq_t *)p;
		p = (cq_t *)p + size_each;
		res[i].s = (float *)p;
		p = (float *)p + size_each / t->c.group_size;
	}

	*ptr = p;

	return res;
}

static int
def_iss_cb(void *opaque_user_pointer, const char *piece)
{
	(void)opaque_user_pointer;

	fprintf(stdout, "%s", piece);
	fflush(stdout);

	return 0;
}

txf_t *
clamma_txf_by_name(const char *name)
{
	txf_t *p = txf_head;

	do {
		if (!strcmp(p->name, name))
			return p;

		p = p->next;
	} while (p);

	return NULL;
}

size_t
clamma_txf_session_size(const txf_t *t)
{
	size_t kvd  = (t->c.dim * t->c.n_kv_heads) / t->c.n_heads;
	size_t size = (((t->c.dim       * 2) +
		(t->c.vocab_size) +
		(t->c.n_layers   * t->c.seq_len * kvd * 2) +
		(t->c.n_layers    * t->c.seq_len)) * sizeof(txi_t));

	switch (t->c.version) {
	case CLAMMA_MODEL_VERSION2_INT8_80:
		size += sizeof(txi_t) * (t->c.vocab_size +
					 t->c.dim + t->c.dim +
					 t->c.hidden_dim);
		break;
	}

	size += 1 * sizeof(float) *
		((t->c.dim * 5) + (t->c.hidden_dim * 4) +
		 (t->c.n_layers * t->c.seq_len)) +
		 (1 * (t->c.dim + t->c.hidden_dim));

	return size;
}

txf_t *
clamma_txf_construct(const clamma_txf_info_t *info)
{
	static const char *access_name[] = { "MMAP", "AllocCache", "Address" };
	int head_size, threads = info->threads ? info->threads : 8;
	char desc[256], thr[64];
	uint32_t *p32 = NULL;
	uint64_t n_layers;
	uint8_t buf[256];
	size_t size;
	txf_t *t;
	void *wp;

	if (info->clamma_api_version != CLAMMA_API_VERSION) {
		fprintf(stderr, "%s: clamma_api_version mismatch\n", __func__);
		return NULL;
	}

	/*
	 * Create and initialize the transformer object
	 */

	t = malloc(sizeof(*t));
	if (!t)
		return NULL;

	memset(t, 0, sizeof(*t));

	clamma_smp_init(threads);

	if (!info->checkpoint_path)
		return t;

	t->model_access = info->model_access;
	t->model_base   = info->model_base;
	t->model_size   = info->model_size;
	t->cache_limit  = info->cache_limit;
	t->model_type   = info->model_type;
	t->max_sessions = info->max_sessions;

	strncpy(t->name, info->name, sizeof(t->name));
	t->name[sizeof(t->name) - 1] = '\0';

	switch (t->model_access) {
	case CLAMMA_MODEL_ACCESS_MMAP:
	case CLAMMA_MODEL_ACCESS_MALLOC_CACHE:
		t->fd = open(info->checkpoint_path, O_RDONLY);
		if (t->fd < 0) {
			snprintf(desc, sizeof(desc) - 1, "%s/%s",
					CLAMMA_MODEL_SEARCH_PATH,
					info->checkpoint_path);
			t->fd = open(desc, O_RDONLY);
			if (t->fd < 0) {
				fprintf(stderr, "Couldn't open file %s\n",
						info->checkpoint_path);
				goto bail;
			}
		}

		t->file_size = lseek(t->fd, 0, SEEK_END);
		lseek(t->fd, 0, SEEK_SET);
		break;
	default:
		break;
	}

	switch (t->model_access) {
	case CLAMMA_MODEL_ACCESS_MMAP:
		t->data = mmap(NULL, t->file_size, PROT_READ, MAP_PRIVATE,
			       t->fd, 0);
		if (t->data == MAP_FAILED) {
			fprintf(stderr, "MMAP failed %s\n",
					info->checkpoint_path);
			goto bail1;
		}
		/* fallthru */
	case CLAMMA_MODEL_ACCESS_MALLOC_CACHE:
		if (read(t->fd, buf, sizeof(buf)) != sizeof(buf)) {
			fprintf(stderr, "Unable to read header from %s: %d\n",
					info->checkpoint_path, errno);
			goto bail1;
		}
		p32 = (uint32_t *)buf;
		break;
	case CLAMMA_MODEL_ACCESS_ABSOLUTE_ADDRESS:
		t->data = t->model_base;
		t->file_size = t->model_size;
		p32 = (uint32_t *)t->data;
		memcpy(buf, p32, sizeof(buf));
		break;
	}

	if (p32[0] == 0x616b3432 && p32[1] == 2) {
		uint8_t *p = (uint8_t *)&p32[9];

		memcpy(&t->c, &p32[2], sizeof(uint32_t) * 7);
		t->c.version = CLAMMA_MODEL_VERSION2_INT8_80;
		t->d_ofs = 256;
		t->c.shared_classifier = *p++;
		t->c.group_size = *p++;
		t->c.group_size |= (*p++) << 8;
		t->c.group_size |= (*p++) << 16;
		t->c.group_size |= (*p++) << 24;
	} else {
		t->c.version = CLAMMA_MODEL_VERSION1_FLOAT;
		t->d_ofs = 7 * sizeof(uint32_t);
		memcpy(&t->c, buf, t->d_ofs);

		t->c.shared_classifier = 1;
		if (((ssize_t)(int32_t)t->c.vocab_size) < (ssize_t)0) {
			t->c.shared_classifier = 0;
			t->c.vocab_size = -(ssize_t)t->c.vocab_size;
		}
	}

	head_size = t->c.dim / t->c.n_heads;
	n_layers = t->c.n_layers;

	if (clamma_vocab_construct(t, info->tokenizer_path))
		goto bail2;

#if defined(LIBCLAMMA_SMP)
	snprintf(thr, sizeof(thr) - 1, "%u x ", threads);
#else
	thr[0] = '\0';
#endif

	size = clamma_txf_session_size(t);
	snprintf(desc, sizeof(desc) - 1,
		       "☙ Clamma ❧  %s%s, model: %s (%uMB) %s %s, "
			"vocab: %u (%uKB),\n"
		       "             Session: %llu.%03lluMB, d: %u, hd: %u, "
			"l: %u, h: %d, kvh: %d, seq_len: %d",
		       thr, LIBCLAMMA_THREAD_MODEL, info->checkpoint_path,
		       (unsigned int)(t->file_size / (1024 * 1024)),
		       t->c.version ? "int8" : "float",
		       access_name[t->model_access], t->c.vocab_size,
		       (int)(t->v.storage_size / 1024),
		       ((unsigned long long)size) / (1024 * 1024),
		       	(((unsigned long long)size) % (1024 * 1024)) / 1000,
		       t->c.dim, t->c.hidden_dim, t->c.n_layers, t->c.n_heads,
		       t->c.n_kv_heads, t->c.seq_len);

	if (info->desc && info->desc_max) {
		strncpy(info->desc, desc, info->desc_max);
		info->desc[info->desc_max - 1] = '\0';
	}

	fprintf(stderr, "%s\n", desc);
	fflush(stderr);

	/*
	 * Layout the structure of the model file
	 */

	switch (t->c.version) {
	case CLAMMA_MODEL_VERSION1_FLOAT:

		t->w.token_embedding_table = (float *)
				((uint8_t *)t->data + t->d_ofs);

		t->w.rms_att_weight = t->w.token_embedding_table +
						(t->c.vocab_size * t->c.dim);

		t->w.wq = (qt_t *)((float *)t->w.rms_att_weight +
						(n_layers * t->c.dim));
		t->w.wk = (qt_t *)((float *)t->w.wq +
			(n_layers * t->c.dim * (t->c.n_heads * head_size)));
		t->w.wv = (qt_t *)((float *)t->w.wk +
			(n_layers * t->c.dim * (t->c.n_kv_heads * head_size)));
		t->w.wo = (qt_t *)((float *)t->w.wv +
			(n_layers * t->c.dim * (t->c.n_kv_heads * head_size)));
		t->w.rms_ffn_weight = (float *)((float *)t->w.wo +
			(n_layers * (t->c.n_heads * head_size) * t->c.dim));
		t->w.w1 = (qt_t *)((float *)t->w.rms_ffn_weight +
			(n_layers * t->c.dim));
		t->w.w2 = (qt_t *)((float *)t->w.w1 +
			(n_layers * t->c.dim * t->c.hidden_dim));
		t->w.w3 = (qt_t *)((float *)t->w.w2 +
			(n_layers * t->c.hidden_dim * t->c.dim));

		t->w.rms_final_weight = (float *)((float *)t->w.w3 +
				(n_layers * t->c.dim * t->c.hidden_dim));
		wp = t->w.rms_final_weight + t->c.dim;
		/* skip what used to be freq_cis_real / ..._imag (for RoPE) */
		wp = (float *)wp + t->c.seq_len * head_size / 2;
		wp = (float *)wp + t->c.seq_len * head_size / 2;

		t->w.wcls = (qt_t *)(t->c.shared_classifier ?
					t->w.token_embedding_table : wp);
		break;

	case CLAMMA_MODEL_VERSION2_INT8_80:

		t->w.rms_att_weight = (float *)((uint8_t *)t->data + t->d_ofs);
		t->w.rms_ffn_weight = t->w.rms_att_weight +
				(t->c.n_layers * t->c.dim);
		t->w.rms_final_weight = t->w.rms_ffn_weight +
				(t->c.n_layers * t->c.dim);

		wp = t->w.rms_final_weight + t->c.dim;

		/* now read all the quantized weights */

		t->w.q_tokens = init_quantized_tensors(t, &wp, 1,
						t->c.dim * t->c.vocab_size);
		if (!t->w.q_tokens)
			goto bail2a;

		/* dequantize token embedding table */

		t->w.token_embedding_table = malloc(t->c.vocab_size *
						t->c.dim * sizeof(float));
		if (!t->w.token_embedding_table)
			goto bail3;

		dequantize(t, t->w.q_tokens, t->w.token_embedding_table,
				t->c.vocab_size * t->c.dim);

		t->w.wq = init_quantized_tensors(t, &wp, t->c.n_layers,
				t->c.dim * (t->c.n_heads * head_size));
		if (!t->w.wq)
			goto bail4;
		t->w.wk = init_quantized_tensors(t, &wp, t->c.n_layers,
				t->c.dim * (t->c.n_kv_heads * head_size));
		if (!t->w.wk)
			goto bail5;
		t->w.wv = init_quantized_tensors(t, &wp, t->c.n_layers,
				t->c.dim * (t->c.n_kv_heads * head_size));
		if (!t->w.wv)
			goto bail6;
		t->w.wo = init_quantized_tensors(t, &wp, t->c.n_layers,
				(t->c.n_heads * head_size) * t->c.dim);
		if (!t->w.wo)
			goto bail7;

		t->w.w1 = init_quantized_tensors(t, &wp, t->c.n_layers,
				t->c.dim * t->c.hidden_dim);
		if (!t->w.w1)
			goto bail8;
		t->w.w2 = init_quantized_tensors(t, &wp, t->c.n_layers,
				t->c.hidden_dim * t->c.dim);
		if (!t->w.w2)
			goto bail9;
		t->w.w3 = init_quantized_tensors(t, &wp, t->c.n_layers,
				t->c.dim * t->c.hidden_dim);
		if (!t->w.w3)
			goto bail10;

		t->w.wcls = t->c.shared_classifier ? t->w.q_tokens :
				init_quantized_tensors(t, &wp, 1,
						t->c.dim * t->c.vocab_size);
		if (!t->w.wcls)
			goto bail11;

		break;

	default:
		fprintf(stderr, "Unknown checkpoint version\n");
		goto bail2a;
	}

	return t;

bail11:
	free(t->w.w3);
bail10:
	free(t->w.w2);
bail9:
	free(t->w.w1);
bail8:
	free(t->w.wo);
bail7:
	free(t->w.wv);
bail6:
	free(t->w.wk);
bail5:
	free(t->w.wq);
bail4:
	free(t->w.token_embedding_table);
bail3:
	free(t->w.q_tokens);
bail2a:
	clamma_vocab_destroy(t);
bail2:
	switch (t->model_access) {
	case CLAMMA_MODEL_ACCESS_MMAP:
		if (t->data != MAP_FAILED)
			munmap(t->data, t->file_size);
		break;
	default:
		break;
	}
bail1:
	close(t->fd);
bail:
	clamma_smp_deinit();
	free(t);

	return NULL;
}

void
clamma_txf_destroy(txf_t *t)
{
	clamma_smp_deinit();

	switch (t->model_access) {
	case CLAMMA_MODEL_ACCESS_MMAP:
		if (t->data != MAP_FAILED)
			munmap(t->data, t->file_size);
		/* fallthru */
	case CLAMMA_MODEL_ACCESS_MALLOC_CACHE:
		if (t->fd != -1)
			close(t->fd);
		break;
	default:
		break;
	}

	switch (t->model_access) {
	case CLAMMA_MODEL_ACCESS_MALLOC_CACHE:
		clamma_weight_cache_clear();
		break;
	default:
		break;
	}

	clamma_vocab_destroy(t);

	free(t);
}

uint64_t
clamma_timestamp_ns(void)
{
	struct timespec time;

	clock_gettime(CLOCK_REALTIME, &time);

	return (time.tv_sec * 1000000000ull) + time.tv_nsec;
}

txf_session_t *
clamma_session_construct(const txf_t *t)
{
	txf_session_state_t *tss;
	unsigned int count_sessions = 0;
	txf_session_t *ts;
	size_t kvd, size;
	float *fp;

	/* limit sessions on this txf to its maximum, if any */

	if (t->max_sessions) {
#if defined(LIBCLAMMA_SMP)
		clamma_mutex_lock(&mut_sessions);
#endif
		ts = sess_head;
		while (ts) {
			if (ts->t == t)
				count_sessions++;
			ts = ts->next;
		}
#if defined(LIBCLAMMA_SMP)
		clamma_mutex_unlock(&mut_sessions);
#endif

		if (count_sessions >= t->max_sessions) {
			fprintf(stderr, "%s: reached max sessions %u\n",
					__func__, count_sessions);

			return NULL;
		}
	}

	ts = malloc(sizeof(*ts));
	if (!t)
		return NULL;

	memset(ts, 0, sizeof(*ts));

	ts->t = t;

	// buffer only used with nucleus sampling; may not need but it's ~small
	ts->sampler.probindex   = malloc(t->c.vocab_size * sizeof(pidx_t));
	if (!ts->sampler.probindex)
		goto bail1;

	kvd  = (t->c.dim * t->c.n_kv_heads) / t->c.n_heads;
	size = clamma_txf_session_size(t);

	ts->s.x = malloc(size);
	if (!ts->s.x)
		goto bail2;

	memset(ts->s.x, 0, size);

	fp = ts->s.x + t->c.dim;
	ts->s.key_cache   = fp;
	fp += t->c.n_layers * t->c.seq_len * kvd;
	ts->s.value_cache = fp;
	fp += t->c.n_layers * t->c.seq_len * kvd;
	ts->s.logits      = fp;
	tss = &ts->s.tss;

	tss->t = t;
	if (clamma_smp_tss_init(tss))
		goto bail3;

	tss->xb   = fp;
	fp += t->c.dim;
	tss->xb2  = fp;
	fp += t->c.dim;
	tss->hb   = fp;
	fp += t->c.hidden_dim;
	tss->hb2  = fp;
	fp += t->c.hidden_dim;
	tss->q    = fp;
	fp += t->c.dim;

	tss->xq.q = (cq_t *)fp;
	fp += t->c.dim / sizeof(txi_t);
	tss->xq.s = fp;
	fp += t->c.dim;
	tss->hq.q = (cq_t *)fp;
	fp += t->c.hidden_dim / sizeof(txi_t);
	tss->hq.s = fp;
	fp += t->c.hidden_dim / sizeof(txi_t);
	tss->att  = fp;
	fp += t->c.n_heads  * t->c.seq_len;

	clamma_mutex_lock(&mut_sessions);
	ts->next = sess_head;
	sess_head = ts;
	clamma_mutex_unlock(&mut_sessions);

	return ts;

bail3:
	free(ts->s.x);
bail2:
	free(ts->sampler.probindex);
bail1:
	free(ts);

	return NULL;
}

void
clamma_session_destroy(struct txf_session *ts)
{
	uint64_t ns;

	if (!ts)
		return;

	ns = (clamma_timestamp_ns() - ts->start) / 1000000l;

	fprintf(stderr, "\n%s: %p: Session: %lu tokens, tok/s: %4.03f\n",
			__func__, (void *)ts,
			(unsigned long)ts->token_count,
			(float)(ts->token_count * 1000ull) / (ns ? ns : 1));

	/* remove us from the list of sessions */

	clamma_mutex_lock(&mut_sessions);
	if (sess_head == ts)
		sess_head = ts->next;
	else {
		struct txf_session *ts1 = sess_head, *ts2 = NULL;
		while (ts1->next) {
			ts2 = ts1;
			ts1 = ts1->next;
		}

		if (ts2)
			ts2->next = ts1->next;
	}
	clamma_mutex_unlock(&mut_sessions);

	if (ts->null_on_destroy)
		*ts->null_on_destroy = NULL;

	clamma_smp_tss_deinit(&ts->s.tss);

	if (ts->tokens) {
		free(ts->tokens);
		ts->tokens = NULL;
	}

	free(ts->sampler.probindex);
	free(ts->s.x);

	free(ts);
}

int
clamma_session_query(txf_session_t *ts, const clamma_txf_info_t *info)
{
	size_t limit = info->limit;
	char desc[256];
	size_t size;
	int ret = 1;
	char *total;

	if (!info->limit || (uint32_t)info->limit > ts->t->c.seq_len)
		limit = ts->t->c.seq_len;

	ts->sampler.size        = ts->t->c.vocab_size;
	ts->sampler.temperature = info->temperature >= 0.0f ? info->temperature : 0.0f;
	ts->sampler.topp        = info->topp >= 0.0f && info->topp <= 1.0f ? info->topp : 0.9f;
	ts->sampler.rng_state   = info->rng_seed ? info->rng_seed :
						   clamma_timestamp_ns();
	ts->issue_cb            = info->issue_cb ? info->issue_cb : def_iss_cb;
	ts->opaque_user_pointer = info->opaque_user_pointer;
	ts->null_on_destroy	= info->null_on_destroy;

	size = 40 + (info->prompt ? strlen(info->prompt) : 0) +
		    (info->system ? strlen(info->system) : 0);
	total = malloc(size);
	if (!total)
		goto bail;

	switch (ts->t->model_type) {
	case CLAMMA_MODEL_GEN:
		snprintf(total, size - 1, "%s\n%s\n",
				info->system ? info->system : "",
				info->prompt ? info->prompt : "");
		break;
	case CLAMMA_MODEL_CHAT:
		if (info->system)
			snprintf(total, size - 1, "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]\n",
				info->system ? info->system : "",
				info->prompt ? info->prompt : "");
		else
			snprintf(total, size - 1, "[INST] %s [/INST]\n",
					info->prompt ? info->prompt : "");
		break;
	}

	snprintf(desc, sizeof(desc) - 1,
			"    Query: temp: %.02f, topp: %.02f, seed: %llu\n",
			ts->sampler.temperature, ts->sampler.topp,
			(unsigned long long)ts->sampler.rng_state);

	if (info->desc && info->desc_max) {
		strncpy(info->desc, desc, info->desc_max);
		info->desc[info->desc_max - 1] = '\0';
	}

	fprintf(stderr, "%s", desc);
	fflush(stderr);

	if (info->prompt && info->prompt[0])
		clamma_session_issue(ts, info->prompt);

	ts->tokens = clamma_vocab_encode(ts->t, total ? total : "", 1, 0,
					 &ts->ct);
	free(total);
	if (!ts->tokens)
		goto bail;

	ts->limit = limit ? limit : ts->t->c.seq_len;
	ts->token = ts->tokens[0];
	ts->pos = 0;
	ts->start = clamma_timestamp_ns();
	ts->token_count = 0;

	ret = 0;

bail:
	return ret;
}

void
clamma_sessions_query_cancel(struct txf_session *ts)
{
	ts->client_gone = 1;
}

int
clamma_sessions_step_next(void)
{
	txf_session_t *ts, *ts1 = NULL;

	ts = sess_head;
	if (!ts) {
		fprintf(stderr, "no sessions\n");
		return 0;
	}

	if (ts->client_gone)
		goto eol;

	if (ts->pos < ts->limit) {
		bool is_prompt = ts->pos + 1 < ts->ct;

		ts->tnext = clamma_session_forward(ts, is_prompt,
						  ts->token, ts->pos++);

		if (ts->pos >= ts->limit)
			goto eol;

		if (!ts->tnext)
			goto eol;

		if (is_prompt)
			ts->tnext = ts->tokens[ts->pos];
		else {
			if (ts->tokens) {
				free(ts->tokens);
				ts->tokens = NULL;
			}
		}

		if (ts->tnext == TOK_BOS)
			goto eol;

		ts->token_count++;

		if (!is_prompt)
			clamma_session_issue(ts,
					 clamma_vocab_decode(ts->t,
						ts->token, ts->tnext));
		if (ts->pos > 5 && ts->tnext == TOK_EOS)
			goto eol;


		ts->token = ts->tnext;

		/* find the penultimate (ts1) and last (ts) entries
		 * in the list */

		clamma_mutex_lock(&mut_sessions);
		ts = sess_head;
		ts1 = NULL;
		while (ts->next) {
			ts1 = ts;
			ts = ts->next;
		}

		/* move the last guy to be the head */
		if (sess_head != ts)
			ts->next = sess_head; /* new head's next is old head */
		sess_head = ts;
		if (ts1)
			ts1->next = NULL;

		clamma_mutex_unlock(&mut_sessions);

		return 1;
	}



	return 0;

eol:
	{
		char eos[2] = { TOK_EOS, 0 };

		clamma_session_issue(ts, eos);
		clamma_session_destroy(ts);

		return !!sess_head;
	}
}

int
clamma_session_issue(const txf_session_t *ts, const char *piece)
{
	if (ts->client_gone)
		return 0;

	/*
	 * Sanity-check and filter the token string, if we still want
	 * to emit it, call the transformer's callback to do so with it
	 */

	if (piece && piece[0] && piece[1] == '\0' && piece[0] != TOK_EOS) {
		uint8_t byte_val = (uint8_t)piece[0];

		if (!(isprint(byte_val) || isspace(byte_val)))
			return 0;
	}

	if (!ts->issue_cb)
		return 0;

	return ts->issue_cb(ts->opaque_user_pointer, piece);
}
