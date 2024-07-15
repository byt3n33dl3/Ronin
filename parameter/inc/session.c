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

static void
quantize(const txf_t *txf, qt_t *qx, const float *x, int n)
{
	int num_groups = n / txf->c.group_size;
	float qmax = 127.0f;

	for (int group = 0; group < num_groups; group++) {

		// find the max absolute value in the current group
		float wmax = 0.0;
		size_t i;

		for (i = 0; i < txf->c.group_size; i++) {
			float val = fabs(x[group * txf->c.group_size + i]);
			if (val > wmax)
				wmax = val;
		}

		// calculate and write the scaling factor
		float scale = wmax / qmax;
		qx->s[group] = scale;

		for (i = 0; i < txf->c.group_size; i++)
			qx->q[group * txf->c.group_size + i] = (cq_t)round(
			    x[group * txf->c.group_size + i] / scale);
	}
}

static int
session_rmsnorm(const txf_t *t, float *o, const float *x, const float *weight,
		size_t size)
{
	// calculate sum of squares
	float ss = 0.0f;
	const float *w;
	size_t j;

	for (j = 0; j < size; j++)
		ss += x[j] * x[j];

	ss /= size;
	ss += 1e-5f;
	ss = 1.0f / sqrtf(ss);

	w = clamma_weight_cache(t, weight, size * sizeof(float));
	if (!w)
		return 1;

	/* normalize and scale */
	for (j = 0; j < size; j++)
		o[j] = w[j] * (ss * x[j]);

	return 0;
}

int
_session_matmul(txf_session_state_t *tss, float *xout, const float *x, const float *w1,
		int i, int dlim, int n, int d)
{
	const float *w = clamma_weight_cache(tss->t, w1, n * d * sizeof(float));

	if (!w)
		return 1;

	w += i * n;
	xout += i;

	for (; i < dlim; i++) {
		float f = 0.0f;
		const float *x1 = x;

		for (int j = 0; j < n; j++)
			f += *w++ * *x1++;

		*xout++ = f;
	}

	return 0;
}

int
_session_matmul_qt(txf_session_state_t *tss, float *xout, const qt_t *x,
		   const qt_t *w1, int i, int dlim, int n, int d)
{
	const cq_t *w_q = clamma_weight_cache(tss->t, w1->q,
				(d * n) + (tss->t->c.group_size * n));
	const float *w_s = clamma_weight_cache(tss->t, w1->s,
				((d * n) / tss->t->c.group_size) * sizeof(*w_s));
	long ln = (long)n;

	if (!w_q || !w_s)
		return 1;

	for (; i < dlim; i++) {

		float val = 0.0f;
		int32_t ival = 0;
		long in = i * n;

		for (long j = 0; j <= ln - (long)tss->t->c.group_size;
						j += tss->t->c.group_size) {
			ival = 0;
			for (unsigned int k = 0; k < tss->t->c.group_size; k++)
				ival = ival + (((int32_t)x->q[j + k]) *
					       ((int32_t)w_q[in + j + k]));

			val += ((float)ival) * w_s[(in + j) / tss->t->c.group_size] *
						     x->s[j / tss->t->c.group_size];
		}

		xout[i] = val;
	}

	return 0;
}

void
session_softmax(float *x, int size)
{
	// find max value (for numerical stability)
	float max_val = x[0];
	float sum = 0.0f;
	int i;

	for (i = 1; i < size; i++)
		if (x[i] > max_val)
			max_val = x[i];

	// exp and sum
	for (i = 0; i < size; i++) {
		x[i] = expf(x[i] - max_val);
		sum += x[i];
	}

	// normalize
	for (i = 0; i < size; i++)
		x[i] /= sum;
}

tok_id_t
clamma_session_forward(txf_session_t *ts, int is_prompt, int token, int pos)
{
	const txf_t *t = ts->t;
	uint32_t kv_dim = (t->c.dim * t->c.n_kv_heads) / t->c.n_heads,
		 kv_mul = t->c.n_heads / t->c.n_kv_heads,
		 head_size = t->c.dim / t->c.n_heads;
	float *content_row = t->w.token_embedding_table + (token * t->c.dim),
	      *key_cache_row, *value_cache_row;
	const float *f = content_row;
	txf_session_state_t *tss = &ts->s.tss;

	switch (t->c.version) {
	case CLAMMA_MODEL_VERSION1_FLOAT:
		f = clamma_weight_cache(t, content_row,
					t->c.dim * sizeof(*ts->s.x));
		if (!f)
			goto bail;
		break;
	}

	/*
	 * Copy the initial token embedding into ts->s.x, this is updated twice
	 * per layer with "residuals"
	 */

	memcpy(ts->s.x, f, t->c.dim * sizeof(*ts->s.x));

	/* for each layer... */

	for (uint64_t l = 0; l < t->c.n_layers; l++) {
		int loff = l * t->c.seq_len * kv_dim;

		// uint64_t start = clamma_timestamp_ns();

		/*
		 * this section parallelizeable ------>
		 */

		/* key and value point to the kv cache */

		tss->k = ts->s.key_cache   + loff + pos * kv_dim;
		tss->v = ts->s.value_cache + loff + pos * kv_dim;

		/*
		 * xb <- resnorm (x, rms_att_weight)
		 *   q  <- matmul(xb, q weights)
		 *   k  <- matmul(xb, v weights)
		 *   v  <- matmul(xb, v weights)
		 */

		switch (t->c.version) {
		case CLAMMA_MODEL_VERSION1_FLOAT:
			/* attention session_rmsnorm */
			if (session_rmsnorm(t, tss->xb, ts->s.x,
					 t->w.rms_att_weight +
					 l * t->c.dim, t->c.dim))
				goto bail;

			/* qkv session_matmuls for this position */
			if (session_matmul(tss, tss->q, tss->xb,
				    (txi_t *)t->w.wq + l * t->c.dim * t->c.dim,
				    t->c.dim, t->c.dim) ||
			    session_matmul(tss, tss->k, tss->xb,
				    (txi_t *)t->w.wk + l * t->c.dim * kv_dim,
				    t->c.dim, kv_dim) ||
			    session_matmul(tss, tss->v, tss->xb,
				    (txi_t *)t->w.wv + l * t->c.dim * kv_dim,
				    t->c.dim, kv_dim))
				goto bail;
			break;
		case CLAMMA_MODEL_VERSION2_INT8_80:
			if (session_rmsnorm(t, tss->xb, ts->s.x,
					 t->w.rms_att_weight +
					 l * t->c.dim, t->c.dim))
				goto bail;

			quantize(t, &tss->xq, tss->xb, t->c.dim);

			/* qkv session_matmuls for this position */
			if (session_matmul_qt(tss, tss->q, &tss->xq, t->w.wq + l,
					   t->c.dim, t->c.dim) ||
			    session_matmul_qt(tss, tss->k, &tss->xq, t->w.wk + l,
					   t->c.dim, kv_dim) ||
			    session_matmul_qt(tss, tss->v, &tss->xq, t->w.wv + l,
					   t->c.dim, kv_dim))
				goto bail;
			break;
		}
		clamma_smp_sync_point(tss);

		/*
		 * RoPE relative positional encoding:
		 *    complex-valued rotate q and optionally k in each head
		 *
		 *     tss->q <-- selfmunge
		 *     tss->k <-- selfmunge
		 */
		for (uint32_t i = 0; i < t->c.dim; i += 2) {
			uint32_t head_dim = i % head_size,
				 do_k = i < kv_dim ? 2 : 1;
			float freq = 1.0f / powf(10000.0f,
					head_dim / (float)head_size),
			      val = pos * freq, fcr = cosf(val), fci = sinf(val);

			for (uint32_t v = 0; v < do_k; v++) {
				float *vec = v == 0 ? tss->q : tss->k,
					v0 = vec[i], v1 = vec[i + 1];

				vec[i]     = v0 * fcr - v1 * fci;
				vec[i + 1] = v0 * fci + v1 * fcr;
			}
		}

		key_cache_row = ts->s.key_cache + loff + pos * kv_dim;
		value_cache_row = ts->s.value_cache + loff + pos * kv_dim;
		memcpy(key_cache_row, tss->k, kv_dim *
					sizeof(*key_cache_row));
		memcpy(value_cache_row, tss->v, kv_dim *
					sizeof(*value_cache_row));

		/* multihead attention. iterate over all heads
		 *
		 *   tss->att <-- tss->s.q, tss->s.key_cache
		 *   tss->xb  <-- zeroed
		 *            <-- value_cache, att
		 */

		for (uint32_t h = 0; h < t->c.n_heads; h++) {
			/* get the query vector for this head */
			float *q = tss->q + h * head_size, *xb,
			      *att = tss->att + h * t->c.seq_len;

			/* iterate over all timesteps, including the current one */
			for (int n = 0; n <= pos; n++) {
				/*
				 * get the key vector for this head
				 * and at this timestep
				 */
				float *k = ts->s.key_cache + loff + n * kv_dim +
						(h / kv_mul) * head_size,
					score = 0.0f;

				for (uint32_t i = 0; i < head_size; i++)
					score += q[i] * k[i];

				score /= sqrtf(head_size);
				att[n] = score;
			}

			/*
			 * softmax the scores to get attention weights,
			 * from 0..pos inclusively
			 */
			session_softmax(att, pos + 1);

			/* weighted sum of the values, store back into xb */
			xb = tss->xb + h * head_size;

			memset(xb, 0, head_size * sizeof(float));

			for (int n = 0; n <= pos; n++) {
				float *v = ts->s.value_cache + loff +
						n * kv_dim +
						(h / kv_mul) * head_size,
					a = att[n];

				for (uint32_t i = 0; i < head_size; i++)
					xb[i] += a * v[i];
			}
		}

		/*
		 * final session_matmul to get the output of the attention
		 *
		 * tss->xb2 = matmul(tss->xb, wo)
		 */

		switch (t->c.version) {
		case CLAMMA_MODEL_VERSION1_FLOAT:
			if (session_matmul(tss, tss->xb2, tss->xb,
				        (txi_t *)t->w.wo +
				        l * t->c.dim * t->c.dim,
					t->c.dim, t->c.dim))
				goto bail;
			break;
		case CLAMMA_MODEL_VERSION2_INT8_80:
			quantize(t, &tss->xq, tss->xb, t->c.dim);
			if (session_matmul_qt(tss, tss->xb2, &tss->xq, t->w.wo + l,
					   t->c.dim, t->c.dim))
				goto bail;
			break;
		}
		clamma_smp_sync_point(tss);

		/* residual connection goes back into ts->s.x */

		/*
		 * ---->  All tss threads must be idle by here
		 *
		 * ts->s.x += tss->xb2
		 */

		for (uint32_t i = 0; i < t->c.dim; i++)
			ts->s.x[i] += tss->xb2[i];

		/*
		 * this section parallelizeable  ---->
		 */

		/* ffn rmsnorm
		 *
		 *   tss->xb <- matmul(ts->s.x, rms_ffn_weight)
		 */
		if (session_rmsnorm(t, tss->xb, ts->s.x,
			     t->w.rms_ffn_weight + l * t->c.dim, t->c.dim))
			goto bail;

		/*
		 * Now for FFN in PyTorch we have:
		 * self.w2(F.silu(self.w1(ts->s.x)) * self.w3(ts->s.x))
		 * first calculate self.w1(ts->s.x) and self.w3(ts->s.x)
		 *
		 *   tss->hb <- matmul(tss->xb, w1)
		 *   tss->hb2 <- matmul(tss->xb, w3)
		 */

		switch (t->c.version) {
		case CLAMMA_MODEL_VERSION1_FLOAT:
			if (session_matmul(tss, tss->hb,  tss->xb,
					   (txi_t *)t->w.w1 +
					   l * t->c.dim * t->c.hidden_dim,
					   t->c.dim, t->c.hidden_dim) ||
			    session_matmul(tss, tss->hb2, tss->xb,
					   (txi_t *)t->w.w3 +
					   l * t->c.dim * t->c.hidden_dim,
				           t->c.dim, t->c.hidden_dim))
				goto bail;
			break;
		case CLAMMA_MODEL_VERSION2_INT8_80:
			quantize(t, &tss->xq, tss->xb, t->c.dim);
			if (session_matmul_qt(tss, tss->hb, &tss->xq, t->w.w1 + l,
					      t->c.dim, t->c.hidden_dim) ||
			    session_matmul_qt(tss, tss->hb2, &tss->xq, t->w.w3 + l,
					      t->c.dim, t->c.hidden_dim))
				goto bail;
			break;
		}
		clamma_smp_sync_point(tss);

		/* SwiGLU non-linearity
		 *
		 *  munge tss->hb
		 */
		for (uint32_t i = 0; i < t->c.hidden_dim; i++)
			/*
			 * silu(ts->s.x)=ts->s.x*σ(ts->s.x), where σ(ts->s.x)
			 * is the logistic sigmoid
			 * elementwise multiply with w3(ts->s.x)
			 */
			tss->hb[i] = (tss->hb[i] * (1.0f /
				     (1.0f + expf(-tss->hb[i])))) * tss->hb2[i];

		/*
		 * tss->xb <-- tss->hb, w2
		 */

		switch (t->c.version) {
		case CLAMMA_MODEL_VERSION1_FLOAT:
			/* final session_matmul to get the output of the ffn */
			if (session_matmul(tss, tss->xb, tss->hb,
				    (txi_t *)t->w.w2 +
				    l * t->c.dim * t->c.hidden_dim,
				    t->c.hidden_dim, t->c.dim))
				goto bail;
			break;
		case CLAMMA_MODEL_VERSION2_INT8_80:
			quantize(t, &tss->hq, tss->hb, t->c.hidden_dim);
			if (session_matmul_qt(tss, tss->xb, &tss->hq, t->w.w2 + l,
					t->c.hidden_dim, t->c.dim))
				goto bail;
			break;
		}
		clamma_smp_sync_point(tss);

		/*
		 * -----> All tss threads must be idle by here
		 */

		/* residual connection goes back into ts->s.x
		 *
		 * ts->s.x += tss->xb
		 */

		for (uint32_t i = 0; i < t->c.dim; i++)
			ts->s.x[i] += tss->xb[i];

		// fprintf(stderr, "%f\n", (double)(clamma_timestamp_ns() - start) / 1000000000.0);
	} /* per layer */

	/*
	 * All tss threads must be idle by here
	 */

	/* final session_rmsnorm
	 *
	 *  ts->s.x <-- matmul(ts.s.x, rms_final_weight)
	 */
	if (session_rmsnorm(t, ts->s.x, ts->s.x, t->w.rms_final_weight, t->c.dim))
		goto bail;

	/* classifier into logits
	 *
	 * logits <-- matmul(ts->s.x, wlcs)
	 */

	switch (t->c.version) {
	case CLAMMA_MODEL_VERSION1_FLOAT:
		if (session_matmul(tss, ts->s.logits, ts->s.x, (txi_t *)t->w.wcls,
			        t->c.dim, t->c.vocab_size))
			goto bail;
		break;
	case CLAMMA_MODEL_VERSION2_INT8_80:
		quantize(t, &ts->s.tss.xq, ts->s.x, t->c.dim);
		if (session_matmul_qt(tss, ts->s.logits, &ts->s.tss.xq, t->w.wcls,
				   t->c.dim, t->c.vocab_size)) {
			goto bail;
		}
		break;
	}
	clamma_smp_sync_point(tss);

	if (is_prompt)
		return token;

	return clamma_sampler_sample(&ts->sampler, ts->s.logits);

bail:
	fprintf(stderr, "%s: bailed\n", __func__);

	return 0;
}
