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

static int
sample_argmax(float *probabilities, int n)
{
	/* return the index that has the highest probability */
	int max_i = 0;
	float max_p = probabilities[0];

	for (int i = 1; i < n; i++)
		if (probabilities[i] > max_p) {
			max_i = i;
			max_p = probabilities[i];
		}

	return max_i;
}

static int
sample_mult(float *probabilities, int n, float coin)
{
	/*
	 * sample index from probabilities (they must sum to 1!)
	 * coin is a random number in [0, 1], usually from random_f32()
	 */
	float cdf = 0.0f;

	for (int i = 0; i < n; i++) {
		cdf += probabilities[i];

		if (coin < cdf)
			return i;
	}

	return n - 1; // in case of rounding errors
}

static int
compare(const void *a, const void* b)
{
	pidx_t *a_ = (pidx_t *) a;
	pidx_t *b_ = (pidx_t *) b;

	if (a_->prob > b_->prob)
		return -1;
	if (a_->prob < b_->prob)
		return 1;

	return 0;
}

static int
sample_topp(float *probabilities, int n, float topp, pidx_t *probindex, float coin)
{
	/*
	 * top-p sampling (or "nucleus sampling") samples from the smallest set
	 * of tokens that exceed probability topp. This way we never sample
	 * tokens that have very low probabilities, and so are less likely to go
	 *  "off the rails".
	 *
	 * coin is a random number in [0, 1], usually from random_f32()
	 *
	 * quicksort indices in descending order of probabilities
	 * values smaller than (1 - topp) / (n - 1) cannot be part of the result
	 * so for efficiency we crop these out as candidates before sorting
	 */

	const float cutoff = (1.0f - topp) / (n - 1);
	float cumulative_prob = 0.0f, cdf = 0.0f, r;
	int n0 = 0, i, last_idx;

	for (i = 0; i < n; i++) {
		if (probabilities[i] >= cutoff) {
			probindex[n0].index = i;
			probindex[n0].prob = probabilities[i];
			n0++;
		}
	}

	qsort(probindex, n0, sizeof(pidx_t), compare);

	/*
	 * truncate the list where cumulative probability exceeds topp
	 */

	/* in case of rounding errors consider all elements */
	last_idx = n0 - 1;
	for (i = 0; i < n0; i++) {
		cumulative_prob += probindex[i].prob;
		if (cumulative_prob > topp) {
			last_idx = i;
			break; /* we've exceeded topp by including last_idx */
		}
	}

	/* sample from the truncated list */

	r = coin * cumulative_prob;

	for (i = 0; i <= last_idx; i++) {
		cdf += probindex[i].prob;
		if (r < cdf)
			return probindex[i].index;
	}

	return probindex[last_idx].index;
}

static unsigned int
random_u32(uint64_t *state)
{
	// xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
	*state ^= *state >> 12;
	*state ^= *state << 25;
	*state ^= *state >> 27;

	return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

static float
random_f32(uint64_t *state)
{
	return (random_u32(state) >> 8) / 16777216.0f;
}

int
clamma_sampler_sample(txf_sampler_t *sampler, float *logits)
{
	float coin = random_f32(&sampler->rng_state);

	if (sampler->temperature == 0.0f)
		/*
		 * greedy argmax sampling: take the token
		 * with the highest probability
		 */
		return sample_argmax(logits, sampler->size);

	/* apply the temperature to the logits */
	for (size_t n = 0; n < sampler->size; n++)
		logits[n] = logits[n] / sampler->temperature;

	/* get the probabilities for next token from logits */
	session_softmax(logits, sampler->size);

	/* we sample from this distribution to get the next token */
	if (sampler->topp <= 0 || sampler->topp >= 1)
		/* simply sample from the predicted probability distribution */
		return sample_mult(logits, sampler->size, coin);

	/* top-p (nucleus) sampling, clamping the least likely tokens to zero */
	return sample_topp(logits, sampler->size, sampler->topp,
			   sampler->probindex, coin);
}
