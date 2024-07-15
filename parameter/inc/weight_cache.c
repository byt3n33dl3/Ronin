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
 *
 * This is an optional implementation for cached access to the model file
 * without requiring mmap.  The speed is similar, this code manually allocates
 * buffers and fills them from the file instead of the kernel doing it.
 *
 * In some cases mmap() is not available on the platform and this can be used
 * instead.
 */

#include "private.h"

static cwc_state_t cwc;

const void *
clamma_weight_cache(const txf_t *t, const void *weight, size_t size)
{
	void *ret = NULL;
	uint64_t ofs;
	ssize_t ar;
	cwc_t *c;

	if (t->model_access != CLAMMA_MODEL_ACCESS_MALLOC_CACHE)
		return weight;

	c = cwc.cwc_head;
	ofs = (uint64_t)((uint8_t *)weight - ((uint8_t *)t->data));

#if defined(LIBCLAMMA_SMP)
	clamma_mutex_lock(&cwc.mut_cwc);
#endif

	while (c) {
		if (c->offset == ofs && c->len == size) {
			c->count++;
			goto hit;
		}
		c = c->next;
	}

	if (t->cache_limit)
		while (cwc.cwc_alloced > t->cache_limit) {
			c = cwc.cwc_head->next;
			cwc.cwc_alloced -= cwc.cwc_head->len;
			free(cwc.cwc_head);
			cwc.cwc_head = c;
		}
	c = malloc(sizeof(*c) + size);
	if (!c) {
		fprintf(stderr, "%s: allocate %llu size failed\n",
				__func__, (unsigned long long)size);
		goto bail;
	}

	c->offset = ofs;
	c->len = size;
	c->count = 1;
	c->next = cwc.cwc_head;

	cwc.cwc_head = c;
	cwc.cwc_created++;
	cwc.cwc_alloced += size;
	cwc.cwc_fetched += size;

	lseek(t->fd, ofs, SEEK_SET);
	ar = read(t->fd, (uint8_t *)c + sizeof(*c), size);
	if (ar != (ssize_t)size) {
		fprintf(stderr, "asked to read %d, read %d\n",
				(int)size, (int)ar);
		goto bail;
	}

hit:
	cwc.cwc_touched += size;
	ret = (uint8_t *)c + sizeof(*c);

bail:
#if defined(LIBCLAMMA_SMP)
	clamma_mutex_unlock(&cwc.mut_cwc);
#endif

	return ret;
}

void
clamma_weight_cache_init(void)
{
#if defined(LIBCLAMMA_SMP)
	clamma_mutex_init(&cwc.mut_cwc);
#endif
}

void
clamma_weight_cache_deinit(void)
{
#if defined(LIBCLAMMA_SMP)
	clamma_mutex_destroy(&cwc.mut_cwc);
#endif
}

void
clamma_weight_cache_clear(void)
{
	cwc_t *c = cwc.cwc_head, *c1;

	fprintf(stderr, "    cwc: created: %d, fetched: %lluM, touched: %lluM\n",
			cwc.cwc_created,
			(unsigned long long)cwc.cwc_fetched / (1024 * 1024),
			(unsigned long long)cwc.cwc_touched / (1024 * 1024));

	while (c) {
		c1 = c;
		c = c->next;
		free(c1);
	}
}
