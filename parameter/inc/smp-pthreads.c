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
 * This isolates the pthreads-specific smp support, it only gets built if the
 * cmake option -DLIBCLAMMA_THREADING=PTHREADS
 *
 * Being thread library agnositic is mainly taken care of at the top of private.h
 * where generic types and handlers are defined to wrap the thread library
 * apis.
 *
 * This file is for more awkward apis like thread creation / join using the
 * native thread library apis, which might be even more awkward for other
 * thread systems.
 */

#include "private.h"

void
clamma_smp_sync_point(txf_session_state_t *tss)
{
	do {
		clamma_sem_wait(&tss->sem_done);
		if (!tss->queued)
			return;
	} while (1);
}

int
clamma_smp_tss_init(txf_session_state_t *tss)
{
	return clamma_sem_init(&tss->sem_done);
}

void
clamma_smp_tss_deinit(txf_session_state_t *tss)
{
	clamma_sem_destroy(&tss->sem_done);
}

void
clamma_smp_deinit(void)
{
	void *vret;
	unsigned int n;

	if (--thread_init_refcount)
		return;

	for (n = 0; n < count_threads; n++)
		if (work_threads[n].running) {
			work_threads[n].exiting = 1;
			clamma_sem_post(&work_threads[n].sem_start);
			pthread_join(work_threads[n].pt, &vret);
			clamma_sem_destroy(&work_threads[n].sem_start);
			work_threads[n].running = 0;
		}

	pthread_mutex_destroy(&work.mut_job);
	pthread_mutex_destroy(&mut_sessions);
	count_threads = 0;
	free(work_threads);
}

int
clamma_smp_init(unsigned int threads)
{
	unsigned int n;

	if (thread_init_refcount++)
		return 0;

	work_threads = malloc(sizeof(*work_threads) * threads);
	if (!work_threads)
		return 1;
	memset(work_threads, 0, sizeof(*work_threads) * threads);
	count_threads = threads;

	for (n = 0; n < count_threads; n++) {
		if (clamma_sem_init(&work_threads[n].sem_start))
			goto bail;

		if (pthread_create(&work_threads[n].pt, NULL,
				   clamma_session_worker,
				   &work_threads[n])) {
			clamma_sem_destroy(&work_threads[n].sem_start);
bail:
			clamma_smp_deinit();

			return 1;
		}
		work_threads[n].running = 1;
	}

	clamma_mutex_init(&work.mut_job);
	clamma_mutex_init(&mut_sessions);

	return 0;
}
