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
 * This file is only built if LIBCLAMMA_THREADING is other than OFF.
 * It contains the generic SMP support that doesn't have to get involved in
 * which threading library is in use.
 */

#include "private.h"

work_threads_t *work_threads;
work_t work;
unsigned int count_threads, thread_init_refcount;

// #define SESSION_THREAD_SHOW_OCCUPANCY
// #define LOG_MATRIX_MUL


#if defined(LOG_MATRIX_MUL)
int fd_log = -1, log_line = 1;
#endif

void *
clamma_session_worker(void *tp)
{
	work_threads_t *w = (work_threads_t *)tp;
#if defined(SESSION_THREAD_SHOW_OCCUPANCY)
	uint64_t ns = 0, begin = clamma_timestamp_ns(), start, end;
#endif

	while (1) {
		job_t temp;

		clamma_sem_wait(&w->sem_start);

		while (1) { /* while jobs in ring to do */

			if (w->exiting)
				goto bail;

			clamma_mutex_lock(&work.mut_job);
			if (work.job_tail == work.job_head) {
				/* ring is empty, wait for sem */
				clamma_mutex_unlock(&work.mut_job);
				break;
			}
			temp = work.job_ring[work.job_tail];
			work.job_tail = (work.job_tail + 1) %
					CLAMMA_ARRAY_SIZE(work.job_ring);
			clamma_mutex_unlock(&work.mut_job);

#if defined(SESSION_THREAD_SHOW_OCCUPANCY)
			start = clamma_timestamp_ns();
#endif
			switch (temp.type) {
			case CLAMMA_JOB_MATMUL:
				_session_matmul(temp.tss, temp.xout, temp.x,
						temp.w1, temp.i, temp.dlim,
						temp.n, temp.d);
			break;
			case CLAMMA_JOB_MATMUL_QT:
				_session_matmul_qt(temp.tss, temp.xout,
						   temp.qt_x, temp.qt_w, temp.i,
						   temp.dlim, temp.n, temp.d);
			break;
			}
#if defined(SESSION_THREAD_SHOW_OCCUPANCY)
			ns += clamma_timestamp_ns() - start;
#endif

			clamma_mutex_lock(&work.mut_job);
			assert(temp.tss->queued);
			if (!--temp.tss->queued)
				/* let tss know its jobs are completed */
				clamma_sem_post(&temp.tss->sem_done);

			clamma_mutex_unlock(&work.mut_job);
		}
	}

bail:
#if defined(SESSION_THREAD_SHOW_OCCUPANCY)
	end = clamma_timestamp_ns();
	fprintf(stderr, "t%d: %llums / %llums\n", (int)(w - work_threads),
			(unsigned long long)ns / 1000000ull,
			(end - begin) / 1000000ull);
#endif

	pthread_exit(NULL);
}

/*
 * These are the pthreads-aware version of matmul[_qt] that splits each run into
 * tc parts and queues them up for the threads to handle concurrently
 */

int
session_matmul(txf_session_state_t *tss, float *xout, const float *x,
	       const float *w1, int n, int d)
{
	unsigned int m, part = 0;

#if defined(LOG_MATRIX_MUL)
	char log[256];
	txf_state_t *ts = ((txf_state_t *)((char *)(tss) - offsetof(txf_state_t, tss)));

	if (fd_log == -1)
		fd_log = open("/tmp/log", O_RDWR | O_CREAT | O_TRUNC, 0640);
	if (fd_log != -1) {
		size_t sl = snprintf(log, sizeof(log), "%u, %llu, %llu, %llu, %llu\n",
				log_line++, (unsigned long long)((uint8_t *)x -
							(uint8_t *)ts->x),
				(unsigned long long)((uint8_t *)w1 - (uint8_t *)tss->t->w.token_embedding_table),
	/* extent of x repeatedly covered */	(unsigned long long)(d),
	/* extent of w1 covered once each */	(unsigned long long)(n));
		write(fd_log, log, sl);
	}

#endif

	clamma_mutex_lock(&work.mut_job);

	for (m = 0; m < count_threads; m++) {
		job_t *j = &work.job_ring[work.job_head];

		j->tss	= tss;
		j->type = CLAMMA_JOB_MATMUL;
		j->xout	= xout;
		j->x	= x;
		j->w1	= w1;
		j->i	= part;
		j->n	= n;
		j->d	= d;
		j->dlim	= m == count_threads - 1 ? (unsigned int)d :
						   part + (d / count_threads);

		part += d / count_threads;
		work.job_head = (work.job_head + 1) %
					CLAMMA_ARRAY_SIZE(work.job_ring);
		/* the job ring needs to be bigger */
		assert(work.job_head != work.job_tail);
		tss->queued++;
		/* the job ring needs to be bigger */
		assert(tss->queued < (int)CLAMMA_ARRAY_SIZE(work.job_ring));
	}

	clamma_mutex_unlock(&work.mut_job);

	for (m = 0; m < count_threads; m++)
		clamma_sem_post(&work_threads[m].sem_start);

	return 0;
}

int
session_matmul_qt(txf_session_state_t *tss, float *xout, const qt_t *x,
		 const qt_t *w1, int n, int d)
{
	unsigned int m, part = 0;

	clamma_mutex_lock(&work.mut_job);

	for (m = 0; m < count_threads; m++) {
		job_t *j = &work.job_ring[work.job_head];

		j->tss	= tss;
		j->type = CLAMMA_JOB_MATMUL_QT;
		j->xout	= xout;
		j->qt_x	= x;
		j->qt_w	= w1;
		j->i	= part;
		j->n	= n;
		j->d	= d;
		j->dlim	= m == count_threads - 1 ? (unsigned int)d :
					part + (d / count_threads);

		part += d / count_threads;
		work.job_head = (work.job_head + 1) %
					CLAMMA_ARRAY_SIZE(work.job_ring);
		tss->queued++;
	}

	clamma_mutex_unlock(&work.mut_job);

	for (m = 0; m < count_threads; m++)
		clamma_sem_post(&work_threads[m].sem_start);

	return 0;
}
