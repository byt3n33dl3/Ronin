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
comp(const void *a, const void *b)
{
	return strcmp(((tidx_t *)a)->str, ((tidx_t*)b)->str);
}

int
clamma_vocab_construct(struct txf *t, const char *tokenizer_path)
{
	char search_path[256];
	uint32_t len;
	size_t i = 0;
	ssize_t n;
	int fd;

	memset(&t->v, 0, sizeof(t->v));
	t->v.size = t->c.vocab_size;

	t->v.vocab = (char **)malloc(t->v.size * sizeof(char *));
	if (!t->v.vocab)
		goto bail;

	t->v.scores = (float *)malloc(t->v.size * sizeof(*t->v.scores));
	if (!t->v.scores)
		goto bail1;

	fd = open(tokenizer_path, O_RDONLY);
	if (fd < 0) {
		snprintf(search_path, sizeof(search_path) - 1, "%s/%s",
				CLAMMA_MODEL_SEARCH_PATH,
				tokenizer_path);
		fd = open(search_path, O_RDONLY);
		if (fd < 0) {
			fprintf(stderr, "couldn't load vocab %s\n", search_path);
			goto bail2;
		}
	}
	t->v.storage_size = lseek(fd, 0, SEEK_END);
	lseek(fd, 0, SEEK_SET);

	n = read(fd, &t->v.max_token_length,
		 sizeof(t->v.max_token_length));
	if (n != (ssize_t)sizeof(t->v.max_token_length)) {
		fprintf(stderr, "failed read 1\n");
		goto bail3;
	}

	for (i = 0; i < t->v.size; i++) {
		if (read(fd, t->v.scores + i, sizeof(float)) != sizeof(float)) {
			fprintf(stderr, "failed read 2\n");
			goto bail4;
		}
		if (read(fd, &len, sizeof(len)) != sizeof(len)) {
			fprintf(stderr, "failed read 3\n");
			goto bail4;
		}
		t->v.vocab[i] = (char *)malloc(len + 1);
		if (!t->v.vocab[i]) {
			goto bail4;
		}
		n = read(fd, t->v.vocab[i], len);
		if (n != (ssize_t)len) {
			fprintf(stderr, "failed read 4 %lld %llu\n",
					(long long)n, (unsigned long long)len);
			free(t->v.vocab[i]);
			goto bail4;
		}
		t->v.vocab[i][len] = '\0';
	}
	close(fd);

	t->v.sorted_vocab = malloc(t->v.size * sizeof(tidx_t));
	if (!t->v.sorted_vocab)
		goto bail4;

	for (size_t i = 0; i < t->v.size; i++) {
		t->v.sorted_vocab[i].str = t->v.vocab[i];
		t->v.sorted_vocab[i].id = i;
	}

	qsort(t->v.sorted_vocab, t->v.size, sizeof(tidx_t), comp);

	return 0;

bail4:
	while (i--)
		free(t->v.vocab[i]);

bail3:
	close(fd);
bail2:
	free(t->v.scores);
bail1:
	free(t->v.vocab);
bail:

	return 1;
}

void
clamma_vocab_destroy(struct txf *t)
{
	for (size_t i = 0; i < t->v.size; i++)
		free(t->v.vocab[i]);
	free(t->v.vocab);
	free(t->v.scores);
	free(t->v.sorted_vocab);
}

const char *
clamma_vocab_decode(const struct txf *t, int prev_token, int token)
{
	const char *piece = t->v.vocab[token];
	char *p = (char *)&t->v.utf8;

	if (prev_token == 1 && piece[0] == ' ')
		piece++;

	if (piece[0] == '<' && piece[1] == '0' && piece[2] == 'x') {
		int m;

		*p = 0;
		piece += 3;

		for (m = 0; ; m++) {
			if (!*piece || m >= 8)
				return piece;

			if (*piece == '>')
				return t->v.utf8;

			if (*piece >= '0' && *piece <= '9')
				*p = (*p << 4) | ((*piece) - '0');
			else
				if (*piece >= 'a' && *piece <= 'f')
					*p = (*p << 4) | ((*piece) - 'a' + 10);
				else
					if (*piece >= 'A' && *piece <= 'F')
						*p = (*p << 4) |
							((*piece) - 'A' + 10);

			if (m & 1) {
				p++;
				*p = 0;
			}

			piece++;
		}

		return t->v.utf8;
	}

	return piece;
}

static int
str_lookup(char *str, tidx_t *sorted_vocab, int size)
{
	tidx_t tok = { .str = str }, // acts as the key to search for
	       *res = bsearch(&tok, sorted_vocab, size, sizeof(tidx_t), comp);

	return res ? res->id : -1;
}

tok_id_t *
clamma_vocab_encode(const txf_t *t, const char *text, int8_t bos, int8_t eos,
		    size_t *n_tokens)
{
	char *str_buffer;
	size_t str_len = 0;
	tok_id_t *tokens, id;

	if (!text)
		return NULL;

	tokens = malloc((strlen(text) + 3) * sizeof(tokens[0]));
	if (!tokens)
		return NULL;

	/*
	 * create a temporary buffer that will store merge candidates of always
	 * two consecutive tokens
	 */
	str_buffer = malloc(t->v.max_token_length * 2 + 1 + 2);
	if (!str_buffer)
		goto bail1;

	*n_tokens = 0;

	if (bos)
		tokens[(*n_tokens)++] = TOK_BOS;

	/*
	 * add_dummy_prefix is true by default
	 * so prepend a dummy prefix token to the input string, but only if
	 * text != ""
	 * TODO: pretty sure this isn't correct in the general case but I
	 * don't have the energy to read more of the sentencepiece code to
	 * figure out what it's doing
	 */
	if (text[0])
		tokens[(*n_tokens)++] =
				str_lookup(" ", t->v.sorted_vocab, t->v.size);

	for (const char *c = text; *c; c++) {

		if ((*c & 0xC0) != 0x80)
			str_len = 0;
		str_buffer[str_len++] = *c;
		str_buffer[str_len] = '\0';

		if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4)
			continue;

		id = str_lookup(str_buffer, t->v.sorted_vocab, t->v.size);
		if (id != -1)
			tokens[(*n_tokens)++] = id;
		else
			/*
			 * byte_fallback encoding: just encode each byte as a
			 * token.  +3 is here because the first 3 vocab elements
			 * are <unk>, <s>, </s>  so the individual bytes only
			 * start at index 3
			 */
			for (size_t i = 0; i < str_len; i++)
				tokens[(*n_tokens)++] =
						(uint8_t)str_buffer[i] + 3;

		str_len = 0;
	}

	/*
	 * merge the 'best' (by score) consecutive pair each iteration
	 */

	while (1) {
		float best_score = -1e10;
		tok_id_t best_id = -1;
		int best_idx = -1, id;

		for (size_t i = 0; *n_tokens > 2 && i < (*n_tokens - 1); i++) {
			sprintf(str_buffer, "%s%s", t->v.vocab[tokens[i]],
						t->v.vocab[tokens[i + 1]]);
			id = str_lookup(str_buffer, t->v.sorted_vocab,
					t->v.size);
			if (id != -1 && t->v.scores[id] > best_score) {
				best_score = t->v.scores[id];
				best_id = id;
				best_idx = i;
			}
		}

		if (best_idx == -1)
			break;

		tokens[best_idx] = best_id;

		for (size_t i = best_idx + 1; i < (*n_tokens - 1); i++)
			tokens[i] = tokens[i + 1];

		(*n_tokens)--;
	}

	free(str_buffer);
	if (eos)
		tokens[(*n_tokens)++] = TOK_EOS;

	return tokens;

bail1:
	free(tokens);

	return NULL;
}
