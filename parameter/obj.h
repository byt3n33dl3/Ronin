/* obj.h: objecty and exceptiony stuff
 *
 * Some macros to make C a bit more like C++, but without bringing in
 * all of C++'s crapola.
 */

#ifndef __OBJ_H__
#define __OBJ_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

/** Exception-type things.  Don't nest them.
 *
 * These allow you to have pseudo-exceptions.  It looks kludgy and it
 * is, but I think it makes the actual code easier to understand.
 */
static char *exception;
#define try for (exception = "Failure"; exception; exception = NULL)
#define fail break
#define pfail {exception = strerror(errno); break;}
#define succeed continue
#define raise(x) {exception = x; break;}
#define except if (exception)

/** Allocate something */
#define new(type) (type *)calloc(1, sizeof(type))

/** Clear something out */
#define zero(x) (void)memset(&x, 0, sizeof(x))

/** Printf debugging macros */
#ifdef NODUMP
#  define DUMPf(fmt, args...)
#else
#  define DUMPf(fmt, args...) fprintf(stderr, "%s:%s:%d " fmt "\n", __FILE__, __FUNCTION__, __LINE__, ##args)
#endif
#define DUMP() DUMPf("")
#define DUMP_d(v) DUMPf("%s = %d", #v, v)
#define DUMP_x(v) DUMPf("%s = 0x%x", #v, v)
#define DUMP_s(v) DUMPf("%s = %s", #v, v)
#define DUMP_c(v) DUMPf("%s = '%c' (0x%02x)", #v, v, v)
#define DUMP_p(v) DUMPf("%s = %p", #v, v)

#endif
