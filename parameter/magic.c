/* magic -- "Magic" (qix) screen saver hack
 * Copyright (C) 2008  Neale Pickett <neale@woozle.org>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <X11/Xlib.h>
#include "obj.h"

#define FPS 17                  /* Frames Per Second */
#define TRAILPCT 50             /* Percentage of width to cover */
#define MAXDELTAPML 14          /* Maximum velocity per-mil width */
#define MINDELTAPML 2           /* Minimum verocity per-mil width */
#define MINCOLORDELTA 40        /* Minimum color steps per line */
#define MAXCOLORDELTA 150       /* Maximum color steps per line */

#define nsec (1000000000 / FPS)
#define max(a,b) (((a) > (b)) ? (a) : (b))


#define randint(x) (random() % (x))
#define rand(l, h) (randint(h-l) + l)

#define sum(a, b, v, m)     \
  if (b < -v) {             \
    v *= -1;                \
    a = v - b;              \
  } else if (m - b <= v) {  \
    v *= -1;                \
    a = v + b;              \
  } else {                  \
    a = v + b;              \
  }

int
main(int argc, char * const argv[])
{
  Display  *display;
  Window    w      = (Window)0;
  GC        egc    = (GC)0;
  XSegment *lines  = NULL;
  char     *winstr = NULL;
  int       i;

  for (i = 1; i < argc; i += 1) {
    if (0 == strcmp(argv[i], "-window-id")) {
      /* For compatibility with xscreensaver.  We just ignore it. */
    } else if (winstr) {
      w = (Window)-1;
      break;
    } else {
      winstr = argv[i];
    }
  }
  if (! winstr) winstr = getenv("XSS_WINDOW");
  if (winstr) {
    char *end;

    if (winstr[0] == '0' && winstr[1] == 'x') {
      w = (Window)strtol(winstr + 2, &end, 16);
    } else {
      w = (Window)strtol(winstr, &end, 10);
    }
    if ('\0' != *end) {
      w = (Window)-1;
    }
  }
  if ((Window)-1 == w) {
    (void)fprintf(stderr, "Usage: %s [WINDOW_ID]\n", argv[0]);
    return 64;                  /* EX_USAGE */
  }

  if (! (display = XOpenDisplay(NULL))) {
    (void)fprintf(stderr, "cannot open display");
    return 69;                  /* EX_UNAVAILABLE */
  }

  try {
    int            screen = DefaultScreen(display);
    Window         root   = RootWindow(display, screen);
    Colormap       cmap   = DefaultColormap(display, screen);
    XSegment       velocity;
    int            width, height, nlines;
    unsigned short red, green, blue; /* because color.red is reset to the color you got */
    int            dred, dgreen, dblue;
    GC             gc;
    XColor         color;

    if (w) {
      XWindowAttributes wa;

      XGetWindowAttributes(display, w, &wa);
      width = wa.width;
      height = wa.height;
    } else {
      XSetWindowAttributes wa;

      zero(wa);

      wa.override_redirect = 1;
      wa.background_pixel = BlackPixel(display, screen);
      width = DisplayWidth(display, screen) / 3;
      height = DisplayHeight(display, screen) / 3;
      w = XCreateWindow(display, root,
                        0, 0,
                        width, height, 0,
                        CopyFromParent, CopyFromParent, CopyFromParent,
                        CWBackPixel,
                        &wa);
      XMapWindow(display, w);
    }

    srandom((unsigned int)time(NULL));

    {
      int u, l;

      u = width * MAXDELTAPML / 1000;
      l = max(width * MINDELTAPML / 1000, 1);
      velocity.x1 = (short)rand(l, u);
      velocity.y1 = (short)rand(l, u);
      velocity.x2 = (short)rand(l, u);
      velocity.y2 = (short)rand(l, u);
    }

    nlines = (width * TRAILPCT) / (max(velocity.y1, velocity.y2) * 100);
    lines = (XSegment *)calloc(nlines, sizeof(XSegment));

    lines[0].x1 = (short)randint(width);
    lines[0].y1 = (short)randint(height);
    lines[0].x2 = (short)randint(width);
    lines[0].y2 = (short)randint(height);


    /* Allocate graphics contexts */
    {
      XGCValues      values;

      values.background = values.foreground = BlackPixel(display, screen);
      egc = XCreateGC(display, w, GCBackground | GCForeground, &values);

      red = color.red = (unsigned short)randint(65536);
      green = color.green = (unsigned short)randint(65536);
      blue = color.blue = (unsigned short)randint(65536);
      dred = rand(MINCOLORDELTA, MAXCOLORDELTA);
      dgreen = rand(MINCOLORDELTA, MAXCOLORDELTA);
      dblue = rand(MINCOLORDELTA, MAXCOLORDELTA);

      if (! XAllocColor(display, cmap, &color)) break;
      values.foreground = color.pixel;
      gc = XCreateGC(display, w, GCBackground | GCForeground, &values);
    }

    for (i = 0; ;) {
      XSegment        segments[2];
      struct timespec req = {0, nsec};
      int             j   = (i + 1) % nlines;

      /* Erase old line */
      (void)memcpy(segments + 0, lines + j, sizeof(XSegment));
      (void)memcpy(segments + 1, lines + j, sizeof(XSegment));
      segments[1].x1 = width - segments[0].x1;
      segments[1].x2 = width - segments[0].x2;
      (void)XDrawSegments(display, (Drawable)w, egc, segments, 2);

      /* Calculate new line */
      sum(lines[j].x1, lines[i].x1, velocity.x1, width);
      sum(lines[j].y1, lines[i].y1, velocity.y1, height);
      sum(lines[j].x2, lines[i].x2, velocity.x2, width);
      sum(lines[j].y2, lines[i].y2, velocity.y2, height);

      /* Draw new line */
      (void)memcpy(segments + 0, lines + j, sizeof(XSegment));
      (void)memcpy(segments + 1, lines + j, sizeof(XSegment));
      segments[1].x1 = width - segments[0].x1;
      segments[1].x2 = width - segments[0].x2;
      (void)XDrawSegments(display, (Drawable)w, gc, segments, 2);

      /* Change color */
      {
        unsigned long pixel = color.pixel;

        sum(color.red = red, red, dred, 65536);
        sum(color.green = green, green, dgreen, 65536);
        sum(color.blue = blue, blue, dblue, 65536);

        if (! XAllocColor(display, cmap, &color)) break;
        if (! XSetForeground(display, gc, color.pixel)) break;
        if (! XFreeColors(display, cmap, &pixel, 1, 0)) break;
      }
      XSync(display, True);

      (void)nanosleep(&req, NULL);

      i = j;
    }

    (void)XFreeGC(display, gc);
  }

  if (lines) {
    free(lines);
  }

  if (display) {
    if (egc) {
      (void)XFreeGC(display, egc);
    }
    (void)XCloseDisplay(display);
  }

  except {
    (void)fprintf(stderr, "Error: %s\n", exception);
    return 69;                  /* EX_UNAVAILABLE */
  }
  return 0;
}
