/* xsswin -- create a screensaver window
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

#include <sys/wait.h>
#include <unistd.h>
#include <stdio.h>

#include <X11/Xlib.h>

#include "obj.h"

int
main(int argc, char * const argv[])
{
  Display *display;
  Pixmap   pmap;
  Window   w;

  if (argc < 2) {
    (void)fprintf(stderr, "Usage: %s PROGRAM [ARGUMENT ...]\n", argv[0]);
    return 64;                  /* EX_USAGE */
  }

  try {
    int                   screen;
    XSetWindowAttributes  wa;
    Window                root;
    int                   i;
    char                 *nargv[argc + 1];
    char                  id[50];

    if (! (display = XOpenDisplay(NULL))) raise("cannot open display");
    screen = DefaultScreen(display);
    root = RootWindow(display, screen);

    wa.override_redirect = 1;
    wa.background_pixel = BlackPixel(display, screen);
    {
      XColor               black;

      pmap = XCreateBitmapFromData(display, root, "\0", 1, 1);
      black.pixel = BlackPixel(display, screen);
      wa.cursor = XCreatePixmapCursor(display, pmap, pmap, &black, &black, 0, 0);
      if (! (XFreePixmap(display, pmap))) break;
    }
    w = XCreateWindow(display, root,
                      0, 0,
                      DisplayWidth(display, screen), DisplayHeight(display, screen), 0,
                      CopyFromParent, CopyFromParent, CopyFromParent,
                      CWOverrideRedirect | CWBackPixel | CWCursor,
                      &wa);
    XMapRaised(display, w);
    XSync(display, False);

    (void)snprintf(id, sizeof(id), "0x%lx", (unsigned long)w);
    (void)setenv("XSS_WINDOW", id, 1);
    for (i = 0; i < argc; i += 1) {
      if (0 == strcmp(argv[i], "XSS_WINDOW")) {
        nargv[i] = id;
      } else {
        nargv[i] = argv[i];
      }
    }
    nargv[argc] = NULL;

    i = fork();
    if (0 == i) {
      (void)execvp(nargv[1], nargv + 1);
      perror("exec");
      return 1;
    }
    (void)waitpid(-1, NULL, 0);
  }

  if (display) {
    (void)XUndefineCursor(display, w);
    (void)XFreePixmap(display, pmap);
    (void)XCloseDisplay(display);
  }

  except {
    (void)fprintf(stderr, "Error: %s\n", exception);
    return 69;                  /* EX_UNAVAILABLE */
  }
  return 0;
}
