/* xss -- xautolock replacement using MIT-SCREEN-SAVER (it doesn't poll)
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
#include <signal.h>
#include <stdio.h>

#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <X11/extensions/scrnsaver.h>

#include "obj.h"

int child = 0;

void
sigchld(int signum)
{
  (void)waitpid(-1, NULL, WNOHANG);
  child = 0;
}

int
main(int argc, char *argv[])
{
  Display *display   = NULL;
  int      screen;
  int      bigwindow = 0;

  if ((argc > 1) && (0 == (strcmp(argv[1], "-w")))) {
    bigwindow = 1;
  }
  if (argc - bigwindow < 2) {
    (void)fprintf(stderr, "Usage: %s [-w] PROGRAM [ARGUMENT ...]\n", argv[0]);
    (void)fprintf(stderr, "\n");
    (void)fprintf(stderr, "-w    Map X server window, export $XSS_WINDOW.  This window\n");
    (void)fprintf(stderr, "      will be unmapped by any keypress or mouse movement.\n");
    return 64;                  /* EX_USAGE */
  }
  signal(SIGCHLD, sigchld);

  try {
    int     ss_event, ss_error;
    XEvent  event;
    Window  root;
    char   *nargv[argc+1];
    char    id[50];
    int     i;

    if (! (display = XOpenDisplay(NULL))) raise("cannot open display");
    screen = DefaultScreen(display);
    if (! XScreenSaverQueryExtension(display, &ss_event, &ss_error)) {
      raise("X server does not support MIT-SCREEN-SAVER extension.");
    }
    if (! XScreenSaverRegister(display, screen, (XID)getpid(), XA_INTEGER)) {
      raise("cannot register screen saver, is another one already running?");
    }
    root = RootWindow(display, screen);
    XScreenSaverSelectInput(display, root, ScreenSaverNotifyMask);

    {
      XSetWindowAttributes wa;
      XScreenSaverInfo     info;
      Pixmap               pmap;
      XColor               black;

      pmap = XCreateBitmapFromData(display, root, "\0", 1, 1);
      black.pixel = BlackPixel(display, screen);
      wa.cursor = XCreatePixmapCursor(display, pmap, pmap, &black, &black, 0, 0);
      wa.background_pixel = BlackPixel(display, screen);
      if (! (XFreePixmap(display, pmap))) break;
      XScreenSaverSetAttributes(display, root,
                                bigwindow?0:-1, bigwindow?0:-1,
                                bigwindow?DisplayWidth(display, screen):1,
                                bigwindow?DisplayHeight(display, screen):1,
                                0,
                                CopyFromParent, CopyFromParent,
                                CopyFromParent,
                                CWBackPixel | CWCursor,
                                &wa);
      XScreenSaverQueryInfo(display, (Drawable)root, &info);
      (void)snprintf(id, sizeof(id), "0x%lx", (unsigned long)info.window);
      (void)setenv("XSS_WINDOW", id, 1);
    }

    for (i = 0; i < argc; i += 1) {
      if (bigwindow && (0 == strcmp(argv[i], "XSS_WINDOW"))) {
        nargv[i] = id;
      } else {
        nargv[i] = argv[i];
      }
    }
    nargv[argc] = NULL;

    while (! XNextEvent(display, &event)) {
      if (ss_event == event.type) {
        XScreenSaverNotifyEvent *sevent = (XScreenSaverNotifyEvent *)&event;

        if (ScreenSaverOn == sevent->state) {
          if (! child) {
            child = fork();
            if (0 == child) {
              (void)execvp(nargv[1+bigwindow], nargv+1+bigwindow);
              perror("exec");
              exit(1);
            }
          } else {
          }
        }
      }
    }
  } while (0);

  if (display) {
    (void)XScreenSaverUnregister(display, screen);
    (void)XCloseDisplay(display);
  }

  except {
    (void)fprintf(stderr, "Error: %s\n", exception);
    return 69;                  /* EX_UNAVAILABLE */
  }

  return 0;
}
