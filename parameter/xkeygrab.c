/*
 * xkeygrab -- grab keyboard and mouse, writing typed lines to stdout
 * Copyright (C) 2008 Neale Pickett <neale@woozle.org> 
 *
 * This program is free software:  you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>. 
 */

#include <stdio.h>
#include <unistd.h>
#include <ctype.h>
#include <poll.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>

#include "obj.h"

int
main(int argc, char *argv[])
{
    Display        *display = NULL;

    if (argc != 1) {
        (void) fprintf(stderr, "Usage: %s\n", argv[0]);
        return 64;              /* EX_USAGE */
    }

    try {
        Window          root;
        char            obuf[4096];
        int             obuflen = 0;
        struct pollfd   fds[2];

        if (!(display = XOpenDisplay(NULL)))
            raise("cannot open display");
        root = DefaultRootWindow(display);

        if (getenv("NOROOT")) {
            int             screen = DefaultScreen(display);

            XSetWindowAttributes wa;

            zero(wa);

            wa.override_redirect = 1;
            wa.background_pixel = BlackPixel(display, screen);
            root = XCreateWindow(display, root,
                                 0, 0,
                                 50, 50, 0,
                                 CopyFromParent, CopyFromParent,
                                 CopyFromParent, CWBackPixel, &wa);
            XMapWindow(display, root);
        }


        fds[0].fd = STDOUT_FILENO;
        fds[0].events = 0;
        fds[1].fd = ConnectionNumber(display);
        fds[1].events = POLLIN;
        while (1) {
            int             tograb = 3;
            int             ret;

            if (tograb) {
                if ((tograb & 1)
                    && (GrabSuccess ==
                        XGrabKeyboard(display, root, True, GrabModeAsync,
                                      GrabModeAsync, CurrentTime))) {
                    tograb |= 1;
                }
                if ((tograb & 2)
                    && (GrabSuccess ==
                        XGrabPointer(display, root, False,
                                     ButtonPressMask | ButtonReleaseMask |
                                     PointerMotionMask, GrabModeAsync,
                                     GrabModeAsync, None, None,
                                     CurrentTime))) {
                    tograb |= 2;
                }
            }

            ret = poll(fds, 2, (tograb ? 100 : -1));
            if (-1 == ret)
                break;

            if (fds[0].revents & POLLERR) {
                break;
            }
            if (fds[1].revents & POLLIN) {
                XEvent          event;

                do {
                    (void) XNextEvent(display, &event);
                    if (KeyPress == event.type) {
                        char            buf[32];
                        KeySym          ksym;
                        int             i;

                        if (obuflen == sizeof(obuf))
                            continue;
                        i = XLookupString(&event.xkey, buf, sizeof(buf),
                                          &ksym, NULL);
                        switch (ksym) {
                            case XK_Return:
                                if (obuflen) {
                                    (void) write(STDOUT_FILENO, obuf,
                                                 obuflen);
                                    (void) write(STDOUT_FILENO, "\n", 1);
                                    obuflen = 0;
                                }
                                break;
                            case XK_BackSpace:
                                if (obuflen)
                                    obuflen -= 1;
                                break;
                            default:
                                if (1 == i) {
                                    switch (buf[0]) {
                                        case '\025':
                                            obuflen = 0;
                                            break;
                                        case ' ' ... '~':
                                            obuf[obuflen++] = buf[0];
                                            break;
                                    }
                                }
                                break;
                        }
                        if (obuflen == sizeof(obuf)) {
                            (void) write(STDOUT_FILENO, obuf, obuflen);
                            obuflen = 0;
                        }
                    }
                } while (XPending(display));
            }
        }
    }
    while (0);

    if (display) {
        (void) XUngrabKeyboard(display, CurrentTime);
        (void) XUngrabPointer(display, CurrentTime);
        (void) XCloseDisplay(display);
    }

    except {
        (void) fprintf(stderr, "Error: %s\n", exception);
        return 69;              /* EX_UNAVAILABLE */
    }

    return 0;
}
