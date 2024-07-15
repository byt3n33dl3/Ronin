/* xbell -- sound the X11 bell
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

#include <stdio.h>
#include <stdlib.h>

#include <X11/Xlib.h>

#include "obj.h"

int
main(int argc, char *argv[])
{
  Display *display = NULL;
  int      percent = 100;

  if (argc == 2) {
    percent = (int)strtol(argv[1], NULL, 10);
  }
  if ((argc > 2) || (0 == percent)) {
    (void)fprintf(stderr, "Usage: %s [PERCENT]\n", argv[0]);
    return 64;                  /* EX_USAGE */
  }

  try {
    if (! (display = XOpenDisplay(NULL))) raise("cannot open display");
    if (! (XBell(display, percent))) break;
  } while (0);

  if (display) {
    (void)XCloseDisplay(display);
  }

  except {
    (void)fprintf(stderr, "Error: %s\n", exception);
    return 69;                  /* EX_UNAVAILABLE */
  }

  return 0;
}
