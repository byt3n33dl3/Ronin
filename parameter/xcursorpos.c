/* xcursorpos -- print current cursor coordinates
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
#include <X11/Xlib.h>
#include "obj.h"

int
main(int argc, char *argv[])
{
  Display *display = NULL;

  if (argc != 1) {
    (void)fprintf(stderr, "Usage: %s\n", argv[0]);
    return 64;                  /* EX_USAGE */
  }

  try {
    Window       root, win;
    int          x, y, win_x, win_y;
    unsigned int mask;

    if (! (display = XOpenDisplay(NULL))) raise("cannot open display");
    root = DefaultRootWindow(display);
    if (! XQueryPointer(display, root, &root, &win, &x, &y, &win_x, &win_y, &mask)) {
      raise("unable to query pointer");
    }
    (void)printf("%d %d\n", x, y);
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
