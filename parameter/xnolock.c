#include <X11/Xlib.h>
#include <X11/extensions/scrnsaver.h>
#include <unistd.h>

int
main(int argc, char *argv[])
{
    Display *display   = XOpenDisplay(NULL);

    XScreenSaverSuspend(display, True);
    XFlush(display);
    pause();

    return 0;
}
