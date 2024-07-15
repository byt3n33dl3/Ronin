#include <stdio.h>
#include <stdlib.h>
#include <X11/Xlib.h>
#include <X11/extensions/scrnsaver.h>


int main(int argc, char *argv[]) {
	Display *display = XOpenDisplay(NULL);
	if (display == NULL) {
		fprintf(stderr, "Couldn't open display!\n");
		exit(EXIT_FAILURE);
	}
	
	int xss_major_opcode;
	int xss_event_offset;
	int xss_error_offset;
	if (!XQueryExtension(display, ScreenSaverName, &xss_major_opcode, &xss_event_offset, &xss_error_offset)) {
		fprintf(stderr, ScreenSaverName " extension is not available!\n");
		exit(EXIT_FAILURE);
	}
	
	
	XScreenSaverSelectInput(display, DefaultRootWindow(display), ScreenSaverNotifyMask | ScreenSaverCycleMask);
	
	int previous_state = -1;
	while (1) {
		XScreenSaverInfo info;
		if (!XScreenSaverQueryInfo(display, DefaultRootWindow(display), &info)) {
			fprintf(stderr, "Couldn't query screen saver info!\n");
			exit(EXIT_FAILURE);
		}
		
		if (info.state != previous_state) {
			previous_state = info.state;
			switch (info.state) {
				case ScreenSaverOn:
					printf("on\n");
					break;
				case ScreenSaverOff:
					printf("off\n");
					break;
				case ScreenSaverDisabled:
					printf("disabled\n");
					break;
			}
			
			fflush(stdout);
		}
		
		XEvent event;
		XNextEvent(display, &event);  // wait for next event
	}
	
	XCloseDisplay(display);
	return 0;
}
