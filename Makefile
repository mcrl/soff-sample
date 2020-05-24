TARGET=vec_add
OBJECTS=

CFLAGS=-std=c99 -Wall -I$(SOFF_PREFIX)/include
LDFLAGS=-lm -lOpenCL

all: $(TARGET)

clean:
	rm -rf $(TARGET) $(OBJECTS)
