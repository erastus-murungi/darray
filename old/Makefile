CC := clang
TARGET := strassen

SRC := strassen.c main.c
OBJS := $(SRC:.c=.o)
COMMON_FLAGS := -pedantic -Wextra

CFLAGS := $(COMMON_FLAGS) -std=gnu11 -march=native
LDFLAGS := -flto

ifeq ($(DEBUG),1)
	CFLAGS += -DDEBUG -O0 -g
else
	CFLAGS += -DNDEBUG -O3
endif

ifeq ($(PROFILE), 1)
	LDFLAGS += -lprofiler
endif

ifeq ($(PARALLEL), 1)
	CFLAGS += -DPARALLEL
	LDFLAGS += -Xpreprocessor -fopenmp -lomp
endif

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS) -Rpass=.*

main.o: main.c
strassen.o: strassen.c

FILE:= main.c
llvm:
	$(CC) -O3 -S -emit-llvm -march=native $(FILE)

asm:
	$(CC) -O3 -S -march=native $(FILE)

profile: $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS) -lprofiler -Rpass=.*
	LD_PRELOAD=/usr/local/lib/libprofiler.so CPUPROFILE=test.prof ./$(TARGET)
	pprof --web ./$(TARGET) test.prof 

.PHONY: clean
clean:
	rm strassen $(OBJS)
