NVCC = nvcc
FLAGS = -O3 -w -arch=sm_61 -rdc=true -std=c++11 -I./src/extern
OBJFILES = obj/Driver.o \
			obj/lodepng.o

.PHONY: all clean 

all: obj ray-tracer

ray-tracer: $(OBJFILES)
	$(NVCC) $(FLAGS) $^ -o $@ 

obj/%.o: src/%.cu
	$(NVCC) $(FLAGS) -c -o $@ $< -lcudadevrt

obj/%.o: src/extern/%.cpp
	$(NVCC) $(FLAGS) -c -o $@ $<

obj:
	mkdir -p obj

clean:
	$(RM) obj/*.o ray-tracer
	$(RM) ./*.png
