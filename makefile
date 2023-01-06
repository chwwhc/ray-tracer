CXX = g++
FLAGS = -g -Wall -O3
OBJFILES = obj/Driver.o

.PHONY: all clean 

all: obj ray-tracer

ray-tracer: $(OBJFILES)
	$(CXX) $(FLAGS) $^ -o $@ 

obj/%.o: src/%.cc
	$(CXX) $(FLAGS) -c -o $@ $<

obj:
	mkdir -p obj

clean:
	$(RM) obj/*.o ray-tracer
	$(RM) ./*.ppm
	