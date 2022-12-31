CXX=g++
CXXFLAGS=-std=c++17 -ggdb -Wall -Og
INC=-Ideps/include
LDFLAGS=-Ldeps/lib -lreg

SRC=$(wildcard src/*.cpp)
OBJS=$(addprefix obj/, $(SRC:.cpp=.o))

.PHONY: dirs clean

all: dirs target

target: $(OBJS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS)

obj/src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) $(INC) -c $< -o $@

dirs:
	mkdir -p obj/src

clean:
	-rm -rf obj/ a.out

