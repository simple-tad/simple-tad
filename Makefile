CXX = g++
CXX_FLAGS = -Wall -O3 --std=c++17 -mavx2
TARGET = main
INCLUDES = -I./src -I./include
HPPS = ./src/*.hpp

.PHONY: all run clean debug

all: run

run: $(TARGET)
	./$(TARGET)

$(TARGET): ./src/main.cpp $(HPPS)
	$(CXX) $(CXX_FLAGS) -o $@ $^ $(INCLUDES)

clean:
	rm -rf $(TARGET) *.o *.exe

debug:
	$(CXX) $(CXX_FLAGS) -g -o $(TARGET) ./src/main.cpp $(INCLUDES)
	gdb $(TARGET)