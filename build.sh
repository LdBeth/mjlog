#!/bin/sh
clang++ -x objective-c++ -std=c++17 -O3 -Wall -flto -fobjc-arc -lz -o out mt19937ar.cc mjlog.cc tenhou.cc
