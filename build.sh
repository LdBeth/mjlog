#!/bin/sh
clang++ -std=c++17 -O3 -Wall -flto -fobjc-arc -o out mt19937ar.mm mjlog.mm tenhou.mm
