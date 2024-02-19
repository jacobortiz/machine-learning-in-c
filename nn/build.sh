#!/bin/sh
set -xe

clang -Wall -Wextra -o nn nn.c -lm