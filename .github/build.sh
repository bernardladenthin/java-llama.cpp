#!/bin/bash

# SPDX-FileCopyrightText: 2023-2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
# SPDX-FileCopyrightText: 2023-2026 Konstantin Heurer
#
# SPDX-License-Identifier: MIT

mkdir -p build
cmake -Bbuild $@ || exit 1
cmake --build build --config Release -j$(nproc) || exit 1
