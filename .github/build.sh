#!/bin/bash

# SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
# SPDX-FileCopyrightText: 2023-2025 Konstantin Herud
#
# SPDX-License-Identifier: MIT

mkdir -p build
cmake -Bbuild $@ || exit 1
cmake --build build --config Release -j$(nproc) || exit 1
