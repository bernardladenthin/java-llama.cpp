REM SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
REM SPDX-FileCopyrightText: 2023-2025 Konstantin Heurer
REM
REM SPDX-License-Identifier: MIT

@echo off

mkdir build
cmake -Bbuild %*
cmake --build build --config Release

if errorlevel 1 exit /b %ERRORLEVEL%