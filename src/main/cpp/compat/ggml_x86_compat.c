#if defined(_MSC_VER) && defined(_M_IX86)
#include <windows.h>

/* On 32-bit x86 MSVC, <intrin.h> declares _InterlockedIncrement64 as
   extern __cdecl but provides no implementation (the intrinsic only exists
   on x64/ARM64). Satisfy the extern with a wrapper around the Win32 API
   InterlockedIncrement64 (implemented via CMPXCHG8B on x86). */
__int64 __cdecl _InterlockedIncrement64(volatile __int64* Addend) {
    return InterlockedIncrement64((volatile LONGLONG*)Addend);
}
#endif
