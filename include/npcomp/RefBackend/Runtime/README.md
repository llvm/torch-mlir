RefBackendRt (namespace `refbackrt`) is the runtime support library for the
RefBackend backend.  It is best practice to keep compiler and runtime code
totally firewalled.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
As such, this directory should have NO DEPENDENCIES ON COMPILER CODE (no
LLVM libSupport, etc.).
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

This will cause some duplication, but history has shown that this
firewalling pays big dividends. In particular, compiler code very
frequently has binary sizes that are simply unacceptable in runtime
scenarios, such as MByte-sized dependencies like LLVM libSupport.
Runtime code should fit in kBytes.
