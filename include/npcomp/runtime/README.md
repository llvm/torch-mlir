This directory is named `runtime` instead of `Runtime` in order to be a
slight reminder that it is a totally separate codebase from the compiler
code. (There is no difference in naming conventions other than this one
directory though)

It is best practice to keep compiler and runtime code totally firewalled.
Right now, we don't have a good place to put the runtime code that fits in
nicely with #include paths and stuff (we would like users to use something
like `npcomp/runtime/UserAPI.h` to be the include path).

We could have a top-level `runtime` directory with
`runtime/include/npcomp/runtime/UserAPI.h` but that just felt too
heavyweight right now.
