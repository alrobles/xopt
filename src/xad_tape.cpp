// xad_tape.cpp — thin wrapper that pulls XAD's single non-header translation
// unit (Tape.cpp, shipped under xadr's inst/include/XAD/) into xopt's shared
// library.
//
// LinkingTo: xadr exposes <XAD/Tape.cpp> on the include path via its
// LinkingTo-driven -I flag, but Tape.cpp is a .cpp, not a .hpp — R's default
// package build rule auto-compiles every *.cpp in src/, so we simply give it
// a .cpp file to find. Compiling Tape.cpp via #include here, inside *our*
// translation unit, gives xopt its own private XAD tape (as required by
// XAD's per-TU tape model) without any Makevars OBJECTS overrides.

#include <XAD/Tape.cpp>
