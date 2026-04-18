// Dependencies configuration
// Documents required header-only dependencies for xopt

#ifndef XOPT_DEPENDENCIES_H
#define XOPT_DEPENDENCIES_H

// Required C++ standard
#if __cplusplus < 202002L
#error "xopt requires C++20 or later (R >= 4.2.0 recommended)"
#endif

// Dependency documentation:
//
// 1. ucminfcpp (alrobles/ucminfcpp)
//    - BFGS quasi-Newton optimizer
//    - License: GPL-3+ (can be changed to AGPL-3 per user)
//    - Will be added via LinkingTo: ucminfcpp
//
// 2. xad-r/XAD (alrobles/xad-r, auto-differentiation/xad)
//    - Automatic differentiation library
//    - License: AGPL-3
//    - Will be added via LinkingTo: xadr
//
// 3. xtensor-r (alrobles/xtensor-r)
//    - Zero-copy R <-> C++ tensor bridge
//    - License: BSD-3
//    - Will be added via LinkingTo: xtensorR
//
// Phase 0: Document dependencies structure
// Phase 1: Add LinkingTo references and integrate headers

#endif // XOPT_DEPENDENCIES_H
