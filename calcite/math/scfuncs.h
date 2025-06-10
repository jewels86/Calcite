//
// Created by jewel on 6/10/2025.
//

#ifndef SCFUNCS_H
#define SCFUNCS_H

#include "ScaledDouble.h"

namespace math {
    ScaledDouble sqrt(const ScaledDouble &x);
    ScaledDouble pow(const ScaledDouble &base, const ScaledDouble &exponent);
    ScaledDouble log(const ScaledDouble &x);
    ScaledDouble exp(const ScaledDouble &x);
    ScaledDouble sin(const ScaledDouble &x);
    ScaledDouble cos(const ScaledDouble &x);
    ScaledDouble tan(const ScaledDouble &x);
}

#endif //SCFUNCS_H
