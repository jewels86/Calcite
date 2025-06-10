//
// Created by jewel on 6/10/2025.
//

#ifndef UNIT_H
#define UNIT_H

namespace math {
    template<int L, int M, int T, int Q, int K, int A>
    struct Unit {
        static constexpr int length = L;  // Length exponent
        static constexpr int mass = M;    // Mass exponent
        static constexpr int time = T;    // Time exponent
        static constexpr int charge = Q;  // Charge exponent
        static constexpr int temperature = K; // Temperature exponent
        static constexpr int amount = A;  // Amount of substance exponent
    };
}


#endif //UNIT_H
