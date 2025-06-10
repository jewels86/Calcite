//
// Created by jewel on 6/10/2025.
//

#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cmath>
#include "Quantity.h"
#include "ScaledDouble.h"
#include "Unit.h"

/* This file contains physical constants used throughout the simulation.
 * These constants are defined in the following units:
 * - Length: meters (m)
 * - Mass: kilograms (kg)
 * - Time: seconds (s)
 * - Charge: coulombs (C)
 * - Temperature: kelvin (K)
 * - Energy: joules (J)
 */

namespace math::constants {
    // === Units ===
    using Length        = Unit<1, 0, 0, 0, 0, 0>;
    using Mass          = Unit<0, 1, 0, 0, 0, 0>;
    using Time          = Unit<0, 0, 1, 0, 0, 0>;
    using Charge        = Unit<0, 0, 0, 1, 0, 0>;
    using Temperature   = Unit<0, 0, 0, 0, 1, 0>;
    using Energy        = Unit<2, 1, -2, 0, 0, 0>;
    using Amount        = Unit<0, 0, 0, 0, 0, 1>;
    using Velocity      = Unit<1, 0, -1, 0, 0, 0>;
    using Acceleration  = Unit<1, 0, -2, 0, 0, 0>;
    using Force         = Unit<1, 1, -2, 0, 0, 0>;
    using Pressure      = Unit<-1, 1, -2, 0, 0, 0>;
    using Frequency     = Unit<0, 0, -1, 0, 0, 0>;
    using EnergyDensity = Unit<2, -1, -2, 0, 0, 0>;
    using ElectricField = Unit<1, 1, -3, -1, 0, 0>;
    using MagneticField = Unit<0, 1, -2, -1, 0, 0>;
    using ElectricPotential = Unit<2, 1, -3, -1, 0, 0>;
    using Capacitance   = Unit<-2, -1, 4, 2, 0, 0>;
    using Inductance    = Unit<2, 1, -2, -1, 0, 0>;
    using Resistivity   = Unit<1, -3, -2, 0, 0, 0>;
    using Conductivity  = Unit<-1, 3, 2, 0, 0, 0>;

    // === Base Quantities ===
    constexpr auto M  = Quantity<Length>(1.0);
    constexpr auto G  = Quantity<Mass>(1.0);
    constexpr auto S  = Quantity<Time>(1.0);
    constexpr auto C  = Quantity<Charge>(1.0);
    constexpr auto K  = Quantity<Temperature>(1.0);
    constexpr auto J  = Quantity<Energy>(1.0);
    constexpr auto MOL = Quantity<Amount>(1.0);

    // === SI Prefixes (ScaledDouble) ===
    constexpr double_sc NANO  = 1e-9;
    constexpr double_sc MICRO = 1e-6;
    constexpr double_sc MILLI = 1e-3;
    constexpr double_sc CENTI = 1e-2;
    constexpr double_sc DECI  = 1e-1;
    constexpr double_sc KILO  = 1e3;
    constexpr double_sc MEGA  = 1e6;
    constexpr double_sc GIGA  = 1e9;

    // === Length Constants ===
    constexpr auto NM = NANO * M;
    constexpr auto UM = MICRO * M;
    constexpr auto MM = MILLI * M;
    constexpr auto CM = CENTI * M;
    constexpr auto DM = DECI  * M;
    constexpr auto KM = KILO  * M;
    constexpr auto MEGAMETER = MEGA * M;
    constexpr auto GIGAMETER = GIGA * M;
    constexpr auto METER     = M;

    // === Mass Constants ===
    constexpr auto NG = NANO * G;
    constexpr auto UG = MICRO * G;
    constexpr auto MG = MILLI * G;
    constexpr auto CG = CENTI * G;
    constexpr auto DG = DECI  * G;
    constexpr auto KG = KILO  * G;
    constexpr auto MEGAGRAM = MEGA * G;
    constexpr auto GIGAGRAM = GIGA * G;
    constexpr auto GRAM = G;

    // === Charge Constants ===
    constexpr auto KC = KILO * C;
    constexpr auto MC = MEGA * C;
    constexpr auto GC = GIGA * C;
    constexpr auto MILLICOULOMB = MILLI * C;
    constexpr auto MICROCOULOMB = MICRO * C;
    constexpr auto NANOCOULOMB  = NANO  * C;
    constexpr auto COULOMB = C;

    // === Time Constants ===
    constexpr auto MS = MILLI * S;
    constexpr auto US = MICRO * S;
    constexpr auto NS = NANO  * S;
    constexpr auto SECOND = S;

    // === Temperature ===
    constexpr auto KELVIN = K;

    // === Energy Constants ===
    constexpr auto KILOJOULE = KILO * J;
    constexpr auto MEGAJOULE = MEGA * J;
    constexpr auto JOULE = J;

    // === Derived Examples (for clarity) ===
    constexpr auto MPS  = M / S;          // Velocity
    constexpr auto MPS2 = M / (S * S);    // Acceleration
    constexpr auto NEWTON = KG * MPS2;    // Force

} // namespace math::constants

#endif // CONSTANTS_H
