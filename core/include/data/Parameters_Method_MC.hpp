#pragma once
#ifndef SPIRIT_CORE_DATA_PARAMETERS_METHOD_MC_HPP
#define SPIRIT_CORE_DATA_PARAMETERS_METHOD_MC_HPP

#include <data/Parameters_Method.hpp>

#include <random>
#include <vector>

namespace Data
{

// LLG_Parameters contains all LLG information about the spin system
struct Parameters_Method_MC : public Parameters_Method
{
    // Temperature [K]
    scalar temperature = 0;
    // Seed for RNG
    int rng_seed = 2006;

    // Mersenne twister PRNG
    std::mt19937 prng = std::mt19937( rng_seed );

    // Whether to sample spins randomly or in sequence in Metropolis algorithm
    bool metropolis_random_sample = true;
    // Whether to use the adaptive cone radius (otherwise just uses full sphere sampling)
    bool metropolis_step_cone = true;
    // Whether to adapt the metropolis cone angle throughout a MC run to try to hit a target acceptance ratio
    bool metropolis_cone_adaptive = true;
    // The metropolis cone angle
    scalar metropolis_cone_angle = 30;
    // Whether to start with a spin-flip (180 degree inversion) betfore applying the cone. For Ising-like spins
    // Set to non-zero to enable (not using boolean b/c the config parser only understands numbers)
    scalar metropolis_spin_flip = 0.0;

    // Target acceptance ratio of mc steps for adaptive cone angle
    scalar acceptance_ratio_target = 0.5;

    //Quantum tunneling parameters. Initially, use the quadratic approximation for tunneling strength/acceptance probability
    bool tunneling_use_tunneling = true;   //On by default
    scalar tunneling_gamma = 2.7e-1;    //Energy scale of quadratic term of Gamma in LiHoF, K/T^2
    int tunneling_spin_flip_count;      //Number of spin flips allowed due to quantum tunneling after each pass through the Metropolis loop.

    // ----------------- Output --------------
    // Energy output settings
    bool output_energy_step                  = false;
    bool output_energy_archive               = false;
    bool output_energy_spin_resolved         = false;
    bool output_energy_divide_by_nspins      = true;
    bool output_energy_add_readability_lines = false;
    // Spin configurations output settings
    bool output_configuration_step    = false;
    bool output_configuration_archive = false;
};

} // namespace Data

#endif