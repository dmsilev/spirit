#pragma once
#ifndef SPIRIT_CORE_UTILITY_CONFIGURATION_CHAIN_HPP
#define SPIRIT_CORE_UTILITY_CONFIGURATION_CHAIN_HPP

#include <Spirit/Spirit_Defines.h>
#include <data/Spin_System_Chain.hpp>
#include <data/State.hpp>
#include <vector>

namespace Utility
{
namespace Configuration_Chain
{

// Add noise to the images of a transition (except the border images)
void Add_Noise_Temperature( State::chain_t & c, int idx_1, int idx_2, scalar temperature );

// Homogeneous rotation of all spins from first to last configuration of the given configurations
void Homogeneous_Rotation( State::chain_t & c, int idx_1, int idx_2 );

} // namespace Configuration_Chain
} // namespace Utility

#endif
