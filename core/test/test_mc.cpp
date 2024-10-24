#include <Spirit/Configurations.h>
#include <Spirit/Constants.h>
#include <Spirit/Geometry.h>
#include <Spirit/Hamiltonian.h>
#include <Spirit/Parameters_MC.h>
#include <Spirit/Quantities.h>
#include <Spirit/Simulation.h>
#include <Spirit/State.h>
#include <Spirit/System.h>
#include <Spirit/Version.h>
#include <data/State.hpp>
#include <engine/Vectormath.hpp>

#include "catch.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>

using Catch::Matchers::WithinAbs;

// Reduce required precision if float accuracy
#ifdef SPIRIT_SCALAR_TYPE_DOUBLE
[[maybe_unused]] constexpr scalar epsilon_2 = 1e-5;
[[maybe_unused]] constexpr scalar epsilon_3 = 1e-6;
[[maybe_unused]] constexpr scalar epsilon_4 = 1e-7;
[[maybe_unused]] constexpr scalar epsilon_5 = 1e-8;
[[maybe_unused]] constexpr scalar epsilon_6 = 1e-9;
#else
[[maybe_unused]] constexpr scalar epsilon_2 = 1e-2;
[[maybe_unused]] constexpr scalar epsilon_3 = 1e-3;
[[maybe_unused]] constexpr scalar epsilon_4 = 1e-4;
[[maybe_unused]] constexpr scalar epsilon_5 = 1e-5;
[[maybe_unused]] constexpr scalar epsilon_6 = 1e-6;
#endif

TEST_CASE( "Direction Constrained Monte Carlo should preserve direction", "[mc]" )
{
    constexpr auto input_file = "core/test/input/mc.cfg";

    // Set up the initial direction of the spins
    auto state                   = std::shared_ptr<State>( State_Setup( input_file ), State_Delete );
    const Vector3 init_direction = Vector3{ 0.99, 0., 0.1 }.normalized();
    Configuration_Domain( state.get(), init_direction.data() );

    // capture the initial magnetization of the system
    Vector3 init_magnetization = Vector3::Zero();
    Quantity_Get_Magnetization( state.get(), init_magnetization.data() );

    REQUIRE_THAT( init_magnetization.norm(), WithinAbs( init_magnetization.dot( init_direction ), epsilon_6 ) );

    const Vector3 magnetization_direction = init_magnetization.normalized();

    Parameters_MC_Set_Metropolis_Cone( state.get(), true, 30, false, 0.5 );

    Simulation_MC_Start( state.get(), MC_Algorithm_Metropolis_MDC, -1, -1, true );

    {
        Vector3 magnetization = Vector3::Zero();
        for( int i = 0; i < 10; ++i )
        {
            Simulation_N_Shot( state.get(), 50 );
            Quantity_Get_Magnetization( state.get(), magnetization.data() );

            const scalar m_para = magnetization_direction.dot( magnetization );
            const scalar m_orth
                = ( ( Matrix3::Identity() - magnetization_direction * magnetization_direction.transpose() )
                    * magnetization )
                      .norm();

            INFO( "Iteration: " << i );
            INFO( "M_para = " << m_para );
            INFO( "M_orth = " << m_orth );

            REQUIRE_THAT( m_orth, WithinAbs( 0, epsilon_6 ) );
        }
    }

    Simulation_Stop( state.get() );
}
