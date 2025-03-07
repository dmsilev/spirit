#pragma once
#include <engine/spin/Method_Solver.hpp>

namespace Engine
{

namespace Spin
{

template<>
class SolverData<Solver::SIB> : public SolverMethods
{
protected:
    using SolverMethods::Calculate_Force;
    using SolverMethods::Calculate_Force_Virtual;
    using SolverMethods::Prepare_Thermal_Field;
    using SolverMethods::SolverMethods;
    // Actual Forces on the configurations
    std::vector<vectorfield> forces_predictor;
    // Virtual Forces used in the Steps
    std::vector<vectorfield> forces_virtual_predictor;

    std::vector<std::shared_ptr<StateType>> configurations_predictor;
};

template<>
inline void Method_Solver<Solver::SIB>::Initialize()
{
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos ) ); // [noi][nos]
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos ) ); // [noi][nos]

    this->forces_predictor         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual_predictor = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );

    this->configurations_predictor = std::vector<std::shared_ptr<StateType>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        configurations_predictor[i] = std::make_shared<StateType>( make_state<StateType>( this->nos ) );
}

/*
    Template instantiation of the Simulation class for use with the SIB Solver.
        The semi-implicit method B is an efficient midpoint solver.
    Paper: J. H. Mentink et al., Stable and fast semi-implicit integration of the stochastic
           Landau-Lifshitz equation, J. Phys. Condens. Matter 22, 176001 (2010).
*/
template<>
inline void Method_Solver<Solver::SIB>::Iteration()
{
    // Generate random vectors for this iteration
    this->Prepare_Thermal_Field();

    // First part of the step
    this->Calculate_Force( this->configurations, this->forces );
    this->Calculate_Force_Virtual( this->configurations, this->forces, this->forces_virtual );
    for( int i = 0; i < this->noi; ++i )
    {
        auto & image     = this->configurations[i]->spin;
        auto & predictor = this->configurations_predictor[i]->spin;

        Solver_Kernels::sib_transform( image, forces_virtual[i], predictor );
        Backend::transform(
            SPIRIT_PAR predictor.begin(), predictor.end(), image.begin(), predictor.begin(),
            [] SPIRIT_LAMBDA( const Vector3 & predictor, const Vector3 & image )
            { return 0.5 * ( predictor + image ); } );
    }

    // Second part of the step
    this->Calculate_Force( this->configurations_predictor, this->forces_predictor );
    this->Calculate_Force_Virtual(
        this->configurations_predictor, this->forces_predictor, this->forces_virtual_predictor );
    for( int i = 0; i < this->noi; ++i )
    {
        auto & image = this->configurations[i]->spin;

        Solver_Kernels::sib_transform( image, forces_virtual_predictor[i], image );
    }
}

} // namespace Spin

} // namespace Engine
