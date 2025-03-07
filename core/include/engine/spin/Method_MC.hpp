#pragma once
#ifndef SPIRIT_CORE_ENGINE_SPIN_METHOD_MC_HPP
#define SPIRIT_CORE_ENGINE_SPIN_METHOD_MC_HPP

#include <Spirit/Simulation.h>
#include <Spirit/Spirit_Defines.h>
#include <data/Spin_System.hpp>
#include <engine/spin/Method.hpp>

namespace Engine
{

namespace Spin
{

enum struct MC_Algorithm
{
    None           = -1,
    Metropolis     = MC_Algorithm_Metropolis,
    Metropolis_MDC = MC_Algorithm_Metropolis_MDC,
};

/*
    The Monte Carlo method
*/
template<MC_Algorithm algorithm>
class Method_MC : public Method
{
public:
    // Constructor
    Method_MC( std::shared_ptr<system_t> system, int idx_img, int idx_chain );

    // Method name as string
    std::string_view Name() override;

private:
    // Solver_Iteration represents one iteration of a certain Solver
    void Iteration() override;

    // Metropolis iteration with adaptive cone radius
    void Step( StateType & spins, Hamiltonian & hamiltonian );

    // Save the current Step's Data: spins and energy
    void Save_Current( std::string starttime, int iteration, bool initial = false, bool final = false ) override;
    // A hook into the Method before an Iteration of the Solver
    void Hook_Pre_Iteration() override;
    // A hook into the Method after an Iteration of the Solver
    void Hook_Post_Iteration() override;

    // Sets iteration_allowed to false for the corresponding method
    void Initialize() override;
    // Sets iteration_allowed to false for the corresponding method
    void Finalize() override;

    // Log message blocks
    void Message_Start() override;
    void Message_Step() override;
    void Message_End() override;

    // Lock systems in order to prevent otherwise access
    void lock() override;
    // Unlock systems to re-enable access
    void unlock() override;
    // Check if iterations are allowed
    bool Iterations_Allowed() override;

    // Systems the Solver will access
    std::shared_ptr<system_t> system;

    std::shared_ptr<Data::Parameters_Method_MC> parameters_mc;

    // Cosine of current cone angle
    scalar cone_angle;
    int n_rejected;
    scalar acceptance_ratio_current;
    int nos_nonvacant;

    // constrained direction for the direction constrained monte carlo algorithm
    Vector3 constrained_direction{ 0, 0, 1 };
    Matrix3 constrained_orthogonal_projector = []
    {
        Matrix3 m;
        m << 1, 0, 0, 0, 1, 0, 0, 0, 0;
        return m;
    }();

    // Random vector array
    vectorfield xi;
};

} // namespace Spin

} // namespace Engine

#endif
