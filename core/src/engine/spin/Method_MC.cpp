#include <Spirit/Spirit_Defines.h>
#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <engine/Indexing.hpp>
#include <engine/Vectormath.hpp>
#include <engine/spin/Method_MC.hpp>
#include <io/IO.hpp>
#include <utility/Constants.hpp>
#include <utility/Logging.hpp>

#include <Eigen/Dense>

#include <cmath>

using namespace Utility;

namespace Engine
{

namespace Spin
{

Method_MC::Method_MC( std::shared_ptr<system_t> system, int idx_img, int idx_chain )
        : Method( system->mc_parameters, idx_img, idx_chain )
{
    // Currently we only support a single image being iterated at once:
    this->system     = system;
    this->SenderName = Log_Sender::MC;

    this->noi           = 1;
    this->nos           = this->system->hamiltonian->get_geometry().nos;
    this->nos_nonvacant = this->system->hamiltonian->get_geometry().nos_nonvacant;

    this->xi = vectorfield( this->nos, { 0, 0, 0 } );

    // We assume it is not converged before the first iteration
    // this->max_torque = system->mc_parameters->force_convergence + 1.0;

    // History
    // this->history = std::map<std::string, std::vector<scalar>>{ { "max_torque", { this->max_torque } },
    //                                                             { "E", { this->max_torque } },
    //                                                             { "M_z", { this->max_torque } } };

    this->parameters_mc = system->mc_parameters;

    // Starting cone angle
    this->cone_angle               = Constants::Pi * this->parameters_mc->metropolis_cone_angle / 180.0;
    this->n_rejected               = 0;
    this->acceptance_ratio_current = this->parameters_mc->acceptance_ratio_target;
}

namespace
{

namespace Metropolis
{

template<typename Distribution, typename RandomFunc>
auto step_cone( Distribution && distribution, RandomFunc && prng, const Vector3 & spin, const scalar cone_angle )
    -> Vector3
{
    Matrix3 local_basis;
    // Calculate local basis for the spin
    if( spin.z() < 1 - 1e-10 )
    {
        local_basis.col( 2 ) = spin;
        local_basis.col( 0 ) = -Vector3{ 0, 0, 1 }.cross( spin ).normalized();
        local_basis.col( 1 ) = -spin.cross( local_basis.col( 0 ) );
    }
    else
    {
        local_basis = Matrix3::Identity();
    }

    // Rotation angle between 0 and cone_angle degrees
    const scalar costheta = 1 - ( 1 - std::cos( cone_angle ) ) * distribution( prng );

    const scalar sintheta = std::sqrt( 1 - costheta * costheta );

    // Random distribution of phi between 0 and 360 degrees
    const scalar phi = 2 * Constants::Pi * distribution( prng );

    // New spin orientation in local basis
    Vector3 local_spin_new{ sintheta * std::cos( phi ), sintheta * std::sin( phi ), costheta };

    // New spin orientation in regular basis
    return local_basis * local_spin_new;
};

template<typename Distribution, typename RandomFunc>
auto step_sphere( Distribution && distribution, RandomFunc && prng ) -> Vector3
{
    // Rotation angle between 0 and 180 degrees
    const scalar costheta = distribution( prng );

    const scalar sintheta = std::sqrt( 1 - costheta * costheta );

    // Random distribution of phi between 0 and 360 degrees
    const scalar phi = 2 * Constants::Pi * distribution( prng );

    // New spin orientation in local basis
    return { sintheta * std::cos( phi ), sintheta * std::sin( phi ), costheta };
};

} // namespace Metropolis

} // namespace

// This implementation is mostly serial as parallelization is nontrivial
//      if the range of neighbours for each atom is not pre-defined.
void Method_MC::Iteration()
{
    // Temporaries
    auto & state_old = *this->system->state;
    auto state_new   = state_old;

    // Generate randomly displaced spin configuration according to cone radius
    // Vectormath::get_random_vectorfield_unitsphere(this->parameters_mc->prng, random_unit_vectors);

    // TODO: add switch between Metropolis and heat bath
    // One Metropolis step
    Metropolis( state_old, state_new );
    Vectormath::set_c_a( 1, state_new.spin, state_old.spin );
}

void Method_MC::AdaptConeAngle()
{
    const scalar diff = 0.01;

    // Cone angle feedback algorithm
    if( this->parameters_mc->metropolis_step_cone && this->parameters_mc->metropolis_cone_adaptive )
    {
        this->acceptance_ratio_current = 1 - (scalar)this->n_rejected / (scalar)this->nos_nonvacant;

        if( ( this->acceptance_ratio_current < this->parameters_mc->acceptance_ratio_target )
            && ( this->cone_angle > diff ) )
            this->cone_angle -= diff;

        if( ( this->acceptance_ratio_current > this->parameters_mc->acceptance_ratio_target )
            && ( this->cone_angle < Constants::Pi - diff ) )
            this->cone_angle += diff;

        this->parameters_mc->metropolis_cone_angle = this->cone_angle * 180.0 / Constants::Pi;
    }
}

// Simple metropolis step
void Method_MC::Metropolis( const StateType & state_old, StateType & state_new )
{
    auto distribution     = std::uniform_real_distribution<scalar>( 0, 1 );
    auto distribution_idx = std::uniform_int_distribution<>( 0, this->nos - 1 );
    scalar kB_T           = Constants::k_B * this->parameters_mc->temperature;

    this->AdaptConeAngle();
    this->n_rejected = 0;

    // One Metropolis step for each spin
    // Loop over NOS samples (on average every spin should be hit once per Metropolis step)
    for( int idx = 0; idx < this->nos; ++idx )
    {
        int ispin;
        if( this->parameters_mc->metropolis_random_sample )
            // Better statistics, but additional calculation of random number
            ispin = distribution_idx( this->parameters_mc->prng );
        else
            // Faster, but worse statistics
            ispin = idx;

        if( Indexing::check_atom_type( this->system->hamiltonian->get_geometry().atom_types[ispin] ) )
        {
            // Sample a cone
            if( this->parameters_mc->metropolis_step_cone )
            {
                state_new.spin[ispin]
                    = Metropolis::step_cone( distribution, this->parameters_mc->prng, state_old.spin[ispin], cone_angle );
            }
            // Sample the entire unit sphere
            else
            {
                state_new.spin[ispin] = Metropolis::step_sphere( distribution, this->parameters_mc->prng );
            }

            // Energy difference of configurations with and without displacement
            const scalar Eold  = this->system->hamiltonian->Energy_Single_Spin( ispin, state_old );
            const scalar Enew  = this->system->hamiltonian->Energy_Single_Spin( ispin, state_new );
            const scalar Ediff = Enew - Eold;

            // Metropolis criterion: accept the step if the energy fell
            if( Ediff <= 1e-14 )
                continue;

            // potentially reject the step if energy rose
            if( this->parameters_mc->temperature < 1e-12 )
            {
                // Restore the spin
                state_new.spin[ispin] = state_old.spin[ispin];
                // Counter for the number of rejections
                ++this->n_rejected;
            }
            else
            {
                // Exponential factor
                scalar exp_ediff = std::exp( -Ediff / kB_T );
                // Metropolis random number
                scalar x_metropolis = distribution( this->parameters_mc->prng );

                // Only reject if random number is larger than exponential
                if( exp_ediff < x_metropolis )
                {
                    // Restore the spin
                    state_new.spin[ispin] = state_old.spin[ispin];
                    // Counter for the number of rejections
                    ++this->n_rejected;
                }
            }
        }
    }
}

// TODO:
// Implement heat bath algorithm, see Y. Miyatake et al, J Phys C: Solid State Phys 19, 2539 (1986)
// void Method_MC::HeatBath(const vectorfield & spins_old, vectorfield & spins_new)
// {
// }

void Method_MC::Hook_Pre_Iteration() {}

void Method_MC::Hook_Post_Iteration() {}

void Method_MC::Initialize() {}

void Method_MC::Finalize()
{
    this->system->iteration_allowed = false;
}

void Method_MC::lock()
{
    this->system->lock();
}

void Method_MC::unlock()
{
    this->system->unlock();
}

bool Method_MC::Iterations_Allowed()
{
    return this->system->iteration_allowed;
}

void Method_MC::Message_Start()
{
    //---- Log messages
    std::vector<std::string> block( 0 );
    block.emplace_back( fmt::format( "------------  Started  {} Calculation  ------------", this->Name() ) );
    block.emplace_back( fmt::format( "    Going to iterate {} step(s)", this->n_log ) );
    block.emplace_back( fmt::format( "                with {} iterations per step", this->n_iterations_log ) );
    if( this->parameters_mc->metropolis_step_cone )
    {
        if( this->parameters_mc->metropolis_cone_adaptive )
        {
            block.emplace_back(
                fmt::format( "   Target acceptance {:>6.3f}", this->parameters_mc->acceptance_ratio_target ) );
            block.emplace_back(
                fmt::format( "   Cone angle (deg): {:>6.3f} (adaptive)", this->cone_angle * 180 / Constants::Pi ) );
        }
        else
        {
            block.emplace_back(
                fmt::format( "   Target acceptance {:>6.3f}", this->parameters_mc->acceptance_ratio_target ) );
            block.emplace_back(
                fmt::format( "   Cone angle (deg): {:>6.3f} (non-adaptive)", this->cone_angle * 180 / Constants::Pi ) );
        }
    }
    block.emplace_back( "-----------------------------------------------------" );
    Log( Log_Level::All, this->SenderName, block, this->idx_image, this->idx_chain );
}

void Method_MC::Message_Step()
{
    // Update time of current step
    auto t_current = std::chrono::system_clock::now();

    // Update the system's energy
    this->system->UpdateEnergy();

    // Send log message
    std::vector<std::string> block( 0 );
    block.emplace_back(
        fmt::format( "----- {} Calculation: {}", this->Name(), Timing::DateTimePassed( t_current - this->t_start ) ) );
    block.emplace_back( fmt::format(
        "    Completed                 {} / {} step(s) (step size {})", this->step, this->n_log,
        this->n_iterations_log ) );
    block.emplace_back( fmt::format( "    Iteration                 {} / {}", this->iteration, this->n_iterations ) );
    block.emplace_back(
        fmt::format( "    Time since last step:     {}", Timing::DateTimePassed( t_current - this->t_last ) ) );
    block.emplace_back( fmt::format(
        "    Iterations / sec:         {}",
        this->n_iterations_log / Timing::SecondsPassed( t_current - this->t_last ) ) );
    if( this->parameters_mc->metropolis_step_cone )
    {
        if( this->parameters_mc->metropolis_cone_adaptive )
        {
            block.emplace_back( fmt::format(
                "    Current acceptance ratio: {:>6.3f} (target {})", this->acceptance_ratio_current,
                this->parameters_mc->acceptance_ratio_target ) );
            block.emplace_back( fmt::format(
                "    Current cone angle (deg): {:>6.3f} (adaptive)", this->cone_angle * 180 / Constants::Pi ) );
        }
        else
        {
            block.emplace_back(
                fmt::format( "    Current acceptance ratio: {:>6.3f}", this->acceptance_ratio_current ) );
            block.emplace_back( fmt::format(
                "    Current cone angle (deg): {:>6.3f} (non-adaptive)", this->cone_angle * 180 / Constants::Pi ) );
        }
    }
    block.emplace_back( fmt::format( "    Total energy:             {:20.10f}", this->system->E.total ) );
    Log( Log_Level::All, this->SenderName, block, this->idx_image, this->idx_chain );

    // Update time of last step
    this->t_last = t_current;
}

void Method_MC::Message_End()
{
    //---- End timings
    auto t_end = std::chrono::system_clock::now();

    //---- Termination reason
    std::string reason = "";
    if( this->StopFile_Present() )
        reason = "A STOP file has been found";
    else if( this->Walltime_Expired( t_end - this->t_start ) )
        reason = "The maximum walltime has been reached";

    // Update the system's energy
    this->system->UpdateEnergy();

    //---- Log messages
    std::vector<std::string> block;
    block.emplace_back( fmt::format( "------------ Terminated {} Calculation ------------", this->Name() ) );
    if( reason.length() > 0 )
        block.emplace_back( fmt::format( "----- Reason:   {}", reason ) );
    block.emplace_back( fmt::format( "----- Duration:       {}", Timing::DateTimePassed( t_end - this->t_start ) ) );
    block.emplace_back( fmt::format( "    Completed         {} / {} step(s)", this->step, this->n_log ) );
    block.emplace_back( fmt::format( "    Iteration         {} / {}", this->iteration, this->n_iterations ) );
    block.emplace_back(
        fmt::format( "    Iterations / sec: {}", this->iteration / Timing::SecondsPassed( t_end - this->t_start ) ) );
    if( this->parameters_mc->metropolis_step_cone )
    {
        if( this->parameters_mc->metropolis_cone_adaptive )
        {
            block.emplace_back( fmt::format(
                "    Acceptance ratio: {:>6.3f} (target {})", this->acceptance_ratio_current,
                this->parameters_mc->acceptance_ratio_target ) );
            block.emplace_back(
                fmt::format( "    Cone angle (deg): {:>6.3f} (adaptive)", this->cone_angle * 180 / Constants::Pi ) );
        }
        else
        {
            block.emplace_back( fmt::format( "    Acceptance ratio: {:>6.3f}", this->acceptance_ratio_current ) );
            block.emplace_back( fmt::format(
                "    Cone angle (deg): {:>6.3f} (non-adaptive)", this->cone_angle * 180 / Constants::Pi ) );
        }
    }
    block.emplace_back( fmt::format( "    Total energy:     {:20.10f}", this->system->E.total ) );
    block.emplace_back( "-----------------------------------------------------" );
    Log( Log_Level::All, this->SenderName, block, this->idx_image, this->idx_chain );
}

void Method_MC::Save_Current( std::string starttime, int iteration, bool initial, bool final ) {}

// Method name as string
std::string_view Method_MC::Name()
{
    return "MC";
}

} // namespace Spin

} // namespace Engine
