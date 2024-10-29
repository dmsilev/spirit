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

template<MC_Algorithm algorithm>
Method_MC<algorithm>::Method_MC( std::shared_ptr<system_t> system, int idx_img, int idx_chain )
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

    // fix current magnetization direction
    if constexpr( algorithm == MC_Algorithm::Metropolis_MDC )
    {
        // caclulate magnetization directly to avoid stale cache data
        const auto m_direction
            = Vectormath::Magnetization( *this->system->state, this->system->hamiltonian->get_geometry().mu_s )
                  .normalized();
        if( m_direction.squaredNorm() > 1e-4 )
            constrained_direction = m_direction;
        else
            constrained_direction = Vector3{ 0, 0, 1 };
    }
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
template<MC_Algorithm algorithm>
void Method_MC<algorithm>::Iteration()
{
    // Temporaries
    auto & state_old = *this->system->state;
    auto state_new   = state_old;

    // One Metropolis step
    Step( state_new, *this->system->hamiltonian );

    Vectormath::set_c_a( 1, state_new.spin, state_old.spin );
}

template<MC_Algorithm algorithm>
void Method_MC<algorithm>::AdaptConeAngle()
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
template<>
void Method_MC<MC_Algorithm::None>::Step( StateType &, HamiltonianVariant & ) {};

// Simple metropolis step
template<>
void Method_MC<MC_Algorithm::Metropolis>::Step( StateType & state, HamiltonianVariant & hamiltonian )
{
    auto distribution     = std::uniform_real_distribution<scalar>( 0, 1 );
    auto distribution_idx = std::uniform_int_distribution<>( 0, this->nos - 1 );
    scalar kB_T           = Constants::k_B * this->parameters_mc->temperature;

    const auto & geometry = hamiltonian.get_geometry();

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

        if( !Indexing::check_atom_type( geometry.atom_types[ispin] ) )
            continue;

        const Vector3 spin_pre  = state.spin[ispin];
        const Vector3 spin_post = [this, &distribution, &spin_pre]
        {
            // Sample a cone
            if( this->parameters_mc->metropolis_step_cone )
            {
                return Metropolis::step_cone( distribution, this->parameters_mc->prng, spin_pre, this->cone_angle );
            }
            // Sample the entire unit sphere
            else
            {
                return Metropolis::step_sphere( distribution, this->parameters_mc->prng );
            }
        }();

        // Energy difference of configurations with and without displacement
        const scalar E_pre  = hamiltonian.Energy_Single_Spin( ispin, state );
        state.spin[ispin]        = spin_post;
        const scalar E_post = hamiltonian.Energy_Single_Spin( ispin, state );
        const scalar Ediff  = E_post - E_pre;

        // Metropolis criterion: accept the step if the energy fell
        if( Ediff <= 1e-14 )
            continue;

        // potentially reject the step if energy rose
        if( this->parameters_mc->temperature < 1e-12 )
        {
            // Restore the spin
            state.spin[ispin] = spin_pre;
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
                state.spin[ispin] = spin_pre;
                // Counter for the number of rejections
                ++this->n_rejected;
            }
        }
    }
}

/* Directon Constrained Metropolis Monte Carlo Algorithm following:
 * https://link.aps.org/doi/10.1103/PhysRevB.82.054415
 * This algorithm extends the one described in the paper by also taking
 * a variable `mu_s` into account.
 */
template<>
void Method_MC<MC_Algorithm::Metropolis_MDC>::Step( StateType & state, HamiltonianVariant & hamiltonian )
{
    auto distribution     = std::uniform_real_distribution<scalar>( 0, 1 );
    auto distribution_idx = std::uniform_int_distribution<>( 0, this->nos - 1 );
    const scalar beta     = scalar( 1.0 ) / ( Constants::k_B * this->parameters_mc->temperature );

    const auto & geometry        = hamiltonian.get_geometry();
    const Vector3 para_projector = constrained_direction;
    const Matrix3 orth_projector = Matrix3::Identity() - para_projector * para_projector.transpose();

    this->AdaptConeAngle();
    this->n_rejected = 0;

    if( !this->parameters_mc->metropolis_random_sample )
        Log( Log_Level::Warning, this->SenderName,
             "Using the direction constrained metropolis algorithm without random sampling is strongly discouraged.",
             this->idx_image, this->idx_chain );

    scalar magnetization_pre = para_projector.dot( Vectormath::Magnetization( state.spin, geometry.mu_s ) );
    if( magnetization_pre < 0 )
        spirit_throw(
            Exception_Classifier::Unknown_Exception, Log_Level::Error,
            "Initial magnetization is not aligned with constraint direction." );

    // One Metropolis step for each spin
    // Loop over NOS samples (on average every spin should be hit once per Metropolis step)
    for( int idx = 0; idx < this->nos; ++idx )
    {
        // changed spin and counteracting spin
        int ispin, jspin;
        if( this->parameters_mc->metropolis_random_sample )
        {
            // Better statistics, but additional calculation of random number
            ispin = distribution_idx( this->parameters_mc->prng );
            if( !Indexing::check_atom_type( geometry.atom_types[ispin] ) )
                continue;

            // try to find a second spin to compensate
            {
                const int max_retry = std::max( this->nos, 10 );
                int try_idx         = 0;
                for( ; try_idx < max_retry; ++try_idx )
                {
                    jspin = distribution_idx( this->parameters_mc->prng );
                    if( jspin != ispin && Indexing::check_atom_type( geometry.atom_types[jspin] ) )
                        break;
                }
                if( try_idx >= max_retry )
                    spirit_throw(
                        Exception_Classifier::Unknown_Exception, Log_Level::Error,
                        "Cannot find a random second compensation spin! (too many retries)" );
            }
        }
        else
        {
            // Faster, but much worse statistics
            ispin = idx;
            if( !Indexing::check_atom_type( geometry.atom_types[ispin] ) )
                continue;
            const int max_retry = this->nos;
            int try_idx         = 0;
            jspin               = ( idx + this->nos / 2 ) % this->nos;
            for( ; try_idx < max_retry; ++try_idx )
            {
                if( jspin != ispin && Indexing::check_atom_type( geometry.atom_types[jspin] ) )
                    break;
                jspin = ( jspin + 1 ) % this->nos;
            }
            if( try_idx >= max_retry )
                spirit_throw(
                    Exception_Classifier::Unknown_Exception, Log_Level::Error,
                    "Cannot find a second compensation spin!" );
        }

        // At this point both chosen atoms should have a valid atom type
        // NOTE: Should atoms of different type serve as compensation for each other?
        assert(
            Indexing::check_atom_type( geometry.atom_types[ispin] )
            && Indexing::check_atom_type( geometry.atom_types[jspin] ) );

        const scalar mu_s_ratio = geometry.mu_s[ispin] / geometry.mu_s[jspin];
        const auto spin_i_pre   = state.spin[ispin];
        const auto spin_j_pre   = state.spin[jspin];
        const auto spin_i_post  = [this, &distribution, &spin_i_pre]
        {
            // Sample a cone
            if( this->parameters_mc->metropolis_step_cone )
            {
                return Metropolis::step_cone( distribution, this->parameters_mc->prng, spin_i_pre, this->cone_angle );
            }
            // Sample the entire unit sphere
            else
            {
                return Metropolis::step_sphere( distribution, this->parameters_mc->prng );
            }
        }();

        // calculate compensation spin
        Vector3 spin_j_post     = orth_projector * ( spin_j_pre + mu_s_ratio * ( spin_i_pre - spin_i_post ) );
        const scalar sz_squared = 1 - spin_j_post.squaredNorm();
        if( sz_squared < 0 )
        {
            this->n_rejected++;
            continue;
        }

        const scalar sz_pre  = spin_j_pre.dot( para_projector );
        const scalar sz_post = std::copysign( std::sqrt( sz_squared ), sz_pre );
        spin_j_post += sz_post * para_projector;

        // calculate magnetization
        const scalar magnetization_post = magnetization_pre
                                          + ( geometry.mu_s[ispin] * para_projector.dot( spin_i_post - spin_i_pre )
                                              + geometry.mu_s[jspin] * ( sz_post - sz_pre ) )
                                                / static_cast<scalar>( state.spin.size() );

        if( magnetization_post < 0 )
        {
            this->n_rejected++;
            continue;
        }

        // Energy difference of configurations with and without displacement
        // The `Energy_Single_Spin` method assumes changes to a single spin.
        // To determine the change in energy from the full change correctly
        // we have to evaluate the differences from each change individually.
        const scalar Ediff = [&hamiltonian, ispin, jspin, &spin_i_post, &spin_j_post, &state]
        {
            scalar Ediff = -1.0 * hamiltonian.Energy_Single_Spin( ispin, state );
            state.spin[ispin] = spin_i_post;
            Ediff += hamiltonian.Energy_Single_Spin( ispin, state ) - hamiltonian.Energy_Single_Spin( jspin, state );
            state.spin[jspin] = spin_j_post;
            Ediff += hamiltonian.Energy_Single_Spin( jspin, state );
            return Ediff;
        }();

        const scalar jacobian_pre  = magnetization_pre * magnetization_pre / std::abs( sz_pre );
        const scalar jacobian_post = magnetization_post * magnetization_post / std::abs( sz_post );
        const scalar metropolis_pb = jacobian_post / jacobian_pre * std::exp( -Ediff * beta );

        // Metropolis criterion: reject the step with probability (1 - min(1, metropolis_pb))
        if( ( metropolis_pb < 1e-12 )
            || ( metropolis_pb < 1.0 && metropolis_pb < distribution( this->parameters_mc->prng ) ) )
        {
            // Restore the spin
            state.spin[ispin] = spin_i_pre;
            state.spin[jspin] = spin_j_pre;
            // Counter for the number of rejections
            ++this->n_rejected;
            continue;
        }

        magnetization_pre = magnetization_post;
    }
}

// TODO:
// Implement heat bath algorithm, see Y. Miyatake et al, J Phys C: Solid State Phys 19, 2539 (1986)
// template<MC_Algorithm::HeatBath>
// void Method_MC::Step( vectorfield & spins, HamiltonianVariant & hamiltonian )
// {
// }

template<MC_Algorithm algorithm>
void Method_MC<algorithm>::Hook_Pre_Iteration()
{
}

template<MC_Algorithm algorithm>
void Method_MC<algorithm>::Hook_Post_Iteration()
{
}

template<MC_Algorithm algorithm>
void Method_MC<algorithm>::Initialize()
{
}

template<MC_Algorithm algorithm>
void Method_MC<algorithm>::Finalize()
{
    this->system->iteration_allowed = false;
}

template<MC_Algorithm algorithm>
void Method_MC<algorithm>::lock()
{
    this->system->lock();
}

template<MC_Algorithm algorithm>
void Method_MC<algorithm>::unlock()
{
    this->system->unlock();
}

template<MC_Algorithm algorithm>
bool Method_MC<algorithm>::Iterations_Allowed()
{
    return this->system->iteration_allowed;
}

template<MC_Algorithm algorithm>
void Method_MC<algorithm>::Message_Start()
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
    if constexpr( algorithm == MC_Algorithm::Metropolis_MDC )
    {
        block.emplace_back( fmt::format( "   constrained direction: {}", this->constrained_direction.transpose() ) );
    }
    block.emplace_back( "-----------------------------------------------------" );
    Log( Log_Level::All, this->SenderName, block, this->idx_image, this->idx_chain );
}

template<MC_Algorithm algorithm>
void Method_MC<algorithm>::Message_Step()
{
    // Update time of current step
    auto t_current = std::chrono::system_clock::now();

    // Update the system's energy
    this->system->update_energy();

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

template<MC_Algorithm algorithm>
void Method_MC<algorithm>::Message_End()
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
    this->system->update_energy();

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

template<MC_Algorithm algorithm>
void Method_MC<algorithm>::Save_Current( std::string starttime, int iteration, bool initial, bool final )
{
}

// Method name as string
template<MC_Algorithm algorithm>
std::string_view Method_MC<algorithm>::Name()
{
    return "MC";
}

template<>
std::string_view Method_MC<MC_Algorithm::None>::Name()
{
    return "MC (None)";
}

template<>
std::string_view Method_MC<MC_Algorithm::Metropolis>::Name()
{
    return "MC (Metropolis)";
}

template<>
std::string_view Method_MC<MC_Algorithm::Metropolis_MDC>::Name()
{
    return "MC (Magnetization Direction Constrained Metropolis)";
}

template class Method_MC<MC_Algorithm::None>;
template class Method_MC<MC_Algorithm::Metropolis>;
template class Method_MC<MC_Algorithm::Metropolis_MDC>;

} // namespace Spin

} // namespace Engine
