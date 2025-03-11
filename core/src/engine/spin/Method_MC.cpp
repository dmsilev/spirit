#include <Spirit/Spirit_Defines.h>
#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <engine/Indexing.hpp>
#include <engine/Vectormath.hpp>
#include <engine/common/Method_MC.hpp>
#include <engine/spin/Method_MC.hpp>
#include <io/HDF5_File.hpp>
#include <io/IO.hpp>
#include <io/OVF_File.hpp>
#include <io/VTK_Geometry.hpp>
#include <io/XML_File.hpp>
#include <utility/Constants.hpp>
#include <utility/Logging.hpp>
#include <utility/Version.hpp>

#include <Eigen/Dense>
#include <fmt/format.h>

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
            = Vectormath::Magnetization( this->system->state->spin, this->system->hamiltonian->get_geometry().mu_s )
                  .normalized();
        if( m_direction.squaredNorm() > 1e-4 )
        {
            constrained_direction = m_direction;
            constrained_orthogonal_projector
                = Matrix3::Identity() - constrained_direction * constrained_direction.transpose();
        }
        else
        {
            constrained_direction = Vector3{ 0, 0, 1 };
            constrained_orthogonal_projector << 1, 0, 0, 0, 1, 0, 0, 0, 0;
        }

        assert( ( this->constrained_orthogonal_projector * this->constrained_direction ).norm() < 1e-12 );
    }
}

// This implementation is mostly serial as parallelization is nontrivial
//      if the range of neighbours for each atom is not pre-defined.
template<MC_Algorithm algorithm>
void Method_MC<algorithm>::Iteration()
{
    // Temporaries
    auto & state_old = *this->system->state;
    auto state_new   = state_old;

    const scalar diff = 0.01;

    this->acceptance_ratio_current = 1 - (scalar)this->n_rejected / (scalar)this->nos_nonvacant;

    // Cone angle feedback algorithm
    if( this->parameters_mc->metropolis_step_cone && this->parameters_mc->metropolis_cone_adaptive )
    {
        if( ( this->acceptance_ratio_current < this->parameters_mc->acceptance_ratio_target )
            && ( this->cone_angle > diff ) )
            this->cone_angle -= diff;

        if( ( this->acceptance_ratio_current > this->parameters_mc->acceptance_ratio_target )
            && ( this->cone_angle < Constants::Pi - diff ) )
            this->cone_angle += diff;

        this->parameters_mc->metropolis_cone_angle = this->cone_angle * 180.0 / Constants::Pi;
    }
    this->n_rejected = 0;

    // One Metropolis step
    Step( state_new, *this->system->hamiltonian );

    Vectormath::set_c_a( 1, state_new.spin, state_old.spin );
}

template<>
void Method_MC<MC_Algorithm::None>::Step( StateType &, Hamiltonian & ) {};

// Simple metropolis step
template<>
void Method_MC<MC_Algorithm::Metropolis>::Step( StateType & state, Hamiltonian & hamiltonian )
{
    struct SharedData
    {
        int n_rejected;
        Data::Parameters_Method_MC & parameters_mc;
        std::uniform_real_distribution<scalar> distribution;
        std::uniform_int_distribution<int> distribution_idx;
        const scalar beta;
        const scalar cone_angle;
    };

    SharedData shared = SharedData{ /*n_rejected=*/0,
                                    /*parameters_mc=*/*this->parameters_mc,
                                    /*distribution=*/std::uniform_real_distribution<scalar>( 0, 1 ),
                                    /*distribution_idx=*/std::uniform_int_distribution<>( 0, this->nos - 1 ),
                                    /*beta=*/scalar( 1.0 ) / ( Constants::k_B * this->parameters_mc->temperature ),
                                    /*cone_angle=*/this->cone_angle };

    // One Metropolis step for each spin
    // Loop over NOS samples (on average every spin should be hit once per Metropolis step)
    for( int idx = 0; idx < this->nos; ++idx )
    {
        Common::Metropolis::trial_spin( idx, state, hamiltonian, shared );
    }

    this->n_rejected = shared.n_rejected;
}

/* Directon Constrained Metropolis Monte Carlo Algorithm*/
template<>
void Method_MC<MC_Algorithm::Metropolis_MDC>::Step( StateType & state, Hamiltonian & hamiltonian )
{
    struct SharedData
    {
        int n_rejected;
        Data::Parameters_Method_MC & parameters_mc;
        std::uniform_real_distribution<scalar> distribution;
        std::uniform_int_distribution<int> distribution_idx;
        const scalar beta;
        const scalar cone_angle;
        const Vector3 para_projector;
        const Matrix3 orth_projector;
        scalar magnetization_pre;
    };

    const auto & geometry = hamiltonian.get_geometry();
    SharedData shared     = SharedData{
        /*n_rejected=*/0,
        /*parameters_mc=*/*this->parameters_mc,
        /*distribution=*/std::uniform_real_distribution<scalar>( 0, 1 ),
        /*distribution_idx=*/std::uniform_int_distribution<>( 0, this->nos - 1 ),
        /*beta=*/scalar( 1.0 ) / ( Constants::k_B * this->parameters_mc->temperature ),
        /*cone_angle=*/this->cone_angle,
        /* para_projector=*/constrained_direction,
        /* orth_projector=*/constrained_orthogonal_projector,
        /* magnetization_pre=*/constrained_direction.dot( Vectormath::Magnetization( state.spin, geometry.mu_s ) )
    };

    if( shared.magnetization_pre < 0 )
        spirit_throw(
            Exception_Classifier::Unknown_Exception, Log_Level::Error,
            "Initial magnetization is not aligned with constraint direction." );

    // One Metropolis step for each spin
    // Loop over NOS samples (on average every spin should be hit once per Metropolis step)
    for( int idx = 0; idx < this->nos; ++idx )
    {
        Common::Metropolis::trial_spin_magnetization_constrained( idx, state, hamiltonian, shared );
    }

    this->n_rejected = shared.n_rejected;
}

// TODO:
// Implement heat bath algorithm, see Y. Miyatake et al, J Phys C: Solid State Phys 19, 2539 (1986)
// template<MC_Algorithm::HeatBath>
// void Method_MC::Step( vectorfield & spins, Hamiltonian & hamiltonian )
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
        if( !this->parameters_mc->metropolis_random_sample )
            Log(
                Log_Level::Warning, this->SenderName,
                "Using the direction constrained metropolis algorithm without random sampling is strongly discouraged.",
                this->idx_image, this->idx_chain );
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
    if( this->system == nullptr )
        return;
    auto & sys = *this->system;

    // History save
    this->history_iteration.push_back( this->iteration );
    this->history_max_torque.push_back( this->max_torque );
    this->history_energy.push_back( sys.E.total );

    // this->history["max_torque"].push_back( this->max_torque );
    // sys.update_energy();
    // this->history["E"].push_back( sys.E );
    // Removed magnetization, since at the moment it required a temporary allocation to compute
    // auto mag = Engine::Vectormath::Magnetization( *sys.spins );
    // this->history["M_z"].push_back( mag[2] );

    // File save
    if( this->parameters->output_any )
    {
        // Convert indices to formatted strings
        auto s_img         = fmt::format( "{:0>2}", this->idx_image );
        auto base          = static_cast<std::int32_t>( log10( this->parameters->n_iterations ) );
        std::string s_iter = fmt::format( fmt::runtime( "{:0>" + fmt::format( "{}", base ) + "}" ), iteration );

        std::string preSpinsFile;
        std::string preEnergyFile;
        std::string fileTag;

        if( sys.llg_parameters->output_file_tag == "<time>" )
            fileTag = starttime + "_";
        else if( sys.llg_parameters->output_file_tag != "" )
            fileTag = sys.llg_parameters->output_file_tag + "_";
        else
            fileTag = "";

        preSpinsFile  = this->parameters->output_folder + "/" + fileTag + "Image-" + s_img + "_Spins";
        preEnergyFile = this->parameters->output_folder + "/" + fileTag + "Image-" + s_img + "_Energy";

        // Function to write or append image and energy files
        auto writeOutputConfiguration = [this, &sys, preSpinsFile, iteration]( const std::string & suffix, bool append )
        {
            try
            {
                // File name and comment
                std::string spinsFile      = preSpinsFile + suffix;
                std::string output_comment = fmt::format(
                    "{} simulation ({} solver)\n# Desc:      Iteration: {}\n# Desc:      Maximum torque: {}",
                    this->Name(), this->SolverFullName(), iteration, this->max_torque );

                // File format
                switch( IO::VF_FileFormat format = sys.llg_parameters->output_vf_filetype )
                {
                    case IO::VF_FileFormat::OVF_BIN:
                    case IO::VF_FileFormat::OVF_BIN4:
                    case IO::VF_FileFormat::OVF_BIN8:
                    case IO::VF_FileFormat::OVF_TEXT:
                    case IO::VF_FileFormat::OVF_CSV:
                    {
                        // Spin Configuration
                        auto & system_state = *sys.state;
                        auto segment        = IO::OVF_Segment( sys.hamiltonian->get_geometry() );
                        std::string title   = fmt::format( "SPIRIT Version {}", Utility::version_full );
                        segment.title       = strdup( title.c_str() );
                        segment.comment     = strdup( output_comment.c_str() );
                        segment.valuedim    = IO::Spin::State::valuedim;
                        segment.valuelabels = strdup( IO::Spin::State::valuelabels.data() );
                        segment.valueunits  = strdup( IO::Spin::State::valueunits.data() );

                        const IO::Spin::State::Buffer buffer( system_state );
                        if( append )
                            IO::OVF_File( spinsFile + ".ovf" )
                                .append_segment( segment, buffer.data(), static_cast<int>( format ) );
                        else
                            IO::OVF_File( spinsFile + ".ovf" )
                                .write_segment( segment, buffer.data(), static_cast<int>( format ) );
                        break;
                    }
                    case IO::VF_FileFormat::VTK_HDF:
                    {
                        // TODO: store this somewhere (e.g. with the method), because creating it is fairly expensive
                        IO::VTK::UnstructuredGrid vtk_geometry( sys.hamiltonian->get_geometry() );
                        if( append )
                            spirit_throw(
                                Exception_Classifier::Not_Implemented, Log_Level::Error,
                                "Append not implemented for VTKHDF format!" );
                        else
                            IO::HDF5::write_fields(
                                spinsFile + ".vtkhdf", vtk_geometry,
                                { IO::VTK::FieldDescriptor{ "spins", &get<Field::Spin>( *sys.state ) } } );
                        break;
                    }
                    case IO::VF_FileFormat::VTK_XML_TEXT:
                    case IO::VF_FileFormat::VTK_XML_BIN:
                    {
                        // TODO: store this somewhere (e.g. with the method), because creating it is fairly expensive
                        IO::VTK::UnstructuredGrid vtk_geometry( sys.hamiltonian->get_geometry() );
                        if( append )
                            spirit_throw(
                                Exception_Classifier::Not_Implemented, Log_Level::Error,
                                "Append not implemented for VTK format!" );
                        else
                            IO::XML::write_fields(
                                spinsFile + ".vtu", vtk_geometry, format,
                                { IO::VTK::FieldDescriptor{ "spins", &get<Field::Spin>( *sys.state ) } } );
                        break;
                    }
                    default:
                        spirit_throw(
                            Exception_Classifier::Not_Implemented, Log_Level::Error,
                            fmt::format(
                                "\"writeOutputConfiguration()\" not implemented for file format: {}", str( format ) ) );
                }
            }
            catch( ... )
            {
                spirit_handle_exception_core( "LLG output failed" );
            }
        };

        IO::Flags flags;
        if( sys.llg_parameters->output_energy_divide_by_nspins )
            flags |= IO::Flag::Normalize_by_nos;
        if( sys.llg_parameters->output_energy_add_readability_lines )
            flags |= IO::Flag::Readability;
        auto writeOutputEnergy = [&sys, flags, preEnergyFile, iteration]( const std::string & suffix, bool append )
        {
            // File name
            std::string energyFile        = preEnergyFile + suffix + ".txt";
            std::string energyFilePerSpin = preEnergyFile + "-perSpin" + suffix + ".txt";

            // Energy
            if( append )
            {
                // Check if Energy File exists and write Header if it doesn't
                std::ifstream f( energyFile );
                if( !f.good() )
                    IO::Write_Energy_Header(
                        sys.E, energyFile, { "iteration", "E_tot" }, IO::Flag::Contributions | flags );
                // Append Energy to File
                IO::Append_Image_Energy( sys.E, sys.hamiltonian->get_geometry(), iteration, energyFile, flags );
            }
            else
            {
                IO::Write_Energy_Header( sys.E, energyFile, { "iteration", "E_tot" }, IO::Flag::Contributions | flags );
                IO::Append_Image_Energy( sys.E, sys.hamiltonian->get_geometry(), iteration, energyFile, flags );
                if( sys.llg_parameters->output_energy_spin_resolved )
                {
                    // Gather the data
                    Data::vectorlabeled<scalarfield> contributions_spins( 0 );
                    sys.update_energy();
                    sys.hamiltonian->Energy_Contributions_per_Spin( *sys.state, sys.E.per_interaction_per_spin );

                    IO::Write_Image_Energy_Contributions(
                        sys.E, sys.hamiltonian->get_geometry(), energyFilePerSpin,
                        sys.llg_parameters->output_vf_filetype );
                }
            }
        };

        // Initial image before simulation
        if( initial && this->parameters->output_initial )
        {
            writeOutputConfiguration( "-initial", false );
            writeOutputEnergy( "-initial", false );
        }
        // Final image after simulation
        else if( final && this->parameters->output_final )
        {
            writeOutputConfiguration( "-final", false );
            writeOutputEnergy( "-final", false );
        }

        // Single file output
        if( sys.llg_parameters->output_configuration_step )
        {
            writeOutputConfiguration( "_" + s_iter, false );
        }
        if( sys.llg_parameters->output_energy_step )
        {
            writeOutputEnergy( "_" + s_iter, false );
        }

        // Archive file output (appending)
        if( sys.llg_parameters->output_configuration_archive )
        {
            writeOutputConfiguration( "-archive", true );
        }
        if( sys.llg_parameters->output_energy_archive )
        {
            writeOutputEnergy( "-archive", true );
        }

        // Save Log
        Log.Append_to_File();
    }
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
