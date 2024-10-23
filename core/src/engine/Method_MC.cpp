#include <Spirit_Defines.h>
#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <engine/Method_MC.hpp>
#include <engine/Vectormath.hpp>
#include <io/IO.hpp>
#include <io/OVF_File.hpp>
#include <utility/Constants.hpp>
#include <utility/Logging.hpp>
#include <utility/Version.hpp>

#include <Eigen/Dense>

#include <cmath>
#include <ctime>
#include <iostream>

using namespace Utility;

namespace Engine
{

Method_MC::Method_MC( std::shared_ptr<Data::Spin_System> system, int idx_img, int idx_chain )
        : Method( system->mc_parameters, idx_img, idx_chain )
{
    // Currently we only support a single image being iterated at once:
    this->systems    = std::vector<std::shared_ptr<Data::Spin_System>>( 1, system );
    this->SenderName = Log_Sender::MC;

    this->noi           = this->systems.size();
    this->nos           = this->systems[0]->geometry->nos;
    this->nos_nonvacant = this->systems[0]->geometry->nos_nonvacant;

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

    this->gammaE_avg = 0;
}

// This implementation is mostly serial as parallelization is nontrivial
//      if the range of neighbours for each atom is not pre-defined.
void Method_MC::Iteration()
{
    // Temporaries
    auto & spins_old = *this->systems[0]->spins;
    auto spins_new   = spins_old;

    // Generate randomly displaced spin configuration according to cone radius
    // Vectormath::get_random_vectorfield_unitsphere(this->parameters_mc->prng, random_unit_vectors);

    // TODO: add switch between Metropolis and heat bath
    // One Metropolis step
    Metropolis( spins_old, spins_new );
    Vectormath::set_c_a( 1, spins_new, spins_old );
}

// Simple metropolis step
void Method_MC::Metropolis( const vectorfield & spins_old, vectorfield & spins_new )
{
    auto distribution     = std::uniform_real_distribution<scalar>( 0, 1 );
    auto distribution_idx = std::uniform_int_distribution<>( 0, this->nos - 1 );
    scalar kB_T           = Constants::k_B * this->parameters_mc->temperature;

    scalar diff = 0.01;

//    this->systems[0]->hamiltonian->Update_Energy_Contributions();  //updates the dipole-dipole internal fields
//    this->systems[0]->UpdateDDIField();

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
    this->n_rejected = 0;

    this->gammaE_avg = 0.0;

    // One Metropolis step for each spin
    Vector3 e_z{ 0, 0, 1 };
    scalar costheta, sintheta, phi;
    Matrix3 local_basis;
    scalar cos_cone_angle = std::cos( cone_angle );

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

        if( Vectormath::check_atom_type( this->systems[0]->geometry->atom_types[ispin] ) )
        {
            // Sample a cone
            if( this->parameters_mc->metropolis_step_cone )
            {
                // Calculate local basis for the spin
                if( spins_old[ispin].z() < 1 - 1e-10 )
                {
                    local_basis.col( 2 ) = spins_old[ispin];
                    local_basis.col( 0 ) = ( local_basis.col( 2 ).cross( e_z ) ).normalized();
                    local_basis.col( 1 ) = local_basis.col( 2 ).cross( local_basis.col( 0 ) );
                }
                else
                {
                    local_basis = Matrix3::Identity();
                }

                // Rotation angle between 0 and cone_angle degrees
                costheta = 1 - ( 1 - cos_cone_angle ) * distribution( this->parameters_mc->prng );

                sintheta = std::sqrt( 1 - costheta * costheta );

                // Either we flip the spin about the Ising axis or allow a small cone angle of unflipped wobbling.
                if (this->parameters_mc->metropolis_spin_flip > 0 && distribution( this->parameters_mc->prng )>0.5)
                {
                    costheta *= -1;
                }

                // Random distribution of phi between 0 and 360 degrees
                phi = 2 * Constants::Pi * distribution( this->parameters_mc->prng );

                // New spin orientation in local basis
                Vector3 local_spin_new{ sintheta * std::cos( phi ), sintheta * std::sin( phi ), costheta };

                // New spin orientation in regular basis
                spins_new[ispin] = local_basis * local_spin_new;
            }
            // Sample the entire unit sphere
            else
            {
                // Rotation angle between 0 and 180 degrees
                costheta = distribution( this->parameters_mc->prng );

                sintheta = std::sqrt( 1 - costheta * costheta );

                // Random distribution of phi between 0 and 360 degrees
                phi = 2 * Constants::Pi * distribution( this->parameters_mc->prng );

                // New spin orientation in local basis
                spins_new[ispin] = Vector3{ sintheta * std::cos( phi ), sintheta * std::sin( phi ), costheta };
            }

            // Energy difference of configurations with and without displacement
            scalar Eold  = this->systems[0]->hamiltonian->Energy_Single_Spin( ispin, spins_old );
            scalar Enew  = this->systems[0]->hamiltonian->Energy_Single_Spin( ispin, spins_new );
            scalar Ediff = Enew - Eold;

            scalar gamma_E = 0.0;
            float B_mag;
            float normal[3];

            if (this->parameters_mc->tunneling_use_tunneling)
            {    
                auto * ham = dynamic_cast<Engine::Hamiltonian_Heisenberg *>( this->systems[0]->hamiltonian.get() );

                normal[0] = (float)ham->external_field_normal[0];
                normal[1] = (float)ham->external_field_normal[1];
                normal[2] = (float)ham->external_field_normal[2];
                B_mag = (float)ham->external_field_magnitude / Constants::mu_B;
                gamma_E = (normal[0]*normal[0] + normal[1]*normal[1])*B_mag*B_mag * this->parameters_mc->tunneling_gamma;
           }

            // Metropolis criterion: reject the step if energy rose
            if( Ediff > 1e-14 )
            {
                if( (this->parameters_mc->temperature < 1e-12) && (Ediff>gamma_E) )
                {
                    // Restore the spin
                    spins_new[ispin] = spins_old[ispin];
                    // Counter for the number of rejections
                    ++this->n_rejected;
                }
                else
                {
                    // Exponential factor
                    scalar exp_ediff = std::exp( -Ediff / kB_T );
                    // Metropolis random number
                    scalar x_metropolis = distribution( this->parameters_mc->prng );

                    // Only reject if random number is larger than exponential and energy difference is greater than tunneling term
                    if( (exp_ediff < x_metropolis) && (Ediff>gamma_E) )
                    {
                        // Restore the spin
                        spins_new[ispin] = spins_old[ispin];
                        // Counter for the number of rejections
                        ++this->n_rejected;
                    }
                    else if (Ediff<gamma_E)
                    {
                        ++this->gammaE_avg;
                    }    
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
    this->systems[0]->iteration_allowed = false;
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
    Log.SendBlock( Log_Level::All, this->SenderName, block, this->idx_image, this->idx_chain );
}

void Method_MC::Message_Step()
{
    // Update time of current step
    auto t_current = std::chrono::system_clock::now();

    // Update the system's energy
    this->systems[0]->UpdateEnergy();

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

    if ( this->parameters_mc->tunneling_use_tunneling )
    {
       block.emplace_back( fmt::format(
            "   Tunneling spin flips: {:>6.3f}", this->gammaE_avg) );
    }    

    block.emplace_back( fmt::format( "    Total energy:             {:20.10f}", this->systems[0]->E ) );
    Log.SendBlock( Log_Level::All, this->SenderName, block, this->idx_image, this->idx_chain );

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
    this->systems[0]->UpdateEnergy();

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

    if ( this->parameters_mc->tunneling_use_tunneling )
    {
       block.emplace_back( fmt::format(
            "    Tunneling Spin Flips: {:>6.3f}", this->gammaE_avg) );
    }    

    block.emplace_back( fmt::format( "    Total energy:     {:20.10f}", this->systems[0]->E ) );
    block.emplace_back( "-----------------------------------------------------" );
    Log.SendBlock( Log_Level::All, this->SenderName, block, this->idx_image, this->idx_chain );
}

void Method_MC::Save_Current( std::string starttime, int iteration, bool initial, bool final ) 
{
    // History save
    this->history_iteration.push_back( this->iteration );
    this->history_max_torque.push_back( this->max_torque );
    this->history_energy.push_back( this->systems[0]->E );

 
    // File save
    if( this->parameters->output_any )
    {
        // Convert indices to formatted strings
        auto s_img         = fmt::format( "{:0>2}", this->idx_image );
        auto base          = static_cast<std::int32_t>( log10( this->parameters->n_iterations ) );
        std::string s_iter = fmt::format( "{:0>" + fmt::format( "{}", base ) + "}", iteration );

        std::string preSpinsFile;
        std::string preEnergyFile;
        std::string fileTag;

        if( this->systems[0]->mc_parameters->output_file_tag == "<time>" )
            fileTag = starttime + "_";
        else if( this->systems[0]->mc_parameters->output_file_tag != "" )
            fileTag = this->systems[0]->mc_parameters->output_file_tag + "_";
        else
            fileTag = "";

        preSpinsFile  = this->parameters->output_folder + "/" + fileTag + "Image-" + s_img + "_Spins";
        preEnergyFile = this->parameters->output_folder + "/" + fileTag + "Image-" + s_img + "_Energy";

        // Function to write or append image and energy files
        auto writeOutputConfiguration
            = [this, preSpinsFile, preEnergyFile, iteration]( const std::string & suffix, bool append )
        {
            try
            {
                // File name and comment
                std::string spinsFile      = preSpinsFile + suffix + ".ovf";
                std::string output_comment = fmt::format(
                    "{} simulation \n# Desc:      Iteration: {}\n# Desc:      Maximum torque: {}",
                    this->Name(), iteration, this->max_torque );

                // File format
                IO::VF_FileFormat format = this->systems[0]->mc_parameters->output_vf_filetype;

                // Spin Configuration
                auto & spins        = *this->systems[0]->spins;
                auto segment        = IO::OVF_Segment( *this->systems[0] );
                std::string title   = fmt::format( "SPIRIT Version {}", Utility::version_full );
                segment.title       = strdup( title.c_str() );
                segment.comment     = strdup( output_comment.c_str() );
                segment.valuedim    = 3;
                segment.valuelabels = strdup( "spin_x spin_y spin_z" );
                segment.valueunits  = strdup( "none none none" );
                if( append )
                    IO::OVF_File( spinsFile ).append_segment( segment, spins[0].data(), int( format ) );
                else
                    IO::OVF_File( spinsFile ).write_segment( segment, spins[0].data(), int( format ) );
            }
            catch( ... )
            {
                spirit_handle_exception_core( "MC output failed" );
            }
        };

        auto writeOutputEnergy
            = [this, preSpinsFile, preEnergyFile, iteration]( const std::string & suffix, bool append )
        {
            bool normalize   = this->systems[0]->mc_parameters->output_energy_divide_by_nspins;
            bool readability = this->systems[0]->mc_parameters->output_energy_add_readability_lines;

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
                        *this->systems[0], energyFile, { "iteration", "E_tot" }, true, normalize, readability );
                // Append Energy to File
                IO::Append_Image_Energy( *this->systems[0], iteration, energyFile, normalize, readability );
            }
            else
            {
                IO::Write_Energy_Header(
                    *this->systems[0], energyFile, { "iteration", "E_tot" }, true, normalize, readability );
                IO::Append_Image_Energy( *this->systems[0], iteration, energyFile, normalize, readability );
                if( this->systems[0]->mc_parameters->output_energy_spin_resolved )
                {
                    // Gather the data
                    std::vector<std::pair<std::string, scalarfield>> contributions_spins( 0 );
                    this->systems[0]->UpdateEnergy();
                    this->systems[0]->hamiltonian->Energy_Contributions_per_Spin(
                        *this->systems[0]->spins, contributions_spins );
                    int datasize = ( 1 + contributions_spins.size() ) * this->systems[0]->nos;
                    scalarfield data( datasize, 0 );
                    for( int ispin = 0; ispin < this->systems[0]->nos; ++ispin )
                    {
                        scalar E_spin = 0;
                        int j         = 1;
                        for( auto & contribution : contributions_spins )
                        {
                            E_spin += contribution.second[ispin];
                            data[ispin + j] = contribution.second[ispin];
                            ++j;
                        }
                        data[ispin] = E_spin;
                    }

                    // Segment
                    auto segment = IO::OVF_Segment( *this->systems[0] );

                    std::string title   = fmt::format( "SPIRIT Version {}", Utility::version_full );
                    segment.title       = strdup( title.c_str() );
                    std::string comment = fmt::format( "Energy per spin. Total={}meV", this->systems[0]->E );
                    for( const auto & contribution : this->systems[0]->E_array )
                        comment += fmt::format( ", {}={}meV", contribution.first, contribution.second );
                    segment.comment  = strdup( comment.c_str() );
                    segment.valuedim = 1 + this->systems[0]->E_array.size();

                    std::string valuelabels = "Total";
                    std::string valueunits  = "meV";
                    for( const auto & pair : this->systems[0]->E_array )
                    {
                        valuelabels += fmt::format( " {}", pair.first );
                        valueunits += " meV";
                    }
                    segment.valuelabels = strdup( valuelabels.c_str() );

                    // File format
                    IO::VF_FileFormat format = this->systems[0]->mc_parameters->output_vf_filetype;

                    // open and write
                    IO::OVF_File( energyFilePerSpin ).write_segment( segment, data.data(), static_cast<int>( format ) );

                    Log( Utility::Log_Level::Info, Utility::Log_Sender::API,
                         fmt::format(
                             "Wrote spins to file \"{}\" with format {}", energyFilePerSpin,
                             static_cast<int>( format ) ),
                         -1, -1 );
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
        if( this->systems[0]->mc_parameters->output_configuration_step )
        {
            writeOutputConfiguration( "_" + s_iter, false );
        }
        if( this->systems[0]->mc_parameters->output_energy_step )
        {
            writeOutputEnergy( "_" + s_iter, false );
        }

        // Archive file output (appending)
        if( this->systems[0]->mc_parameters->output_configuration_archive )
        {
            writeOutputConfiguration( "-archive", true );
        }
        if( this->systems[0]->mc_parameters->output_energy_archive )
        {
            writeOutputEnergy( "-archive", true );
        }

        // Save Log
        Log.Append_to_File();
    }    
}

// Method name as string
std::string Method_MC::Name()
{
    return "MC";
}

} // namespace Engine
