#include <io/Filter_File_Handle.hpp>
#include <io/hamiltonian/Hamiltonian.hpp>
#include <io/hamiltonian/Interactions.hpp>

using Utility::Log_Level;
using Utility::Log_Sender;

namespace IO
{

namespace
{

std::string Hamiltonian_Type_from_Config( const std::string & config_file_name, const std::string_view default_type )
{
    std::string hamiltonian_type{ default_type };

    // Hamiltonian type
    if( !config_file_name.empty() )
    {
        try
        {
            Log( Log_Level::Debug, Log_Sender::IO, "Hamiltonian: deciding type" );
            IO::Filter_File_Handle config_file_handle( config_file_name );

            // What hamiltonian do we use?
            config_file_handle.Read_Single( hamiltonian_type, "hamiltonian" );
        }
        catch( ... )
        {
            spirit_handle_exception_core( fmt::format(
                "Unable to read Hamiltonian type from config file  \"{}\". Using default.", config_file_name ) );
            hamiltonian_type = default_type;
        }
    }
    else
        Log( Log_Level::Parameter, Log_Sender::IO,
             fmt::format( "Hamiltonian: Using default Hamiltonian: {}", hamiltonian_type ) );

    return hamiltonian_type;
}

#define log_error( expr )                                                                                              \
    if( auto error = ( expr ); error.has_value() )                                                                     \
        Log( Log_Level::Error, Log_Sender::IO, *error );

std::unique_ptr<Engine::Spin::Hamiltonian> Hamiltonian_Heisenberg_from_Config(
    const std::string & config_file_name, Data::Geometry geometry, intfield boundary_conditions,
    const std::string_view hamiltonian_type )
{
    Log( Log_Level::Debug, Log_Sender::IO, "Hamiltonian_Heisenberg: building" );
    // pull in the relevant namespaces and type we want to build
    namespace Interaction = Engine::Spin::Interaction;
    using Engine::Spin::Hamiltonian;

    std::vector<std::string> parameter_log;
    parameter_log.emplace_back( "Hamiltonian Heisenberg:" );
    parameter_log.emplace_back( fmt::format(
        "    {:<21} = {} {} {}", "boundary conditions", boundary_conditions[0], boundary_conditions[1],
        boundary_conditions[2] ) );

    //-------------- Insert default values here -----------------------------
    // External Magnetic Field
    scalar external_field_magnitude = 0;
    Vector3 external_field_normal   = { 0.0, 0.0, 1.0 };

    // Anisotropy
    auto anisotropy_indices          = intfield( geometry.n_cell_atoms, 0 );
    auto anisotropy_magnitudes       = scalarfield( geometry.n_cell_atoms, 0.0 );
    auto anisotropy_normals          = vectorfield( geometry.n_cell_atoms, Vector3{ 0.0, 0.0, 1.0 } );
    auto cubic_anisotropy_indices    = intfield( geometry.n_cell_atoms, 0 );
    auto cubic_anisotropy_magnitudes = scalarfield( geometry.n_cell_atoms, 0.0 );

    auto biaxial_anisotropy_indices           = intfield( 0 );
    auto biaxial_anisotropy_polynomial_bases  = field<PolynomialBasis>{};
    auto biaxial_anisotropy_polynomial_site_p = field<unsigned int>{};
    auto biaxial_anisotropy_polynomial_terms  = field<PolynomialTerm>{};

    // ------------ Pair Interactions ------------
    auto exchange_pairs      = pairfield( 0 );
    auto exchange_magnitudes = scalarfield( 0 );
    auto dmi_pairs           = pairfield( 0 );
    auto dmi_magnitudes      = scalarfield( 0 );
    auto dmi_normals         = vectorfield( 0 );

    int dm_chirality = 1;

    auto ddi_method                = Engine::Spin::DDI_Method::None;
    intfield ddi_n_periodic_images = { 4, 4, 4 };
    scalar ddi_radius              = 0.0;
    bool ddi_pb_zero_padding       = false;

    // ------------ Quadruplet Interactions ------------
    auto quadruplets           = quadrupletfield( 0 );
    auto quadruplet_magnitudes = scalarfield( 0 );

    //------------------------------- Parser --------------------------------

    // Iteration variables
    if( !config_file_name.empty() )
    {
        Zeeman_from_Config( config_file_name, parameter_log, external_field_magnitude, external_field_normal );

        Anisotropy_from_Config(
            config_file_name, geometry, parameter_log, anisotropy_indices, anisotropy_magnitudes, anisotropy_normals,
            cubic_anisotropy_indices, cubic_anisotropy_magnitudes );

        Biaxial_Anisotropy_from_Config(
            config_file_name, geometry, parameter_log, biaxial_anisotropy_indices, biaxial_anisotropy_polynomial_bases,
            biaxial_anisotropy_polynomial_site_p, biaxial_anisotropy_polynomial_terms );

        if( hamiltonian_type == "heisenberg_neighbours" )
            Pair_Interactions_from_Shells_from_Config(
                config_file_name, geometry, parameter_log, exchange_magnitudes, dmi_magnitudes, dm_chirality );
        else
            Pair_Interactions_from_Pairs_from_Config(
                config_file_name, geometry, parameter_log, exchange_pairs, exchange_magnitudes, dmi_pairs,
                dmi_magnitudes, dmi_normals );

        DDI_from_Config(
            config_file_name, geometry, parameter_log, ddi_method, ddi_n_periodic_images, ddi_pb_zero_padding,
            ddi_radius );

        Quadruplets_from_Config( config_file_name, geometry, parameter_log, quadruplets, quadruplet_magnitudes );
    }
    else
        Log( Log_Level::Parameter, Log_Sender::IO, "Hamiltonian_Heisenberg: Using default configuration!" );

    Log( Log_Level::Parameter, Log_Sender::IO, parameter_log );

    auto hamiltonian = std::make_unique<Hamiltonian>( std::move( geometry ), std::move( boundary_conditions ) );

    log_error( hamiltonian->set_data<Interaction::Zeeman>(
        external_field_magnitude * Utility::Constants::mu_B, external_field_normal ) );

    log_error( hamiltonian->set_data<Interaction::Anisotropy>(
        anisotropy_indices, anisotropy_magnitudes, anisotropy_normals ) );

    log_error( hamiltonian->set_data<Interaction::Biaxial_Anisotropy>(
        biaxial_anisotropy_indices, biaxial_anisotropy_polynomial_bases, biaxial_anisotropy_polynomial_site_p,
        biaxial_anisotropy_polynomial_terms ) );

    log_error(
        hamiltonian->set_data<Interaction::Cubic_Anisotropy>( cubic_anisotropy_indices, cubic_anisotropy_magnitudes ) );

    Interaction::Exchange::Data exchange;
    Interaction::DMI::Data dmi;
    if( hamiltonian_type == "heisenberg_neighbours" )
    {
        log_error( hamiltonian->set_data<Interaction::Exchange>( exchange_magnitudes ) );
        log_error( hamiltonian->set_data<Interaction::DMI>( dmi_magnitudes, dm_chirality ) );
    }
    else
    {
        log_error( hamiltonian->set_data<Interaction::Exchange>( exchange_pairs, exchange_magnitudes ) );
        log_error( hamiltonian->set_data<Interaction::DMI>( dmi_pairs, dmi_magnitudes, dmi_normals ) );
    }

    log_error( hamiltonian->set_data<Interaction::Quadruplet>( quadruplets, quadruplet_magnitudes ) );
    log_error(
        hamiltonian->set_data<Interaction::DDI>( ddi_method, ddi_radius, ddi_pb_zero_padding, ddi_n_periodic_images ) );

    assert( hamiltonian->Name() == "Heisenberg" );
    Log( Log_Level::Debug, Log_Sender::IO, fmt::format( "Hamiltonian_{}: built", hamiltonian->Name() ) );
    return hamiltonian;
}

std::unique_ptr<Engine::Spin::Hamiltonian> Hamiltonian_Gaussian_from_Config(
    const std::string & config_file_name, Data::Geometry geometry, intfield boundary_conditions )
{
    Log( Log_Level::Debug, Log_Sender::IO, "Hamiltonian_Gaussian: building" );
    // pull in the relevant namespaces and type we want to build
    namespace Interaction = Engine::Spin::Interaction;
    using Engine::Spin::Hamiltonian;

    std::vector<std::string> parameter_log;
    parameter_log.emplace_back( "Hamiltonian Gaussian:" );

    //-------------- Insert default values here -----------------------------
    // Amplitudes
    scalarfield amplitude = { 1 };
    // Widths
    scalarfield width = { 1 };
    // Centers
    vectorfield center = { Vector3{ 0, 0, 1 } };

    //------------------------------- Parser --------------------------------
    if( !config_file_name.empty() )
    {
        Gaussian_from_Config( config_file_name, parameter_log, amplitude, width, center );
    }
    else
        Log( Log_Level::Parameter, Log_Sender::IO, "Hamiltonian_Gaussian: Using default configuration!" );

    Log( Log_Level::Parameter, Log_Sender::IO, parameter_log );

    auto hamiltonian = std::make_unique<Hamiltonian>( std::move( geometry ), std::move( boundary_conditions ) );

    if( auto error = hamiltonian->set_data<Interaction::Gaussian>( amplitude, width, center ); error.has_value() )
        Log( Log_Level::Error, Log_Sender::IO, *error );

    assert( hamiltonian->Name() == "Gaussian" );
    Log( Log_Level::Debug, Log_Sender::IO, fmt::format( "Hamiltonian_{}: built", hamiltonian->Name() ) );
    return hamiltonian;
}

} // namespace

template<>
std::unique_ptr<Engine::Spin::Hamiltonian>
Hamiltonian_from_Config( const std::string & config_file_name, Data::Geometry geometry, intfield boundary_conditions )
{
    //------------------------------- Parser --------------------------------
    Log( Log_Level::Debug, Log_Sender::IO, "Hamiltonian: building" );

    const std::string hamiltonian_type = Hamiltonian_Type_from_Config( config_file_name, "heisenberg_neighbours" );

    // Hamiltonian
    std::unique_ptr<Engine::Spin::Hamiltonian> hamiltonian;
    try
    {
        if( hamiltonian_type == "heisenberg_neighbours" || hamiltonian_type == "heisenberg_pairs" )
        {
            hamiltonian = Hamiltonian_Heisenberg_from_Config(
                config_file_name, std::move( geometry ), std::move( boundary_conditions ), hamiltonian_type );
        }
        else if( hamiltonian_type == "gaussian" )
        {
            hamiltonian = Hamiltonian_Gaussian_from_Config(
                config_file_name, std::move( geometry ), std::move( boundary_conditions ) );
        }
        else
        {
            spirit_throw(
                Utility::Exception_Classifier::System_not_Initialized, Log_Level::Severe,
                fmt::format( "Hamiltonian: Invalid type \"{}\"", hamiltonian_type ) );
        }
    }
    catch( ... )
    {
        spirit_handle_exception_core(
            fmt::format( "Unable to initialize Hamiltonian from config file \"{}\"", config_file_name ) );
    }

    // Return
    Log( Log_Level::Debug, Log_Sender::IO,
         fmt::format( "Hamiltonian: built hamiltonian of type: {}", hamiltonian_type ) );
    return hamiltonian;
}

} // namespace IO
