#pragma once
#ifndef SPIRIT_CORE_ENGINE_SPIN_HAMILTONIAN_HPP
#define SPIRIT_CORE_ENGINE_SPIN_HAMILTONIAN_HPP

#include <engine/Vectormath_Defines.hpp>
#include <engine/common/Hamiltonian.hpp>
#include <engine/spin/Interaction_Standalone_Adaptor.hpp>
#include <engine/spin/interaction/Anisotropy.hpp>
#include <engine/spin/interaction/Biaxial_Anisotropy.hpp>
#include <engine/spin/interaction/Cubic_Anisotropy.hpp>
#include <engine/spin/interaction/DDI.hpp>
#include <engine/spin/interaction/DMI.hpp>
#include <engine/spin/interaction/Exchange.hpp>
#include <engine/spin/interaction/Gaussian.hpp>
#include <engine/spin/interaction/Quadruplet.hpp>
#include <engine/spin/interaction/Zeeman.hpp>
#include <utility/Variadic_Traits.hpp>

namespace Engine
{

namespace Spin
{
// TODO: look into mixins and decide if they are more suitable to compose the `Hamiltonian` and `StandaloneAdaptor` types

namespace Accessor = Common::Accessor;
namespace Functor  = Common::Functor;

// clang-format off
using HamiltonianBase = Common::Hamiltonian<
    StateType, Spin::Interaction::StandaloneAdaptor<StateType>,
    Interaction::Zeeman,
    Interaction::Anisotropy,
    Interaction::Biaxial_Anisotropy,
    Interaction::Cubic_Anisotropy,
    Interaction::Exchange,
    Interaction::DMI,
    Interaction::Quadruplet,
    Interaction::DDI,
    Interaction::Gaussian>;
// clang-format on

// Hamiltonian for (pure) spin systems
struct Hamiltonian : public HamiltonianBase
{
public:
    using HamiltonianBase::Hamiltonian;

    using state_t = typename HamiltonianBase::state_t;

    void Hessian( const state_t & state, MatrixX & hessian )
    {
        hessian.setZero();
        Hessian_Impl( state, Interaction::Functor::dense_hessian_wrapper( hessian ) );
    };

    void Sparse_Hessian( const state_t & state, SpMatrixX & hessian )
    {
        std::vector<Common::Interaction::triplet> tripletList;
        tripletList.reserve( get_geometry().n_cells_total * Sparse_Hessian_Size_per_Cell() );
        Hessian_Impl( state, Interaction::Functor::sparse_hessian_wrapper( tripletList ) );
        hessian.setFromTriplets( tripletList.begin(), tripletList.end() );
    };

    std::size_t Sparse_Hessian_Size_per_Cell() const
    {
        auto func = []( const auto &... interaction ) -> std::size_t
        { return ( std::size_t( 0 ) + ... + interaction.Sparse_Hessian_Size_per_Cell() ); };
        return Backend::apply( func, local ) + Backend::apply( func, nonlocal );
    };

    void Gradient( const state_t & state, vectorfield & gradient )
    {
        const auto nos = state.spin.size();

        if( gradient.size() != nos )
            gradient = vectorfield( nos, Vector3::Zero() );
        else
            Vectormath::fill( gradient, Vector3::Zero() );

        Backend::transform(
            SPIRIT_PAR indices.begin(), indices.end(), gradient.begin(),
            Functor::transform_op(
                Functor::tuple_dispatch<Accessor::Gradient>( local ), Vector3{ 0.0, 0.0, 0.0 }, state ) );

        Functor::apply( Functor::tuple_dispatch<Accessor::Gradient>( nonlocal ), state, gradient );
    };

    // provided for backwards compatibility, this function no longer serves a purpose
    [[nodiscard]] scalar Gradient_and_Energy( const state_t & state, vectorfield & gradient )
    {
        Gradient( state, gradient );
        return Energy( state );
    };

    [[nodiscard]] std::string_view Name() const noexcept
    {
        if( !this->template is_contributing<Interaction::Gaussian>() )
            return "Heisenberg";

        auto gaussian_func = []( auto &... interaction )
        {
            return (
                true && ...
                && ( std::is_same_v<
                         typename std::decay_t<decltype( interaction )>::Interaction, Spin::Interaction::Gaussian>
                     || !interaction.is_contributing() ) );
        };

        if( Backend::apply( gaussian_func, local ) && Backend::apply( gaussian_func, nonlocal ) )
            return "Gaussian";

        return "Unknown";
    };

private:
    template<typename Callable>
    void Hessian_Impl( const state_t & state, Callable hessian )
    {
        Backend::cpu::for_each(
            indices.begin(), indices.end(),
            Functor::for_each_op( Functor::tuple_dispatch<Accessor::Hessian>( local ), state, hessian ) );

        Functor::apply( Functor::tuple_dispatch<Accessor::Hessian>( nonlocal ), state, hessian );
    };
};

} // namespace Spin

} // namespace Engine

#endif
