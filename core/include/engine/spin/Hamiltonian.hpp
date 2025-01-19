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

// Hamiltonian for (pure) spin systems
template<typename state_type, typename StandaloneAdaptorType, typename... InteractionTypes>
class Hamiltonian : public Common::Hamiltonian<state_type, StandaloneAdaptorType, InteractionTypes...>
{
    using base_t = Common::Hamiltonian<state_type, StandaloneAdaptorType, InteractionTypes...>;

public:
    using Common::Hamiltonian<state_type, StandaloneAdaptorType, InteractionTypes...>::Hamiltonian;

    using state_t = state_type;

    void Hessian( const state_t & state, MatrixX & hessian )
    {
        hessian.setZero();
        Hessian_Impl( state, Interaction::Functor::dense_hessian_wrapper( hessian ) );
    };

    void Sparse_Hessian( const state_t & state, SpMatrixX & hessian )
    {
        std::vector<Common::Interaction::triplet> tripletList;
        tripletList.reserve( this->get_geometry().n_cells_total * Sparse_Hessian_Size_per_Cell() );
        Hessian_Impl( state, Interaction::Functor::sparse_hessian_wrapper( tripletList ) );
        hessian.setFromTriplets( tripletList.begin(), tripletList.end() );
    };

    std::size_t Sparse_Hessian_Size_per_Cell() const
    {
        auto func = []( const auto &... interaction ) -> std::size_t
        { return ( std::size_t( 0 ) + ... + interaction.Sparse_Hessian_Size_per_Cell() ); };
        return Backend::apply( func, this->local ) + Backend::apply( func, this->nonlocal );
    };

    void Gradient( const state_t & state, vectorfield & gradient )
    {
        const auto nos = state.spin.size();

        if( gradient.size() != nos )
            gradient = vectorfield( nos, Vector3::Zero() );
        else
            Vectormath::fill( gradient, Vector3::Zero() );

        Backend::transform(
            SPIRIT_PAR this->indices.begin(), this->indices.end(), gradient.begin(),
            Functor::transform_op(
                Functor::tuple_dispatch<Accessor::Gradient>( this->local ), Vector3{ 0.0, 0.0, 0.0 }, state ) );

        Functor::apply( Functor::tuple_dispatch<Accessor::Gradient>( this->nonlocal ), state, gradient );
    };

    // provided for backwards compatibility, this function no longer serves a purpose
    [[nodiscard]] scalar Gradient_and_Energy( const state_t & state, vectorfield & gradient )
    {
        Gradient( state, gradient );
        return this->Energy( state );
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

        if( Backend::apply( gaussian_func, this->local ) && Backend::apply( gaussian_func, this->nonlocal ) )
            return "Gaussian";

        return "Unknown";
    };

private:
    template<typename Callable>
    void Hessian_Impl( const state_t & state, Callable hessian )
    {
        Backend::cpu::for_each(
            this->indices.begin(), this->indices.end(),
            Functor::for_each_op( Functor::tuple_dispatch<Accessor::Hessian>( this->local ), state, hessian ) );

        Functor::apply( Functor::tuple_dispatch<Accessor::Hessian>( this->nonlocal ), state, hessian );
    };
};

struct HamiltonianVariantTypes
{
    using state_t     = StateType;
    using AdaptorType = Spin::Interaction::StandaloneAdaptor<state_t>;

    using Gaussian   = Hamiltonian<state_t, AdaptorType, Interaction::Gaussian>;
    using Heisenberg = Hamiltonian<
        state_t, AdaptorType, Interaction::Zeeman, Interaction::Anisotropy, Interaction::Biaxial_Anisotropy,
        Interaction::Cubic_Anisotropy, Interaction::Exchange, Interaction::DMI, Interaction::Quadruplet,
        Interaction::DDI, Interaction::Gaussian>;

    using Variant = Heisenberg;
};

// Single Type wrapper around Variant Hamiltonian type
// Should the visitors split up into standalone function objects?
class HamiltonianVariant : public Common::HamiltonianVariant<HamiltonianVariant, HamiltonianVariantTypes>
{
public:
    using state_t     = typename HamiltonianVariantTypes::state_t;
    using Gaussian    = typename HamiltonianVariantTypes::Gaussian;
    using Heisenberg  = typename HamiltonianVariantTypes::Heisenberg;
    using Variant     = typename HamiltonianVariantTypes::Variant;
    using AdaptorType = typename HamiltonianVariantTypes::AdaptorType;

private:
    using base_t = Common::HamiltonianVariant<HamiltonianVariant, HamiltonianVariantTypes>;

public:
    explicit HamiltonianVariant( Heisenberg && heisenberg ) noexcept( std::is_nothrow_move_constructible_v<Heisenberg> )
            : base_t( std::move( heisenberg ) ) {};

    void Gradient( const state_t & state, vectorfield & gradient )
    {
        hamiltonian.Gradient( state, gradient );
    }

    void Hessian( const state_t & state, MatrixX & hessian )
    {
        hamiltonian.Hessian( state, hessian );
    }

    void Sparse_Hessian( const state_t & state, SpMatrixX & hessian )
    {
        hamiltonian.Sparse_Hessian( state, hessian );
    }

    void Gradient_and_Energy( const state_t & state, vectorfield & gradient, scalar & energy )
    {
        energy = hamiltonian.Gradient_and_Energy( state, gradient );
    }

    [[nodiscard]] std::string_view Name() const noexcept
    {
        return hamiltonian.Name();
    }
};

} // namespace Spin

} // namespace Engine

#endif
