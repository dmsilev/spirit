#pragma once
#ifndef SPIRIT_CORE_ENGINE_COMMON_HAMILTONIAN_HPP
#define SPIRIT_CORE_ENGINE_COMMON_HAMILTONIAN_HPP

#include <Spirit/Hamiltonian.h>
#include <Spirit/Spirit_Defines.h>
#include <data/Geometry.hpp>
#include <data/Misc.hpp>
#include <engine/Backend.hpp>
#include <engine/FFT.hpp>
#include <engine/Span.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <engine/common/Functor_Dispatch.hpp>
#include <engine/common/Interaction_Wrapper.hpp>
#include <utility/Variadic_Traits.hpp>

namespace Engine
{

namespace Common
{

namespace Trait
{

template<typename T>
struct Index
{
    using type = typename T::Index;
};

template<typename T>
struct IndexStorage
{
    using type = typename T::IndexStorage;
};

} // namespace Trait

namespace Interaction
{

template<typename... InteractionTypes>
void setPtrAddress(
    Backend::tuple<InteractionWrapper<InteractionTypes>...> & interactions, const ::Data::Geometry * geometry,
    const intfield * boundary_conditions ) noexcept
{
    Backend::apply(
        [geometry, boundary_conditions]( InteractionWrapper<InteractionTypes> &... interaction )
        {
            ( ...,
              [geometry, boundary_conditions]( typename InteractionTypes::Cache & entry )
              {
                  using cache_t = typename InteractionTypes::Cache;
                  if constexpr( has_geometry_member<cache_t>::value )
                      entry.geometry = geometry;

                  if constexpr( has_bc_member<cache_t>::value )
                      entry.boundary_conditions = boundary_conditions;
              }( interaction.cache ) );
        },
        interactions );
}

template<typename T>
SPIRIT_HOSTDEVICE void clearIndexStorage( Backend::optional<T> & index )
{
    index.reset();
}

template<typename T>
SPIRIT_HOSTDEVICE void clearIndexStorage( T & index )
{
    index.clear();
}

} // namespace Interaction

// Hamiltonian for (pure) spin systems
template<typename state_type, typename InteractionAdaptorType, typename... InteractionTypes>
struct Hamiltonian
{
    static_assert( std::conjunction<std::is_same<state_type, typename InteractionTypes::state_t>...>::value );

private:
    template<typename... Ts>
    using WrappedInteractions = Backend::tuple<Interaction::InteractionWrapper<Ts>...>;

    template<template<class> class Pred, template<class...> class Variadic>
    using filtered = typename Utility::variadic_filter<Pred, Variadic, InteractionTypes...>::type;

    template<template<class> class Pred>
    using indexed = typename Utility::variadic_filter_index_sequence<Pred, InteractionTypes...>::type;

    using LocalInteractions = filtered<Interaction::is_local, Backend::tuple>;

    using StandaloneFactory = Interaction::StandaloneFactory<InteractionAdaptorType>;

    template<class Tuple, std::size_t... I_local, std::size_t... I_nonlocal>
    Hamiltonian(
        Data::Geometry geometry, intfield boundary_conditions, std::index_sequence<I_local...>,
        std::index_sequence<I_nonlocal...>, Tuple && data )
            : geometry( std::make_shared<::Data::Geometry>( std::move( geometry ) ) ),
              boundary_conditions( std::move( boundary_conditions ) ),
              indices( 0 ),
              index_storage( 0 ),
              local( Backend::make_tuple( std::get<I_local>( std::forward<Tuple>( data ) )... ) ),
              nonlocal( Backend::make_tuple( std::get<I_nonlocal>( std::forward<Tuple>( data ) )... ) )
    {
        static_assert(
            Utility::are_disjoint<std::index_sequence<I_local...>, std::index_sequence<I_nonlocal...>>::value );

        static_assert( sizeof...( InteractionTypes ) == std::tuple_size<Tuple>::value );
        static_assert( sizeof...( I_local ) + sizeof...( I_nonlocal ) == std::tuple_size<Tuple>::value );

        applyGeometry();
    };

    template<typename T>
    using is_local = Interaction::is_local<T>;

    template<typename T>
    using is_nonlocal = std::negation<Interaction::is_local<T>>;

public:
    using state_t     = state_type;
    using AdaptorType = InteractionAdaptorType;

    using local_indices    = indexed<is_local>;
    using nonlocal_indices = indexed<is_nonlocal>;

    using InteractionTuple = WrappedInteractions<InteractionTypes...>;

    using NonLocalInteractionTuple = filtered<is_nonlocal, WrappedInteractions>;

    using LocalInteractionTuple = filtered<is_local, WrappedInteractions>;
    using IndexTuple            = typename Utility::variadic_map<Trait::Index, Backend::tuple, LocalInteractions>::type;
    using IndexVector           = field<IndexTuple>;
    using IndexStorageTuple =
        typename Utility::variadic_map<Trait::IndexStorage, Backend::tuple, LocalInteractions>::type;
    using IndexStorageVector = field<IndexStorageTuple>;

    template<typename... DataTypes>
    Hamiltonian( Data::Geometry geometry, intfield boundary_conditions, DataTypes &&... data )
            : Hamiltonian(
                  geometry, boundary_conditions, local_indices{}, nonlocal_indices{},
                  std::make_tuple(
                      Interaction::InteractionWrapper<InteractionTypes>( std::forward<DataTypes>( data ) )... ) ){};

    Hamiltonian( Data::Geometry geometry, intfield boundary_conditions )
            : geometry( std::make_shared<::Data::Geometry>( std::move( geometry ) ) ),
              boundary_conditions( std::move( boundary_conditions ) ),
              indices{},
              index_storage{},
              local{},
              nonlocal{}
    {
        applyGeometry();
    };

    // rule of five, because we use pointers to the geometry and the boundary_conditions in the cache
    // this choice should keep the interfaces a bit cleaner and allow adding more global dependencies
    // in the future.
    ~Hamiltonian() = default;
    Hamiltonian( const Hamiltonian & other )
            : geometry( other.geometry ),
              boundary_conditions( other.boundary_conditions ),
              index_storage( other.index_storage ),
              indices( other.indices ),
              local( other.local ),
              nonlocal( other.nonlocal )
    {
        setPtrAddress();
        updateIndexVector();
    };
    Hamiltonian & operator=( const Hamiltonian & other )
    {
        if( this != &other )
        {
            geometry            = other.geometry;
            boundary_conditions = other.boundary_conditions;
            index_storage       = other.index_storage;
            indices             = other.indices;
            local               = other.local;
            nonlocal            = other.nonlocal;
            setPtrAddress();
            updateIndexVector();
        };
        return *this;
    };
    Hamiltonian( Hamiltonian && other ) noexcept
            : geometry( std::move( other.geometry ) ),
              boundary_conditions( std::move( other.boundary_conditions ) ),
              index_storage( std::move( other.index_storage ) ),
              indices( std::move( other.indices ) ),
              local( std::move( other.local ) ),
              nonlocal( std::move( other.nonlocal ) )
    {
        setPtrAddress();
        updateIndexVector();
    };
    Hamiltonian & operator=( Hamiltonian && other ) noexcept
    {
        if( this != &other )
        {
            geometry            = std::move( other.geometry );
            boundary_conditions = std::move( other.boundary_conditions );
            index_storage       = std::move( other.index_storage );
            indices             = std::move( other.indices );
            local               = std::move( other.local );
            nonlocal            = std::move( other.nonlocal );
            setPtrAddress();
            updateIndexVector();
        };
        return *this;
    };

    void setPtrAddress() noexcept
    {
        Interaction::setPtrAddress( local, geometry.get(), &boundary_conditions );
        Interaction::setPtrAddress( nonlocal, geometry.get(), &boundary_conditions );
    }

    void Energy_per_Spin( const state_t & state, scalarfield & energy_per_spin )
    {
        if( energy_per_spin.size() != geometry->nos )
            energy_per_spin = scalarfield( geometry->nos, 0.0 );
        else
            Vectormath::fill( energy_per_spin, 0.0 );

        Backend::transform(
            SPIRIT_PAR indices.begin(), indices.end(), energy_per_spin.begin(),
            Functor::transform_op( Functor::tuple_dispatch<Accessor::Energy>( local ), scalar( 0.0 ), state ) );

        Functor::apply( Functor::tuple_dispatch<Accessor::Energy>( nonlocal ), state, energy_per_spin );
    };

    void Energy_Contributions_per_Spin( const state_t & state, Data::vectorlabeled<scalarfield> & contributions )
    {
        auto active           = active_interactions();
        const auto & n_active = active.size();

        if( contributions.size() != n_active )
        {
            contributions = Data::vectorlabeled<scalarfield>( n_active, { "", scalarfield( geometry->nos, 0.0 ) } );
        }

        Backend::cpu::transform(
            active.begin(), active.end(), contributions.begin(),
            [&state, nos = geometry->nos]( const std::unique_ptr<InteractionAdaptorType> & interaction )
            {
                scalarfield energy_per_spin( nos, 0.0 );
                interaction->Energy_per_Spin( state, energy_per_spin );
                return std::make_pair( interaction->Name(), energy_per_spin );
            } );
    };

    [[nodiscard]] Data::vectorlabeled<scalar> Energy_Contributions( const state_t & state )
    {
        auto active           = active_interactions();
        const auto & n_active = active.size();
        Data::vectorlabeled<scalar> contributions( n_active, { "", 0.0 } );

        Backend::cpu::transform(
            active.begin(), active.end(), contributions.begin(),
            [&state]( const std::unique_ptr<InteractionAdaptorType> & interaction )
            { return std::make_pair( interaction->Name(), interaction->Energy( state ) ); } );

        return contributions;
    };

    // Calculate the total energy for a single spin to be used in Monte Carlo.
    //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.
    [[nodiscard]] scalar Energy_Single_Spin( const int ispin, const state_t & state )
    {
        return std::invoke(
                   Functor::transform_op(
                       Functor::tuple_dispatch<Accessor::Energy_Single_Spin>( local ), scalar( 0.0 ), state ),
                   indices[ispin] )
               + Functor::apply_reduce(
                   Functor::tuple_dispatch<Accessor::Energy_Single_Spin>( nonlocal ), scalar( 0.0 ), ispin, state );
    };

    [[nodiscard]] scalar Energy( const state_t & state )
    {
        return Backend::transform_reduce(
                   SPIRIT_PAR indices.begin(), indices.end(), scalar( 0.0 ), Backend::plus<scalar>{},
                   Functor::transform_op( Functor::tuple_dispatch<Accessor::Energy>( local ), scalar( 0.0 ), state ) )
               + Functor::apply_reduce(
                   Functor::tuple_dispatch<Accessor::Energy_Total>( nonlocal ), scalar( 0.0 ), state );
    };

    [[nodiscard]] std::size_t active_count() const
    {
        auto f = []( const auto &... interaction )
        { return ( std::size_t( 0 ) + ... + ( interaction.is_contributing() ? 1 : 0 ) ); };
        return Backend::apply( f, local ) + Backend::apply( f, nonlocal );
    }

    [[nodiscard]] auto active_interactions() -> std::vector<std::unique_ptr<InteractionAdaptorType>>
    {
        auto interactions = std::vector<std::unique_ptr<InteractionAdaptorType>>( active_count() );
        auto it = Interaction::generate_active_local<StandaloneFactory>( local, indices, interactions.begin() );
        Interaction::generate_active_nonlocal<StandaloneFactory>( nonlocal, it );
        return interactions;
    };

    [[nodiscard]] auto active_interactions() const -> std::vector<std::unique_ptr<const InteractionAdaptorType>>
    {
        auto interactions = std::vector<std::unique_ptr<InteractionAdaptorType>>( active_count() );
        auto it = Interaction::generate_active_local<StandaloneFactory>( local, indices, interactions.begin() );
        Interaction::generate_active_nonlocal<StandaloneFactory>( nonlocal, it );
        return interactions;
    };

    // compile time getter
    template<class T>
    [[nodiscard]] constexpr auto getInteraction() -> std::unique_ptr<InteractionAdaptorType>
    {
        if constexpr( hasInteraction_Local<T>() )
            return StandaloneFactory::make_standalone(
                Backend::get<Interaction::InteractionWrapper<T>>( local ), indices );
        else if constexpr( hasInteraction_NonLocal<T>() )
            return StandaloneFactory::make_standalone( Backend::get<Interaction::InteractionWrapper<T>>( nonlocal ) );
        else
            return nullptr;
    };

    // compile time getter
    template<class T>
    [[nodiscard]] constexpr auto getInteraction() const -> std::unique_ptr<const InteractionAdaptorType>
    {
        if constexpr( hasInteraction_Local<T>() )
            return StandaloneFactory::make_standalone(
                Backend::get<Interaction::InteractionWrapper<T>>( local ), indices );
        else if constexpr( hasInteraction_NonLocal<T>() )
            return StandaloneFactory::make_standalone( Backend::get<Interaction::InteractionWrapper<T>>( nonlocal ) );
        else
            return nullptr;
    };

    template<class T>
    [[nodiscard]] static constexpr bool hasInteraction()
    {
        return Utility::contains<Interaction::InteractionWrapper<T>, LocalInteractionTuple>::value
               || Utility::contains<Interaction::InteractionWrapper<T>, NonLocalInteractionTuple>::value;
    };

    void applyGeometry()
    {
        if( geometry == nullptr )
        {
            setPtrAddress();
            return;
        }

        if( index_storage.size() != static_cast<std::size_t>( geometry->nos ) )
        {
            index_storage = IndexStorageVector( geometry->nos, IndexStorageTuple{} );
        }
        else
        {
            Backend::cpu::for_each(
                index_storage.begin(), index_storage.end(),
                []( IndexStorageTuple & storage_tuple ) {
                    Backend::apply(
                        []( auto &... item ) { ( ..., Interaction::clearIndexStorage( item ) ); }, storage_tuple );
                } );
        }

        Backend::apply(
            [this]( auto &... interaction ) -> void
            { ( ..., interaction.applyGeometry( *geometry, boundary_conditions, index_storage ) ); }, local );

        Backend::apply(
            [this]( auto &... interaction ) -> void
            { ( ..., interaction.applyGeometry( *geometry, boundary_conditions ) ); }, nonlocal );

        updateIndexVector();
    }

    template<typename T>
    [[nodiscard]] auto data() const -> const typename T::Data *
    {
        if constexpr( hasInteraction_Local<T>() )
            return &Backend::get<Interaction::InteractionWrapper<T>>( local ).data;
        else if constexpr( hasInteraction_NonLocal<T>() )
            return &Backend::get<Interaction::InteractionWrapper<T>>( nonlocal ).data;
        else
            return nullptr;
    };

    template<typename T>
    [[nodiscard]] auto set_data( typename T::Data && data ) -> std::optional<std::string>
    {
        std::optional<std::string> error{};

        if constexpr( hasInteraction_Local<T>() )
        {
            error = Backend::get<Interaction::InteractionWrapper<T>>( local ).set_data( std::move( data ) );
            applyGeometry<T>();
        }
        else if constexpr( hasInteraction_NonLocal<T>() )
        {
            error = Backend::get<Interaction::InteractionWrapper<T>>( nonlocal ).set_data( std::move( data ) );
            applyGeometry<T>();
        }
        else
            error = fmt::format( "The Hamiltonian doesn't contain an interaction of type \"{}\"", T::name );

        return error;
    };

    template<typename T, typename... Args>
    [[nodiscard]] auto set_data( Args &&... args ) -> std::optional<std::string>
    {
        return set_data<T>( typename T::Data{ std::forward<Args>( args )... } );
    }

    template<typename T>
    [[nodiscard]] auto cache() const -> const typename T::Cache *
    {
        if constexpr( hasInteraction_Local<T>() )
            return &Backend::get<Interaction::InteractionWrapper<T>>( local ).cache;
        else if constexpr( hasInteraction_NonLocal<T>() )
            return &Backend::get<Interaction::InteractionWrapper<T>>( nonlocal ).cache;
        else
            return nullptr;
    };

    template<class T>
    [[nodiscard]] bool is_contributing() const
    {
        if constexpr( hasInteraction<T>() )
            return T::is_contributing( *data<T>(), *cache<T>() );
        else
            return false;
    }

    [[nodiscard]] const auto & get_boundary_conditions() const
    {
        return boundary_conditions;
    }

    void set_boundary_conditions( const intfield & bc )
    {
        boundary_conditions = bc;
        applyGeometry();
    }

    [[nodiscard]] const auto & get_geometry() const
    {
        return *geometry;
    }

    void set_geometry( const ::Data::Geometry & g )
    {
        set_geometry_impl( g );
    }

    void set_geometry( ::Data::Geometry && g )
    {
        set_geometry_impl( std::move( g ) );
    }

private:
    template<typename InteractionType>
    void applyGeometry()
    {
        static_assert( hasInteraction<InteractionType>() );

        if( geometry == nullptr )
        {
            setPtrAddress();
            return;
        }

        // TODO(feature-template_hamiltonian): turn this into an error
        if( geometry->nos != index_storage.size() )
            return;

        if constexpr( is_local<InteractionType>::value )
        {
            Backend::cpu::for_each(
                index_storage.begin(), index_storage.end(),
                []( IndexStorageTuple & index_tuple ) {
                    Interaction::clearIndexStorage(
                        Backend::get<typename InteractionType::IndexStorage>( index_tuple ) );
                } );

            Backend::get<Interaction::InteractionWrapper<InteractionType>>( local ).applyGeometry(
                *geometry, boundary_conditions, index_storage );

            updateIndexVector<InteractionType>();
        }
        else
        {
            Backend::get<Interaction::InteractionWrapper<InteractionType>>( nonlocal )
                .applyGeometry( *geometry, boundary_conditions );
        };
    };

    template<typename Geometry>
    void set_geometry_impl( Geometry && g )
    {
        static_assert( std::is_same_v<std::decay_t<Geometry>, ::Data::Geometry> );
        // lazy copy mechanism for the geometry
        // We allow shallow copies when the geometry stays the same,
        // but if we want to change it we ensure that we are the sole owner of the Geometry
        // This only works, because the Geometry class is only shared between Hamiltonian objects
        if( geometry.use_count() > 1 || geometry == nullptr )
        {
            geometry = std::make_shared<::Data::Geometry>( std::forward<Geometry>( g ) );
        }
        else
        {
            *geometry = std::forward<Geometry>( g );
        }
        applyGeometry();
    }

    template<typename InteractionType = void>
    void updateIndexVector()
    {
        static_assert( std::is_void<InteractionType>::value || hasInteraction<InteractionType>() );

        if constexpr( !std::is_void<InteractionType>::value )
        {
            for( int i = 0; i < indices.size(); ++i )
            {
                Backend::get<typename InteractionType::Index>( indices[i] ) = Interaction::make_index(
                    Backend::get<typename InteractionType::IndexStorage>( index_storage[i] ) );
            }
        }
        else
        {
            if( indices.size() != index_storage.size() )
                indices.resize( index_storage.size() );

            Backend::cpu::transform(
                index_storage.begin(), index_storage.end(), indices.begin(),
                []( const IndexStorageTuple & storage ) -> IndexTuple
                {
                    return Backend::apply(
                        []( const auto &... item )
                        { return Backend::make_tuple( Interaction::make_index( item )... ); }, storage );
                } );
        }
    };

    template<class T>
    [[nodiscard]] static constexpr bool hasInteraction_Local()
    {
        return is_local<T>::value
               && Utility::contains<Interaction::InteractionWrapper<T>, LocalInteractionTuple>::value;
    };

    template<class T>
    [[nodiscard]] static constexpr bool hasInteraction_NonLocal()
    {
        return !is_local<T>::value
               && Utility::contains<Interaction::InteractionWrapper<T>, NonLocalInteractionTuple>::value;
    };

protected:
    IndexVector indices;
    LocalInteractionTuple local;
    NonLocalInteractionTuple nonlocal;

private:
    std::shared_ptr<Data::Geometry> geometry;
    intfield boundary_conditions;

    IndexStorageVector index_storage;
};

} // namespace Common

} // namespace Engine

#endif
