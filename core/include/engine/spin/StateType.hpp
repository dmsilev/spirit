#pragma once

#include <engine/Vectormath_Defines.hpp>

#include <Eigen/Core>

namespace Engine
{

template<typename state_type>
struct state_traits;

namespace Spin
{

enum struct Field
{
    Spin = 0,
};

template<typename T>
struct quantity
{
    T spin;
};

template<typename T>
struct quantity<field<T>>
{
    field<T> spin;

    auto data() -> typename state_traits<quantity<field<T>>>::pointer
    {
        return { spin.data() };
    }

    auto data() const -> typename state_traits<quantity<field<T>>>::const_pointer
    {
        return { spin.data() };
    }
};

template<typename T>
constexpr auto make_quantity( T && value ) -> quantity<std::decay_t<T>>
{
    return { std::forward<T>( value ) };
}

template<Field field, typename T>
T & get( quantity<T> & q )
{
    if constexpr( field == Field::Spin )
        return q.spin;
}

template<Field field, typename T>
const T & get( const T & q )
{
    if constexpr( field == Field::Spin )
        return q.spin;
}

using StateType = quantity<vectorfield>;
using StatePtr  = quantity<Vector3 *>;
using StateCPtr = quantity<const Vector3 *>;

} // namespace Spin

template<typename state_type>
struct state_traits;

template<typename T>
struct state_traits<Spin::quantity<T>>
{
    using type          = Spin::quantity<T>;
    using pointer       = Spin::quantity<typename T::pointer>;
    using const_pointer = Spin::quantity<typename T::const_pointer>;
};

template<typename state_t>
state_t make_state( int nos );

template<>
inline Spin::StateType make_state( int nos )
{
    return { vectorfield( nos ) };
};

} // namespace Engine
