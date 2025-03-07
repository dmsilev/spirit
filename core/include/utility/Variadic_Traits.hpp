#pragma once

#include <array>
#include <tuple>
#include <type_traits>

namespace Utility
{

template<template<class> class Pred, template<class...> class Variadic, class... Ts>
struct variadic_filter
{
private:
    template<template<class> class, class, class...>
    struct Impl;

    template<template<class> class P, class Sequence>
    struct Impl<P, Sequence>
    {
        using type = Sequence;
    };

    template<template<class> class P, class... I, class Head, class... Tail>
    struct Impl<P, Variadic<I...>, Head, Tail...>
    {
        using type = typename std::conditional<
            P<Head>::value, typename Impl<P, Variadic<I..., Head>, Tail...>::type,
            typename Impl<P, Variadic<I...>, Tail...>::type>::type;
    };

public:
    using type =
        typename std::conditional<sizeof...( Ts ) == 0, Variadic<>, typename Impl<Pred, Variadic<>, Ts...>::type>::type;
};

template<template<class> class Pred, template<class...> class Variadic, class... Ts>
using variadic_filter_t = typename variadic_filter<Pred, Variadic, Ts...>::type;

// generate index sequences for variadic parameter packs based on a filter condition
template<template<class> class, class...>
struct variadic_filter_index_sequence;

template<template<class> class Pred>
struct variadic_filter_index_sequence<Pred>
{
    using type = std::index_sequence<>;
};

template<template<class> class Pred, class... Ts>
struct variadic_filter_index_sequence
{
private:
    template<template<class> class, std::size_t, class, class...>
    struct Impl;

    template<template<class> class P, std::size_t Size, class Sequence>
    struct Impl<P, Size, Sequence>
    {
        using type = Sequence;
    };

    template<template<class> class P, std::size_t Size, std::size_t... I, class Head, class... Tail>
    struct Impl<P, Size, std::index_sequence<I...>, Head, Tail...>
    {
        using type = typename std::conditional<
            P<Head>::value, typename Impl<P, Size + 1, std::index_sequence<I..., Size>, Tail...>::type,
            typename Impl<P, Size + 1, std::index_sequence<I...>, Tail...>::type>::type;
    };

public:
    using type = typename std::conditional<
        sizeof...( Ts ) == 0, std::index_sequence<>, typename Impl<Pred, 0, std::index_sequence<>, Ts...>::type>::type;
};

// map metafunction for the elements of a variadic container
template<
    template<class> class F, template<class...> class Variadic, class T,
    template<class...> class NewVariadic = Variadic>
struct variadic_map;

template<template<class> class F, template<class...> class Variadic, template<class...> class NewVariadic, class... Ts>
struct variadic_map<F, Variadic, Variadic<Ts...>, NewVariadic>
{
    using type = NewVariadic<typename F<Ts>::type...>;
};

template<
    template<class> class F, template<class...> class Variadic, class T,
    template<class...> class NewVariadic = Variadic>
using variadic_map_t = typename variadic_map<F, Variadic, T, NewVariadic>::type;

// metafunction to transfer types from tuple to a different variadic type
template<template<typename...> typename, typename>
struct variadic_from_tuple;

template<template<typename...> typename Variadic, typename... T>
struct variadic_from_tuple<Variadic, std::tuple<T...>>
{
    using type = Variadic<T...>;
};

// metafunction to transfer types to tuple from a different variadic type
template<template<typename...> typename, typename>
struct variadic_to_tuple;

template<template<typename...> typename Variadic, typename... T>
struct variadic_to_tuple<Variadic, Variadic<T...>>
{
    using type = std::tuple<T...>;
};

// metafunction to test if all types within a variadic container are the same
template<class T, template<class...> class Variadic = std::tuple>
struct all_same;

template<template<class...> class Variadic>
struct all_same<Variadic<>, Variadic> : std::true_type
{
};

template<typename Head, typename... Tail, template<class...> class Variadic>
struct all_same<Variadic<Head, Tail...>, Variadic> : std::conjunction<std::is_same<Head, Tail>...>
{
};

template<class T>
static constexpr bool all_same_v = all_same<T>::value;

// trait to detemine if a type is contained in a tuple
template<typename T, typename Variadic>
struct contains;

template<typename T, typename... Us, template<typename...> typename Variadic>
struct contains<T, Variadic<Us...>> : std::disjunction<std::is_same<T, Us>...>
{
};

template<typename T, typename... Us>
static constexpr bool contains_v = contains<T, Us...>::value;

template<std::size_t I, typename Variadic>
struct contains_index;

template<std::size_t I, std::size_t... Js>
struct contains_index<I, std::index_sequence<Js...>> : std::disjunction<std::bool_constant<I == Js>...>
{
};

template<template<class> class F, typename T, typename = void>
struct has_attr_impl : std::false_type
{
};

template<template<class> class F, typename T>
struct has_attr_impl<F, T, std::void_t<typename F<T>::type>> : std::true_type
{
};

template<template<class> class F, typename T>
using has_attr = has_attr_impl<F, T>;

// test if two variadics are disjoint
template<typename, typename>
struct are_disjoint;

template<std::size_t... Is, std::size_t... Js>
struct are_disjoint<std::index_sequence<Is...>, std::index_sequence<Js...>>
        : std::negation<std::disjunction<contains_index<Is, std::index_sequence<Js...>>...>>
{
};

template<template<class...> class FirstVariadic, class... Ts, typename SecondVariadic>
struct are_disjoint<FirstVariadic<Ts...>, SecondVariadic>
        : std::negation<std::disjunction<contains<Ts, SecondVariadic>...>>
{
};

template<typename... Arrays>
struct arrays_to_tuple
{
private:
    template<typename...>
    struct concat;

    template<typename... Ts>
    struct concat<std::tuple<Ts...>>
    {
        using type = std::tuple<Ts...>;
    };

    template<typename... Ts, typename... Us, typename... Tuple>
    struct concat<std::tuple<Ts...>, std::tuple<Us...>, Tuple...>
    {
        using type = typename concat<std::tuple<Ts..., Us...>, Tuple...>::type;
    };

    template<typename>
    struct flatten;

    template<typename T, std::size_t N>
    struct flatten<std::array<T, N>>
    {
    private:
        template<typename>
        struct helper;

        template<std::size_t... Is>
        struct helper<std::index_sequence<Is...>>
        {
            using type = std::tuple<typename std::tuple_element<Is, std::array<T, N>>::type...>;
        };

    public:
        using type = typename helper<decltype( std::make_index_sequence<N>{} )>::type;
    };

public:
    using type = typename concat<typename flatten<Arrays>::type...>::type;
};

} // namespace Utility

static_assert( std::is_same_v<
               typename Utility::arrays_to_tuple<std::array<int, 3>, std::array<float, 2>>::type,
               std::tuple<int, int, int, float, float>> );

static_assert( std::is_same_v<
               typename Utility::arrays_to_tuple<std::array<int, 3>, std::array<float, 2>, std::array<int, 1>>::type,
               std::tuple<int, int, int, float, float, int>> );
