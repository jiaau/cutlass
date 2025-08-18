#pragma once

#include <stdio.h>
#include <type_traits>
#include "cute/util/print.hpp"

template <typename... Ts>
void debug_types() {
    ( (void)Ts::this_will_fail, ... );
}

template <auto... Values>
void debug_constexpr() {
    debug_types<std::integral_constant<decltype(Values), Values>...>();
}

template <typename T>
struct TypeDisplayer {
    static constexpr bool this_will_fail = false;
};

template <auto V>
struct ValueDisplayer {
    static constexpr bool this_will_fail = false;
};

template <typename T>
constexpr TypeDisplayer<T> type_() {
    return {};
}

template <auto V>
constexpr ValueDisplayer<V> value_() {
    return {};
}

template <typename... Displayers>
void debug_simultaneously(Displayers... displayers) {
    ( [](auto T) {
        static_assert(decltype(T)::this_will_fail, "Debugging...");
    }(displayers), ... );
    ( [](auto T) {
        static_assert(decltype(T)::this_will_fail, "Debugging...");
    }(displayers), ... );
}

template <typename Layout>
void print_1d_layout(Layout layout) {
    cute::print(layout);
    puts("");
    for (int i = 0; i < size(layout); i++) {
        cute::print(layout(i));
        printf(" ");
    }
}

// 编译时输出器 - 接受常量字符串
template<int N, const char* Msg = nullptr>
struct DebugMsg {
    [[deprecated]]
    static constexpr void print() {}
};

// 简洁的编译时输出宏
#define COMPILE_TIME_PRINT(msg) \
    do { \
        static constexpr const char debug_str[] = msg; \
        DebugMsg<__LINE__, debug_str>::print(); \
    } while(0)
    