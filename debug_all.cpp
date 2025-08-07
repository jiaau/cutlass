#include <iostream>
#include <string>
#include <vector>

// =================================================================
// 1. 定义我们的 “显示器” 包装类型
// =================================================================

// 一个包装器，它的类型参数 T 就是我们想在编译错误中看到的类型。
template <typename T>
struct TypeDisplayer {
    // 我们在结构体内部保留一个会失败的成员，
    // 以便让所有使用它的地方都触发相似的编译错误。
    static constexpr bool this_will_fail = false;
};

// 另一个包装器，它的非类型模板参数 V 就是我们想看到的值。
template <auto V>
struct ValueDisplayer {
    static constexpr bool this_will_fail = false;
};

// =================================================================
// 2. 提供简单易用的辅助函数来创建这些包装器对象
// =================================================================

// 传入一个类型 T，返回一个 TypeDisplayer<T> 对象。
template <typename T>
constexpr TypeDisplayer<T> type_() {
    return {};
}

// 传入一个编译期常量 V，返回一个 ValueDisplayer<V> 对象。
template <auto V>
constexpr ValueDisplayer<V> value_() {
    return {};
}

// =================================================================
// 3. 统一的调试入口函数
// =================================================================

// 这个函数接受任意数量、任意种类的“显示器”对象作为参数。
template <typename... Displayers>
void debug_simultaneously(Displayers... displayers) {
    // 使用 C++17 的折叠表达式，对每一个传入的 "displayer" 对象
    // 触发一个依赖于其类型的、永远为假的 static_assert。
    // 编译器在报告断言失败时，会清晰地列出导致失败的 Displayers 类型。
    // 使用立即调用的 lambda 表达式，将 static_assert 包装成一个表达式
    // 创建一个通用的 lambda，它接受一个参数 T。
    // 然后，我们对函数参数包 `displayers` (注意是小写，代表函数参数) 进行折叠调用。
    ( [](auto T) {
        // 在 lambda 内部，我们使用 decltype(T) 来获取显示器对象的准确类型
        // (例如 TypeDisplayer<int> 或 ValueDisplayer<42>)，
        // 然后访问其静态成员来触发断言。
        static_assert(decltype(T)::this_will_fail, "Debugging...");
    }(displayers), ... ); // 注意：这里是将参数包中的每个对象依次传入 lambda
}


// =================================================================
// 4. 实际使用示例
// =================================================================

constexpr int factorial(int n) {
    return n <= 1 ? 1 : n * factorial(n - 1);
}

int main() {
    std::string my_string = "hello";
    const std::vector<int> my_vec;

    // --- 在一处同时调试类型和 constexpr 值 ---
    // 下面这行代码会触发编译失败，并在错误信息中显示所有你想看的信息。
    debug_simultaneously(
        type_<decltype(my_string)>(),         // 调试一个变量的类型
        value_<factorial(6)>(),               // 调试一个 constexpr 函数的结果
        type_<decltype(my_vec)&>(),           // 调试一个引用类型
        value_<sizeof(my_string)>,            // 调试一个 sizeof 的结果
        type_<int>(),                         // 调试一个内置类型
        value_<alignof(double) == 8>()        // 调试一个布尔表达式的结果
    );

    return 0;
}
/*
g++ -std=c++17 debug_all.cpp -o debug_all && rm debug_all
*/