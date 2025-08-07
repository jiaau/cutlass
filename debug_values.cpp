#include <iostream>
#include <string>
#include <vector>
#include <type_traits> // 需要 <type_traits> 来使用 std::integral_constant

// (复用上一节的工具)
// 第一步：类型调试器，用于触发编译错误并显示类型名称。
template <typename... Types>
void debug_types() {
    ( (void)Types::this_will_fail, ... );
}

// 第二步：我们新的 constexpr 值调试器
// 使用 C++17 的 `auto` 模板参数，可以接受任何可作为非类型模板参数的常量值
// (如 int, bool, char, enum 等)
template <auto... Values>
void debug_constexpr() {
    // 关键转换：
    // 对于 "Values" 参数包中的每一个值...
    // 1. 获取它的类型: decltype(Values)
    // 2. 将值和类型包装进 std::integral_constant
    // 3. 将生成的所有 std::integral_constant 类型传递给 debug_types
    debug_types<std::integral_constant<decltype(Values), Values>...>();
}

// --- 用于测试的 constexpr 函数和变量 ---
constexpr int factorial(int n) {
    return n <= 1 ? 1 : n * factorial(n - 1);
}

constexpr bool is_factorial_greater_than_100(int n) {
    return factorial(n) > 100;
}

int main() {
    // --- 触发 constexpr 调试 ---
    // 下面这行代码是我们的“断点”。
    // 编译它将会失败，并在错误信息中显示这些表达式在编译期的计算结果。
    debug_constexpr<
        42,                             // 一个普通的字面量
        factorial(5),                   // 一个 constexpr 函数调用的结果
        sizeof(long long),              // sizeof 表达式的结果
        alignof(std::string),           // alignof 表达式的结果
        is_factorial_greater_than_100(5),// 一个返回 bool 的 constexpr 函数的结果
        (1 + 2) * 3 == 9                // 一个纯粹的布尔表达式
    >();

    return 0;
}
/*
g++ -std=c++17 debug_values.cpp -o debug_values && rm debug_values
*/