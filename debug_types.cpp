#include <iostream>
#include <string>
#include <vector>
#include <type_traits> // 仅用于方案二

// 方案一: C++17 折叠表达式 (推荐)
// Ts 是一个模板参数包，可以代表 0 个或多个类型
template <typename... Ts>
void debug_types() {
    // 这个折叠表达式会为参数包中的每一个类型 T 实例化 T::this_will_fail;
    // (void) 是为了处理当 Ts 为空时的情况，并抑制 "unused variable" 警告。
    ( (void)Ts::this_will_fail, ... );
}

int main() {
    std::cout << "准备触发编译期类型调试..." << std::endl;

    // --- 我们想要调试的变量 ---
    std::string my_string = "hello";
    const int x = 42;
    std::vector<double> my_vec;
    auto& ref_to_string = my_string;

    // --- 触发调试 ---
    // 下面这行代码就是我们的“断点”。
    // 我们使用 decltype 来获取变量的类型，并将它们传递给 debug_types。
    // 编译这行代码将会失败，并在错误信息中显示这些类型。
    debug_types<decltype(my_string), decltype(x), decltype(my_vec), decltype(ref_to_string)>();

    // 因为编译会失败，所以这行代码永远不会被执行。
    std::cout << "编译成功！" << std::endl;
    return 0;
}

/*
g++ -std=c++17 debug_types.cpp -o debug_types && rm debug_types
*/
