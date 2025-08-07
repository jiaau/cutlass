#include "cute/layout.hpp"
#include <assert.h>

int main() {
    using namespace cute;
    auto ctensor = make_layout(make_shape(Int<8>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    auto t_tile = make_tile(Int<16>{}, Int<16>{});

    auto t_tensor = logical_divide(ctensor, t_tile); // (PermM,PermN)
    print_latex(ctensor);
    print_latex(t_tensor);
    // static_assert(size(t_tensor) == size(ctensor), "");
    // for(int i = 0; i < size(t_tensor); i ++) {
    //     assert(ctensor(i) == t_tensor(i));
    // }
    // std::cout << "pass" << std::endl;
    return 0;
}