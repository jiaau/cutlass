#include "cute/layout.hpp"
#include "cute/util/print.hpp"
#include "cute/tensor.hpp"
#include "cutlass/layout/matrix.h"

using namespace cute;

int main() {
    TiledMMA mma = make_tiled_mma(SM70_8x8x4_F32F16F16F32_NT{},
        Layout<Shape <_2,_2>,
               Stride<_2,_1>>{});   // 2x2 n-major layout of Atoms
    mma.get_slice(16);
}