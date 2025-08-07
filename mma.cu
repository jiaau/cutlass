#include <cuda.h>
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"
#include "cutlass/layout/matrix.h"

using namespace cute;

int main() {
    using namespace cute;
    using T = half;

    constexpr int M = 16;
    constexpr int N = 16;

    // create global memory for D
    T *Dptr;
    T *Dptr_host;
    Dptr_host = (T *)malloc(sizeof(T) * M * N);
    cudaMalloc(&Dptr, sizeof(T) * M * N);
    cudaMemcpy(Dptr, Dptr_host, sizeof(T) * M * N, cudaMemcpyHostToDevice);
    Tensor D =
        make_tensor(make_gmem_ptr((T *)Dptr), make_shape(M, N), make_stride(N, Int<1>{})); // (M, N)

    // tile D for each CTA
    constexpr int TileM = 16;
    constexpr int TileN = 16;
    Tensor gD = local_tile(D, make_tile(Int<TileM>{}, Int<TileN>{}), make_coord(0, 0)); // (M, N)

    TiledMMA tiled_mma =
        make_tiled_mma(SM70_8x8x4_F32F16F16F32_NT{},
                       Layout<Shape<_2, _2>, Stride<_2, _1>>{}); // 2x2 n-major layout of Atoms

    // get MMA tile for each thread
    auto thr_mma = tiled_mma.get_slice(31);
    auto tCrD = thr_mma.partition_fragment_C(gD); // (MMA, MMA_M, MMA_N)
}