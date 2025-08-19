#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cute/util/print.hpp"
#include "cutlass/layout/matrix.h"
#include <cuda.h>

using namespace cute;

int main() {
  using namespace cute;
  using T = half;

  constexpr int M = 128;
  constexpr int N = 128;

  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

  using G2SCopyA =
      decltype(make_tiled_copy(g2s_copy_atom{},
                               make_layout(make_shape(Int<32>{}, Int<4>{}),
                                           make_stride(Int<4>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using G2SCopyB = G2SCopyA;

  // create global memory for D
  T *Dptr;
  T *Dptr_host;
  Dptr_host = (T *)malloc(sizeof(T) * M * N);
  cudaMalloc(&Dptr, sizeof(T) * M * N);
  cudaMemcpy(Dptr, Dptr_host, sizeof(T) * M * N, cudaMemcpyHostToDevice);
  Tensor D = make_tensor(make_gmem_ptr((T *)Dptr), make_shape(M, N),
                         make_stride(N, Int<1>{})); // (M, N)

  // tile D for each CTA
  constexpr int TileM = 128;
  constexpr int TileN = 128;
  Tensor gD = local_tile(D, make_tile(Int<TileM>{}, Int<TileN>{}),
                         make_coord(0, 0)); // (M, N)

  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(0);
  auto tAgA_copy = g2s_thr_copy_a.partition_S(gD); // (CPY, CPY_M, CPY_K, k)

}