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
  // constexpr int K = 32;

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

  using mma_op = SM80_16x8x16_F16F16F16F16_TN;

  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  static constexpr int kMmaEURepeatM = 2;
  static constexpr int kMmaEURepeatN = 2;
  static constexpr int kMmaEURepeatK = 1;

  using mma_atom_shape = mma_traits::Shape_MNK;
  static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
  static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
  static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

  using MMA_EU_RepeatT = decltype(make_layout(make_shape(
      Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
  using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;

  using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));
  using TiledMMA = MMA;

  TiledMMA tiled_mma;

  // get MMA tile for each thread
  auto thr_mma = tiled_mma.get_slice(31);
  auto tCrD = thr_mma.partition_fragment_C(gD); // (MMA, MMA_M, MMA_N)
  // print(tCrD);
}