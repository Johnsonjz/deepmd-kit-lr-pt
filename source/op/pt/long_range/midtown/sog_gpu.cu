// sog_gpu.cu — plugin-internal GPU SOG kspace (raw CUDA + cuFFT). Tier-2 full-GPU pipeline:
//   spread(q@x) -> cuFFT fwd -> green-multiply (grad meshes + potential + energy/6-virial reductions)
//   -> one cuFFT inverse of the mesh potential -> adjoint weight-derivative gather.
// Two splines: separable QuadS (PRIMARY — 3D tensor-product 1D weights, GPU-friendly scatter once
// cuFFT makes the FFT ~free and spreading dominates) and CubeS2-4 (BASELINE — bit-matches CPU sog.cpp).
// Math ported from sog.cpp::compute_single. STATUS: nvcc-syntax-compiled; NOT yet GPU-bit-validated
// against the CPU path (cuFFT is unnormalized vs LAMMPS FFT3d 1/ngrid — the normalization constants
// below must be confirmed by the run-0 bit-match once the GPU frees). See sog-progress.md §8.
#include "sog_gpu.cuh"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <dlfcn.h>
#include "sog_spline.h"     // CubeS2 node tables + xi constants (single source of truth)
#include <cufft.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"sog_gpu CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); }}while(0)

// ── device state ──
struct SogGpuState {
  int nx=0, ny=0, nz=0; size_t ngrid=0;
  cufftHandle plan=0;                 // Z2Z, 3D (replicated, full grid)
  // Distributed (slab) k-space plans.  The (x,y) pair is real-to-complex: the
  // density in the brick is real, so only the compact half-spectrum (Nh =
  // nx/2+1) is stored/transported along x.  The z transform stays full-length
  // complex (the +/-kz partners are needed after the x compression).
  cufftHandle plan_xy_rc=0, plan_xy_cr=0, plan_z=0;
  int slab_ZL=-1, slab_YL=-1, slab_Nh=-1;
  // Whether the fp64 mesh set (d_rho/d_pot/d_ge/gf/gv) is allocated.  The
  // replicated solve never reads it, so a single-precision caller skips it and
  // this keeps the realloc predicate honest across a switch.
  int double_mesh=1;
  // Single-precision replicated path: the mesh, the spectrum and the green
  // tables are float.  Every GPU here is a consumer card whose FP64 pipe is
  // 1/64 of FP32, so double mesh work is the single largest cost.  The
  // charges, the energies and every reduction still accumulate in double.
  cufftHandle f_plan=0;
  cufftComplex *f_rho=nullptr, *f_pot=nullptr;
  float2 *f_gp=nullptr;   // (green energy, green virial) packed, one load per mode
  // Half-spectrum path: real charge and potential meshes, half spectrum.
  float *f_rho_r=nullptr, *f_pot_r=nullptr;
  unsigned long long *f_rho_fx=nullptr;  // fixed-point charge mesh (2^40)
  cufftComplex *f_spec=nullptr;
  // ik-mode buffers: three gradient meshes and a half-spectrum scratch, reused
  // across the three inverse transforms (allocated on demand by ensure_ik_grads).
  float *f_gradx_r=nullptr, *f_grady_r=nullptr, *f_gradz_r=nullptr;
  cufftComplex *f_grad_spec=nullptr;
  // QuadS ik-mode buffers: the QuadS solve keeps a full complex spectrum, so
  // the three Fourier gradients and their scratch spectrum are full grids.
  cufftComplex *f_qgx=nullptr, *f_qgy=nullptr, *f_qgz=nullptr, *f_qgspec=nullptr;
  cufftHandle f_plan_r2c=0, f_plan_c2r=0;
  // One work area shared by the R2C and C2R plans (they never run at the same
  // time); cuFFT would otherwise keep one per plan, 4 B per grid point each.
  void *f_fft_work=nullptr;
  cufftDoubleComplex *d_rho=nullptr;  // charge mesh / rho_hat (in place)
  cufftDoubleComplex *d_gx=nullptr, *d_gy=nullptr, *d_gz=nullptr, *d_pot=nullptr;
  double *d_ge=nullptr, *d_gf=nullptr, *d_gv=nullptr;  // green energy/force/virial tables
  // (The bare K and K_v "self" tables are gone: the self term is closed-form,
  //  see self_coeff in sog_gpu_solve_dev, so no per-mode table is uploaded.)
  double *d_x=nullptr, *d_q=nullptr, *d_force=nullptr, *d_vpot=nullptr;  // per-atom
  float *d_brick=nullptr;              // local owned+ghost FLOAT potential brick
  unsigned long long *s_brick_fx=nullptr;  // local owned+ghost fixed-point (2^40) density brick
  size_t cap_brick=0;
  double *d_boxlo=nullptr;           // 3 doubles, box origin (persistent device buffer)
  // Triclinic: A^{-1} of the cell (row-major, row d = fractional axis), the
  // true volume |det A|, and a flag.  With the flag the kernels evaluate in
  // fractional coordinates on a unit cell, the k vectors come from the metric,
  // and the volume replaces lx*ly*lz.
  double *d_hinv=nullptr;            // 9 doubles on the device
  double volume=0.0;
  int metric=0;
  double *d_red=nullptr;             // energy + 6 fv_local + 6 r⊗F (20 doubles; slots 7..13 unused)
  int cap_atoms=0;

  // GPU timing (cudaEvent-based, accumulated per N steps)
  cudaEvent_t evt_spread_start=0, evt_spread_stop=0;
  cudaEvent_t evt_fftfwd_start=0, evt_fftfwd_stop=0;
  cudaEvent_t evt_kspace_start=0, evt_kspace_stop=0;
  cudaEvent_t evt_fftinv_start=0, evt_fftinv_stop=0;
  cudaEvent_t evt_gather_start=0, evt_gather_stop=0;
  double acc_spread=0, acc_fftfwd=0, acc_kspace=0, acc_fftinv=0, acc_gather=0;
  int timing_count=0, timing_interval=100;  // report every N steps

  // ── slab (distributed) mode ──
  // Same precision contract as the replicated solve: a float32 mesh, a 64-bit
  // fixed-point (2^40) density accumulation, a compact half spectrum (R2C) and
  // a packed (green energy, green virial) float table.  The fp64 d_rho / d_pot
  // grid set belongs to the LAMMPS replicated-grid entry points alone; the slab
  // keeps its own s_* buffers so the two never alias across a mode switch.
  bool slab_mode=false;
  int slab_Y=0;                 // this rank's y length (pencil slab)
  size_t slab_cap=0;            // complex elements in s_rho / s_pot
  size_t slab_gn=0;             // float2 entries in s_gp
  cufftComplex *s_rho=nullptr, *s_pot=nullptr;  // packed blocks / z pencil
  float2 *s_gp=nullptr;         // (green energy, green virial), compact half spectrum
  // scratch for uploading a received halo plane before it is folded in
  void *d_halo=nullptr;
  size_t cap_halo=0;            // capacity in 8-byte elements
  // aligned FLOAT scratch for R2C/C2R when the brick offset is not 16-aligned
  float *s_real=nullptr;
  size_t cap_real=0;

  // Per-atom fractional coordinates of the replicated solve (k_midtown_frac).
  void *d_frac=nullptr;
  int cap_frac=0;

};

// ── helpers ──
// The reciprocal length comes from the host: three correctly-rounded fp64
// divisions per atom cost ~15 us at 12288 atoms on a card whose fp64 pipe is
// 1/64 of fp32, and the CPU can form 1/L exactly once.
__device__ __forceinline__ double periodic_frac_i(double xi, double lo, double inv){
  double t = (xi - lo) * inv;
  t -= floor(t);
  return t;
}

__device__ __forceinline__ double periodic_frac(double xi, double lo, double L){
  double t=(xi-lo)/L; t-=floor(t); return t;
}
__device__ __forceinline__ int wrap(int i,int n){ i%=n; return i<0? i+n : i; }
// The stencil offsets move the index by at most two cells, so a compare and a
// conditional add replaces the integer division that `wrap` compiles to (the
// divider is one of the slowest instructions on the device).
__device__ __forceinline__ int wrap_near(int i,int n){
  if(i<0) i+=n; else if(i>=n) i-=n;
  return i;
}
__device__ __forceinline__ size_t midx(int ix,int iy,int iz,int nx,int ny){
  return (size_t)iz*ny*nx + (size_t)iy*nx + ix;   // matches sog.cpp mesh_index (x fastest)
}

// ── separable QuadS 1D weights — correct port of sog_spline.h / quads_spline.py ──
// True midtown quadrature splines (§A.2/§A.3): ξ○ = 1/√3 (order 4), 0.72879488 (order 6),
// weights c₁…c_ν with reflection c_i(θ)=c_{1-i}(1-θ). Offsets 1−ν…ν. (The earlier port used
// CubeS₂'s L and ξ=2/√15 — WRONG; that has been replaced.)
__device__ const double kQuadsXi6_d = 0.72879488;   // 0.72879488² etc computed inline
// order 4 (ξ○=1/√3 ⇒ ξ²=1/3): c₁ = -½t³+½t²-(3ξ²-2)/2·t+ξ²/2 ; c₂ = ⅙t³+(3ξ²-1)/6·t
__device__ __forceinline__ double q_c1_4(double t){ const double x2=1.0/3.0;
  return -0.5*t*t*t + 0.5*t*t - (3.0*x2-2.0)/2.0*t + 0.5*x2; }
__device__ __forceinline__ double q_c2_4(double t){ const double x2=1.0/3.0;
  return (1.0/6.0)*t*t*t + (3.0*x2-1.0)/6.0*t; }
__device__ __forceinline__ void quads4_w(double t, double *w /*[4]*/){   // offsets -1,0,1,2
  w[0]=q_c2_4(1.0-t); w[1]=q_c1_4(1.0-t); w[2]=q_c1_4(t); w[3]=q_c2_4(t);
}
__device__ __forceinline__ double q_dc1_4(double t){ const double x2=1.0/3.0;
  return -1.5*t*t+t-(3.0*x2-2.0)/2.0; }
__device__ __forceinline__ double q_dc2_4(double t){ const double x2=1.0/3.0;
  return 0.5*t*t+(3.0*x2-1.0)/6.0; }
__device__ __forceinline__ void quads4_wd(double t,double*w,double*dw){
  quads4_w(t,w); dw[0]=-q_dc2_4(1.0-t); dw[1]=-q_dc1_4(1.0-t);
  dw[2]=q_dc1_4(t); dw[3]=q_dc2_4(t);
}
// order 6 (ξ○=0.72879488): c₁,c₂,c₃ degree-5 (§A.3)
__device__ __forceinline__ double q_c1_6(double t,double x2,double x4){
  double t2=t*t,t3=t2*t,t4=t3*t,t5=t4*t;
  return (1.0/12.0)*t5 - (1.0/6.0)*t4 + (10.0*x2-7.0)/12.0*t3
       - (3.0*x2-2.0)/3.0*t2 + (5.0*x4-7.0*x2+4.0)/4.0*t - (3.0*x4-4.0*x2)/6.0; }
__device__ __forceinline__ double q_c2_6(double t,double x2,double x4){
  double t2=t*t,t3=t2*t,t4=t3*t,t5=t4*t;
  return -(1.0/24.0)*t5 + (1.0/24.0)*t4 - (10.0*x2-7.0)/24.0*t3
       + (6.0*x2-1.0)/24.0*t2 - (5.0*x4-7.0*x2+2.0)/8.0*t + (3.0*x4-x2)/24.0; }
__device__ __forceinline__ double q_c3_6(double t,double x2,double x4){
  double t3=t*t*t,t5=t3*t*t;
  return (1.0/120.0)*t5 + (2.0*x2-1.0)/24.0*t3 + (15.0*x4-15.0*x2+4.0)/120.0*t; }
__device__ __forceinline__ void quads6_w(double t, double *w /*[6]*/){   // offsets -2,-1,0,1,2,3
  const double x2=kQuadsXi6_d*kQuadsXi6_d, x4=x2*x2;
  w[0]=q_c3_6(1.0-t,x2,x4); w[1]=q_c2_6(1.0-t,x2,x4); w[2]=q_c1_6(1.0-t,x2,x4);
  w[3]=q_c1_6(t,x2,x4);     w[4]=q_c2_6(t,x2,x4);     w[5]=q_c3_6(t,x2,x4);
}
__device__ __forceinline__ double q_dc1_6(double t,double x2,double x4){
  double t2=t*t,t3=t2*t,t4=t3*t;
  return (5.0/12.0)*t4-(2.0/3.0)*t3+(10.0*x2-7.0)/4.0*t2
       -2.0*(3.0*x2-2.0)/3.0*t+(5.0*x4-7.0*x2+4.0)/4.0; }
__device__ __forceinline__ double q_dc2_6(double t,double x2,double x4){
  double t2=t*t,t3=t2*t,t4=t3*t;
  return -(5.0/24.0)*t4+(1.0/6.0)*t3-(10.0*x2-7.0)/8.0*t2
       +(6.0*x2-1.0)/12.0*t-(5.0*x4-7.0*x2+2.0)/8.0; }
__device__ __forceinline__ double q_dc3_6(double t,double x2,double x4){
  double t2=t*t,t4=t2*t2;
  return (1.0/24.0)*t4+(2.0*x2-1.0)/8.0*t2+(15.0*x4-15.0*x2+4.0)/120.0; }
__device__ __forceinline__ void quads6_wd(double t,double*w,double*dw){
  const double x2=kQuadsXi6_d*kQuadsXi6_d,x4=x2*x2; quads6_w(t,w);
  dw[0]=-q_dc3_6(1.0-t,x2,x4); dw[1]=-q_dc2_6(1.0-t,x2,x4);
  dw[2]=-q_dc1_6(1.0-t,x2,x4); dw[3]=q_dc1_6(t,x2,x4);
  dw[4]=q_dc2_6(t,x2,x4); dw[5]=q_dc3_6(t,x2,x4);
}

// Sorted QuadS stencil table: the (2nu)^3 tensor-product nodes with their
// grid offsets (dx,dy,dz) and their 1-D weight indices (a,b,c), ordered with x
// fastest.  A warp reads consecutive entries, so the lanes of one round touch
// adjacent mesh cells; this mirrors the CubeS2 c_nodes*s tables and removes the
// per-node div/mod decode that dominated the naive gather.
struct QNodeS { signed char dx,dy,dz,a,b,c,pad0,pad1; };  // 8 B, aligned
__constant__ QNodeS c_qnodes4s[64];
__constant__ QNodeS c_qnodes6s[216];

// The 1-D weights are identical for every lane of a warp (all lanes share the
// atom), so evaluate them once on lane 0 and broadcast.  For order 6 the
// degree-5 polynomials cost far more than the shuffles; computing them in all
// 32 lanes was the dominant per-node cost.
// float weights: the mesh is float and the accumulate stays double; keeping
// the 3x2xNORD arrays in float halves the register footprint, which is what
// limits occupancy of the warp gather.
template<int NORD>
__device__ __forceinline__ void bcast_quads_w(double t,double*w,int lane){
  double W[NORD]={0.0};
  if(lane==0){ if(NORD==4) quads4_w(t,W); else quads6_w(t,W); }
  #pragma unroll
  for(int i=0;i<NORD;++i) w[i]=__shfl_sync(0xffffffffu,W[i],0);
}
template<int NORD>
__device__ __forceinline__ void bcast_quads_wd(double t,double*w,double*dw,
                                               int lane){
  double W[NORD]={0.0},D[NORD]={0.0};
  if(lane==0){ if(NORD==4) quads4_wd(t,W,D); else quads6_wd(t,W,D); }
  #pragma unroll
  for(int i=0;i<NORD;++i){
    w[i] =__shfl_sync(0xffffffffu,W[i],0);
    dw[i]=__shfl_sync(0xffffffffu,D[i],0);
  }
}

// ── CubeS₂ device port — EXACT port of sog_spline.h (bit-matches CPU sog.cpp) ──
// Node tables live in __constant__ memory (uploaded once at create). GpuNode layout == CubeS2Node4.
typedef LAMMPS_NS::CubeS2Node4 GpuNode;
__constant__ GpuNode c_nodes4[32];
__constant__ GpuNode c_nodes6[88];
// The same tables sorted by grid offset (z slowest, x fastest).  A warp walks
// consecutive table entries, so a sorted table makes the lanes of one round
// touch adjacent mesh cells, whose atomics and loads coalesce into a few cache
// lines instead of scattering over the whole mesh.
__constant__ GpuNode c_nodes4s[32];
__constant__ GpuNode c_nodes6s[88];
__device__ const double kXi4 = 0.5773502691896258;   // 1/√3  (kCubes2Xi4)
__device__ const double kXi6 = 0.6503998764035732;   // kCubes2Xi6

__device__ __forceinline__ double cs2_L(double th,double xi){ double x2=xi*xi;
  return -0.5*th*th*th + 0.5*th*th - (9.0*x2-2.0)/6.0*th + 0.5*x2; }
__device__ __forceinline__ double cs2_R(double th,double xi){ double x2=xi*xi;
  return (1.0/6.0)*th*th*th + (3.0*x2-1.0)/6.0*th; }
__device__ __forceinline__ double eta_pos_off(int off,double c){ return (off==1)?c:(1.0-c); }

__device__ __forceinline__ float cs2_w4(float tx,float ty,float tz,const GpuNode&nd,float xi){
  if(nd.cls==0){
    float ex=(nd.dx==1)?tx:(1.0-tx), ey=(nd.dy==1)?ty:(1.0-ty), ez=(nd.dz==1)?tz:(1.0-tz);
    return cs2_L(ex,xi)*ey*ez + cs2_L(ey,xi)*ex*ez + cs2_L(ez,xi)*ex*ey;
  } else {
    float es,e1,e2;
    if(nd.sp_axis==0){ es=nd.sp_is_neg?(1.0-tx):tx; e1=(nd.dy==1)?ty:(1.0-ty); e2=(nd.dz==1)?tz:(1.0-tz); }
    else if(nd.sp_axis==1){ es=nd.sp_is_neg?(1.0-ty):ty; e1=(nd.dx==1)?tx:(1.0-tx); e2=(nd.dz==1)?tz:(1.0-tz); }
    else { es=nd.sp_is_neg?(1.0-tz):tz; e1=(nd.dx==1)?tx:(1.0-tx); e2=(nd.dy==1)?ty:(1.0-ty); }
    return cs2_R(es,xi)*e1*e2;
  }
}

__device__ __forceinline__ float cs2_L111_6(float t,float xi){ float x2=xi*xi,x4=x2*x2;
  return (1.0/12.0)*t*t*t*t*t - (1.0/6.0)*t*t*t*t + (10.0*x2-1.0)/12.0*t*t*t
       - (6.0*x2-1.0)/6.0*t*t + (5.0*x4-x2)/4.0*t - (3.0*x4-x2)/6.0; }
__device__ __forceinline__ float cs2_L311_6(float t,float xi){ float x2=xi*xi,x4=x2*x2;
  return (1.0/120.0)*t*t*t*t*t + (2.0*x2-1.0)/24.0*t*t*t + (15.0*x4-15.0*x2+4.0)/120.0*t; }
__device__ __forceinline__ float cs2_L211_6(float t,float xi){ return -0.25*cs2_L111_6(t,xi)-2.5*cs2_L311_6(t,xi); }
__device__ __forceinline__ float cs2_S6(float t,float xi){
  return -0.5*t*t*t + 0.5*t*t - (3.0*xi*xi-2.0)/2.0*t + 0.5*xi*xi; }

__device__ __forceinline__ float cs2_w6(float tx,float ty,float tz,const GpuNode&nd,float xi){
  float coords[3]={tx,ty,tz}; int offs[3]={nd.dx,nd.dy,nd.dz};
  if(nd.cls==0){
    float ex=(nd.dx==1)?tx:(1.0-tx), ey=(nd.dy==1)?ty:(1.0-ty), ez=(nd.dz==1)?tz:(1.0-tz);
    return cs2_L111_6(ex,xi)*ey*ez + cs2_L111_6(ey,xi)*ez*ex + cs2_L111_6(ez,xi)*ex*ey
         + cs2_S6(ex,xi)*cs2_S6(ey,xi)*cs2_S6(ez,xi);
  } else if(nd.cls==1){
    int sp=nd.sp_axis; float es=nd.sp_is_neg?(1.0-coords[sp]):coords[sp];
    int n1=(sp+1)%3,n2=(sp+2)%3; if(n1>n2){int t=n1;n1=n2;n2=t;}
    return cs2_L211_6(es,xi)*eta_pos_off(offs[n1],coords[n1])*eta_pos_off(offs[n2],coords[n2])
         + cs2_R(es,xi)*cs2_S6(eta_pos_off(offs[n1],coords[n1]),xi)*cs2_S6(eta_pos_off(offs[n2],coords[n2]),xi);
  } else if(nd.cls==2){
    int sp=nd.sp_axis; float es=nd.sp_is_neg?(1.0-coords[sp]):coords[sp];
    int n1=(sp+1)%3,n2=(sp+2)%3; if(n1>n2){int t=n1;n1=n2;n2=t;}
    return cs2_L311_6(es,xi)*eta_pos_off(offs[n1],coords[n1])*eta_pos_off(offs[n2],coords[n2]);
  } else if(nd.cls==3){
    int nm=nd.sp_axis; int s1=(nm+1)%3,s2=(nm+2)%3; if(s1>s2){int t=s1;s1=s2;s2=t;}
    float en=eta_pos_off(offs[nm],coords[nm]);
    float es1=(nd.sp_is_neg&1)?(1.0-coords[s1]):coords[s1];
    float es2=(nd.sp_is_neg&2)?(1.0-coords[s2]):coords[s2];
    return cs2_S6(en,xi)*cs2_R(es1,xi)*cs2_R(es2,xi);
  } else {
    float ex=(nd.sp_is_neg&1)?(1.0-tx):tx, ey=(nd.sp_is_neg&2)?(1.0-ty):ty, ez=(nd.sp_is_neg&4)?(1.0-tz):tz;
    return cs2_R(ex,xi)*cs2_R(ey,xi)*cs2_R(ez,xi);
  }
}

struct GpuWeightGrad { float v, dx, dy, dz; };
__device__ __forceinline__ GpuWeightGrad gg(float v,float dx=0,float dy=0,float dz=0){return {v,dx,dy,dz};}
__device__ __forceinline__ GpuWeightGrad ga(GpuWeightGrad a,GpuWeightGrad b){return gg(a.v+b.v,a.dx+b.dx,a.dy+b.dy,a.dz+b.dz);}
__device__ __forceinline__ GpuWeightGrad gs(GpuWeightGrad a,GpuWeightGrad b){return gg(a.v-b.v,a.dx-b.dx,a.dy-b.dy,a.dz-b.dz);}
__device__ __forceinline__ GpuWeightGrad gm(GpuWeightGrad a,GpuWeightGrad b){return gg(a.v*b.v,a.dx*b.v+a.v*b.dx,a.dy*b.v+a.v*b.dy,a.dz*b.v+a.v*b.dz);}
__device__ __forceinline__ GpuWeightGrad gc(float a,GpuWeightGrad b){return gg(a*b.v,a*b.dx,a*b.dy,a*b.dz);}
// fp32 literals throughout: these run once per node of every atom in the
// gather, and a double literal promotes the whole polynomial to the fp64 pipe,
// which is 1/64 of the fp32 one on the cards this runs on.  Measured on a 4090
// at 250^3 / 196608 atoms, the fp64 form costs 1.0 ms and the fp32 form 0.24 ms
// for the same 88 evaluations per atom.
__device__ __forceinline__ GpuWeightGrad gL(GpuWeightGrad t,float xi){auto t2=gm(t,t),t3=gm(t2,t);return ga(ga(gc(-.5f,t3),gc(.5f,t2)),ga(gc(-(9.f*xi*xi-2.f)/6.f,t),gg(.5f*xi*xi)));}
__device__ __forceinline__ GpuWeightGrad gR(GpuWeightGrad t,float xi){return ga(gc(1.f/6.f,gm(gm(t,t),t)),gc((3.f*xi*xi-1.f)/6.f,t));}
__device__ __forceinline__ GpuWeightGrad gL111(GpuWeightGrad t,float xi){float x2=xi*xi,x4=x2*x2;auto t2=gm(t,t),t3=gm(t2,t),t4=gm(t3,t),t5=gm(t4,t);return ga(ga(ga(gc(1.f/12.f,t5),gc(-1.f/6.f,t4)),ga(gc((10.f*x2-1.f)/12.f,t3),gc(-(6.f*x2-1.f)/6.f,t2))),ga(gc((5.f*x4-x2)/4.f,t),gg(-(3.f*x4-x2)/6.f)));}
__device__ __forceinline__ GpuWeightGrad gL311(GpuWeightGrad t,float xi){float x2=xi*xi,x4=x2*x2;auto t3=gm(gm(t,t),t),t5=gm(gm(t3,t),t);return ga(ga(gc(1.f/120.f,t5),gc((2.f*x2-1.f)/24.f,t3)),gc((15.f*x4-15.f*x2+4.f)/120.f,t));}
__device__ __forceinline__ GpuWeightGrad gL211(GpuWeightGrad t,float xi){return gs(gc(-.25f,gL111(t,xi)),gc(2.5f,gL311(t,xi)));}
__device__ __forceinline__ GpuWeightGrad gS6(GpuWeightGrad t,float xi){auto t2=gm(t,t),t3=gm(t2,t);return ga(ga(gc(-.5f,t3),gc(.5f,t2)),ga(gc(-(3.f*xi*xi-2.f)/2.f,t),gg(.5f*xi*xi)));}
__device__ __forceinline__ GpuWeightGrad gcoord(float v,int a,bool refl=false){return gg(v,a==0?(refl?-1:1):0,a==1?(refl?-1:1):0,a==2?(refl?-1:1):0);}
__device__ __forceinline__ GpuWeightGrad geta(const GpuWeightGrad*c,const int*o,int a){return o[a]==1?c[a]:gcoord(1-c[a].v,a,true);}
__device__ __forceinline__ GpuWeightGrad cs2_wg4(float tx,float ty,float tz,const GpuNode&nd,float xi){
  GpuWeightGrad c[3]={gcoord(tx,0),gcoord(ty,1),gcoord(tz,2)};int o[3]={nd.dx,nd.dy,nd.dz};
  if(nd.cls==0){auto ex=geta(c,o,0),ey=geta(c,o,1),ez=geta(c,o,2);return ga(ga(gm(gm(gL(ex,xi),ey),ez),gm(gm(gL(ey,xi),ex),ez)),gm(gm(gL(ez,xi),ex),ey));}
  int sp=nd.sp_axis,n1=(sp+1)%3,n2=(sp+2)%3;if(n1>n2){int z=n1;n1=n2;n2=z;}auto es=nd.sp_is_neg?gcoord(1-c[sp].v,sp,true):c[sp];return gm(gm(gR(es,xi),geta(c,o,n1)),geta(c,o,n2));
}
__device__ __forceinline__ GpuWeightGrad cs2_wg6(float tx,float ty,float tz,const GpuNode&nd,float xi){
  GpuWeightGrad c[3]={gcoord(tx,0),gcoord(ty,1),gcoord(tz,2)};int o[3]={nd.dx,nd.dy,nd.dz};
  if(nd.cls==0){auto ex=geta(c,o,0),ey=geta(c,o,1),ez=geta(c,o,2);return ga(ga(ga(gm(gm(gL111(ex,xi),ey),ez),gm(gm(gL111(ey,xi),ez),ex)),gm(gm(gL111(ez,xi),ex),ey)),gm(gm(gS6(ex,xi),gS6(ey,xi)),gS6(ez,xi)));}
  if(nd.cls==1||nd.cls==2){int sp=nd.sp_axis,n1=(sp+1)%3,n2=(sp+2)%3;if(n1>n2){int z=n1;n1=n2;n2=z;}auto es=nd.sp_is_neg?gcoord(1-c[sp].v,sp,true):c[sp];if(nd.cls==1)return ga(gm(gm(gL211(es,xi),geta(c,o,n1)),geta(c,o,n2)),gm(gm(gR(es,xi),gS6(geta(c,o,n1),xi)),gS6(geta(c,o,n2),xi)));return gm(gm(gL311(es,xi),geta(c,o,n1)),geta(c,o,n2));}
  if(nd.cls==3){int nm=nd.sp_axis,s1=(nm+1)%3,s2=(nm+2)%3;if(s1>s2){int z=s1;s1=s2;s2=z;}auto e1=(nd.sp_is_neg&1)?gcoord(1-c[s1].v,s1,true):c[s1];auto e2=(nd.sp_is_neg&2)?gcoord(1-c[s2].v,s2,true):c[s2];return gm(gm(gS6(geta(c,o,nm),xi),gR(e1,xi)),gR(e2,xi));}
  auto ex=(nd.sp_is_neg&1)?gcoord(1-tx,0,true):c[0];auto ey=(nd.sp_is_neg&2)?gcoord(1-ty,1,true):c[1];auto ez=(nd.sp_is_neg&4)?gcoord(1-tz,2,true):c[2];return gm(gm(gR(ex,xi),gR(ey,xi)),gR(ez,xi));
}

// ── fp32 1-D basis of the order-4 transfer function ─────────────────────
//
// The literals above are ``double``, so an evaluation of cs2_L or cs2_R
// promotes to the fp64 pipe, which runs at 1/64 of the fp32 one on the cards
// this runs on.  The order-4 spread builds its per-atom table from these
// instead.
__device__ __forceinline__ float cs2_fL(float t,float x2){
  return -0.5f*t*t*t + 0.5f*t*t - (9.0f*x2-2.0f)/6.0f*t + 0.5f*x2; }
__device__ __forceinline__ float cs2_fR(float t,float x2){
  return (1.0f/6.0f)*t*t*t + (3.0f*x2-1.0f)/6.0f*t; }

// ── Per-atom 1-D table of the order-4 spread ────────────────────────────
//
// The order-4 transfer function is a sum of products of three 1-D functions
// (one per node class), so a table of the 1-D values of an atom's own (t, 1-t)
// replaces one polynomial evaluation per node.  The pack evaluates the table
// once per atom in fp32 and scatters 32 nodes from it, which is what the order
// 6 pack already did: without it the order-4 spread cost 1.11 ms at 250^3 /
// 196608 atoms, more than the order-6 spread despite having a third of the
// nodes, because its per-node `cs2_w4` ran in fp64.
//
// Layout, per axis a, at base = a * 10: kind*2 + e for kind 0 elt, 1 L, 2 R,
// e = 0 reading f(t) and e = 1 f(1-t); then eta[i], i = offset + 1 in [0, 4).
struct Cs2Tab4 { float d[30]; };

#define CS2_4V(b,k,a,e) ((b).d[(a)*10 + (k)*2 + (e)])
#define CS2_4E(b,a,i)   ((b).d[(a)*10 + 6 + (i)])

__device__ __forceinline__ void cs2_fill_tab4(Cs2Tab4& b, const float* t,
                                              float xi, int lane) {
  if (lane >= 30) return;
  const float x2 = xi*xi;
  const int a = lane / 10, r = lane - a*10;
  const float v = t[a], u = 1.f - v;
  float out;
  if (r < 6) {
    const int kind = r >> 1, e = r & 1;
    const float arg = e ? u : v;
    out = (kind == 0) ? arg : (kind == 1 ? cs2_fL(arg, x2) : cs2_fR(arg, x2));
  } else {
    const int i = r - 6;
    out = (i == 2) ? v : u;
  }
  b.d[lane] = out;
}

__device__ __forceinline__ float cs2_w4_tab(const Cs2Tab4& b, const GpuNode& nd) {
  const int o[3] = {nd.dx + 1, nd.dy + 1, nd.dz + 1};
  if (nd.cls == 0) {
    const int ix = 1-nd.dx, iy = 1-nd.dy, iz = 1-nd.dz;
    const float ex = CS2_4V(b,0,0,ix), ey = CS2_4V(b,0,1,iy), ez = CS2_4V(b,0,2,iz);
    return CS2_4V(b,1,0,ix)*ey*ez + CS2_4V(b,1,1,iy)*ez*ex
         + CS2_4V(b,1,2,iz)*ex*ey;
  }
  const int sp = nd.sp_axis;
  int n1 = (sp+1)%3, n2 = (sp+2)%3;
  const int neg = nd.sp_is_neg ? 1 : 0;
  return CS2_4V(b,2,sp,neg) * CS2_4E(b,n1,o[n1]) * CS2_4E(b,n2,o[n2]);
}

// Zero a single-precision mesh.
__global__ void k_zero_complex_f(size_t n,cufftComplex*rho){
  size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n){ rho[i].x=0.f; rho[i].y=0.f; }
}

// CubeS2 weights from a per-atom table of one-dimensional values.
//
// The transfer function is a sum of a few products of 1-D functions (that is
// what the midtown node table encodes), so the per-node evaluation that the
// branchy cs2_w6 does can be replaced by a handful of multiplies once the 1-D
// values are known.  These are all functions of the fractional coordinate
// alone, so one thread computes them for its atom and every node of the
// stencil reads them.
struct Cs2Tab {
  float elt[3][2];    // t and 1-t
  float L111[3][2], L211[3][2], L311[3][2], S6[3][2], R[3][2];
  float eta[3][4];    // offsets -1, 0, 1, 2
  float S6eta[3][4];
};

__device__ __forceinline__ void cs2_fill_tab(Cs2Tab& b, float tx, float ty,
                                             float tz, float xi) {
  const float t[3] = {tx, ty, tz};
  for (int a = 0; a < 3; ++a) {
    const float v = t[a], u = 1.f - v;
    b.elt[a][0] = v;
    b.elt[a][1] = u;
    b.L111[a][0] = cs2_L111_6(v, xi);
    b.L111[a][1] = cs2_L111_6(u, xi);
    b.L211[a][0] = cs2_L211_6(v, xi);
    b.L211[a][1] = cs2_L211_6(u, xi);
    b.L311[a][0] = cs2_L311_6(v, xi);
    b.L311[a][1] = cs2_L311_6(u, xi);
    b.S6[a][0] = cs2_S6(v, xi);
    b.S6[a][1] = cs2_S6(u, xi);
    b.R[a][0] = cs2_R(v, xi);
    b.R[a][1] = cs2_R(u, xi);
    const float e[4] = {eta_pos_off(-1, v), eta_pos_off(0, v),
                        eta_pos_off(1, v), eta_pos_off(2, v)};
    for (int i = 0; i < 4; ++i) {
      b.eta[a][i] = e[i];
      b.S6eta[a][i] = cs2_S6(e[i], xi);
    }
  }
}

// The same table, filled by every lane of the warp: the struct is 60 floats
// with no padding, so the flat view can be split over the 32 lanes.  One lane
// filling it serializes 42 polynomial evaluations per atom while the other 31
// wait, which made the spread kernel five times slower than its atomics.
__device__ __forceinline__ void cs2_fill_tab_par(Cs2Tab& b, float tx, float ty,
                                                 float tz, float xi, int lane) {
  float* f = reinterpret_cast<float*>(&b);
  const float t[3] = {tx, ty, tz};
  #pragma unroll
  for (int s = lane; s < 60; s += 32) {
    int a, e;
    float val;
    if (s < 6) {
      a = s >> 1;
      e = s & 1;
      const float v = t[a];
      val = e ? (1.f - v) : v;
    } else if (s < 12) {
      a = (s - 6) >> 1;
      e = s & 1;
      val = cs2_L111_6(e ? (1.f - t[a]) : t[a], xi);
    } else if (s < 18) {
      a = (s - 12) >> 1;
      e = s & 1;
      val = cs2_L211_6(e ? (1.f - t[a]) : t[a], xi);
    } else if (s < 24) {
      a = (s - 18) >> 1;
      e = s & 1;
      val = cs2_L311_6(e ? (1.f - t[a]) : t[a], xi);
    } else if (s < 30) {
      a = (s - 24) >> 1;
      e = s & 1;
      val = cs2_S6(e ? (1.f - t[a]) : t[a], xi);
    } else if (s < 36) {
      a = (s - 30) >> 1;
      e = s & 1;
      val = cs2_R(e ? (1.f - t[a]) : t[a], xi);
    } else if (s < 48) {
      const int q = s - 36;
      a = q >> 2;
      const int i = q & 3;
      val = eta_pos_off(i - 1, t[a]);
    } else {
      const int q = s - 48;
      a = q >> 2;
      const int i = q & 3;
      val = cs2_S6(eta_pos_off(i - 1, t[a]), xi);
    }
    f[s] = val;
  }
}

__device__ __forceinline__ float cs2_w6_tab(const Cs2Tab& b, const GpuNode& nd) {
  const int o[3] = {nd.dx + 1, nd.dy + 1, nd.dz + 1};
  if (nd.cls == 0) {
    // The stencil picks t at offset 1 and 1-t at offset 0.
    const int ix = 1 - nd.dx, iy = 1 - nd.dy, iz = 1 - nd.dz;
    const float ex = b.elt[0][ix], ey = b.elt[1][iy], ez = b.elt[2][iz];
    return b.L111[0][ix] * ey * ez + b.L111[1][iy] * ez * ex +
           b.L111[2][iz] * ex * ey +
           b.S6[0][ix] * b.S6[1][iy] * b.S6[2][iz];
  }
  if (nd.cls == 1 || nd.cls == 2) {
    int sp = nd.sp_axis, n1 = (sp + 1) % 3, n2 = (sp + 2) % 3;
    if (n1 > n2) {
      const int t = n1;
      n1 = n2;
      n2 = t;
    }
    const int neg = nd.sp_is_neg ? 1 : 0;
    const float a1 = b.eta[n1][o[n1]], a2 = b.eta[n2][o[n2]];
    if (nd.cls == 1) {
      return b.L211[sp][neg] * a1 * a2 +
             b.R[sp][neg] * b.S6eta[n1][o[n1]] * b.S6eta[n2][o[n2]];
    }
    return b.L311[sp][neg] * a1 * a2;
  }
  if (nd.cls == 3) {
    int nm = nd.sp_axis, s1 = (nm + 1) % 3, s2 = (nm + 2) % 3;
    if (s1 > s2) {
      const int t = s1;
      s1 = s2;
      s2 = t;
    }
    return b.S6eta[nm][o[nm]] * b.R[s1][(nd.sp_is_neg & 1) ? 1 : 0] *
           b.R[s2][(nd.sp_is_neg & 2) ? 1 : 0];
  }
  return b.R[0][(nd.sp_is_neg & 1) ? 1 : 0] * b.R[1][(nd.sp_is_neg & 2) ? 1 : 0] *
         b.R[2][(nd.sp_is_neg & 4) ? 1 : 0];
}

// Fixed-point (2^40) charge mesh: integer atomics are faster than float ones
// and make the spread reproducible, as in the B-spline sampler.
constexpr double kFxScale = 1099511627776.0;  // 2^40

__global__ void k_zero_fx(size_t n,unsigned long long*p){
  size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) p[i]=0ull;
}

__global__ void k_fx_to_float(size_t n,const unsigned long long*p,float*out,double inv){
  size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) out[i]=(float)(inv*(double)(long long)p[i]);
}

// ── The fractional coordinate of every atom, once ─────────────────────────
//
// Placing an atom on the mesh takes three fp64 divisions, three floors and
// three subtractions -- about twenty fp64 instructions.  Inside the gather the
// whole warp computes the SAME atom's coordinate, so the pipeline executes
// those twenty instructions for one atom at a time: an fp64 warp instruction
// occupies the fp64 pipe whatever the number of its active lanes, and on a card
// whose fp64 pipe runs at 1/64 of its fp32 pipe that is the single largest term
// of the gather.  Measured on a 4090 over the two CubeS2 orders, the
// part of the gather that does not scale with the node count is 0.64 ms of
// 1.00 ms at 250^3 / 196608 atoms.
//
// One thread per atom puts thirty-two atoms in one warp instruction instead:
// the same twenty instructions now place 32 atoms rather than 1.  The pack
// (0.75 ms at the same size) and the gather both read the result back from
// L1 -- all lanes of a warp read the same 24-byte entry, which is a broadcast.
struct FracNode { int i[3]; float t[3]; };

__global__ void k_midtown_frac(int nlocal,const double*x,const double*boxlo,
    double ilx,double ily,double ilz,int nx,int ny,int nz,FracNode*out){
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=nlocal) return;
  const double fx=periodic_frac_i(x[3*i+0],boxlo[0],ilx)*nx;
  const double fy=periodic_frac_i(x[3*i+1],boxlo[1],ily)*ny;
  const double fz=periodic_frac_i(x[3*i+2],boxlo[2],ilz)*nz;
  FracNode& o=out[i];
  o.i[0]=(int)floor(fx); o.i[1]=(int)floor(fy); o.i[2]=(int)floor(fz);
  o.t[0]=(float)(fx-o.i[0]); o.t[1]=(float)(fy-o.i[1]); o.t[2]=(float)(fz-o.i[2]);
}

// The buffer of k_midtown_frac, grown to hold ``nlocal`` atoms.
static FracNode* ensure_frac(SogGpuState*s,int nlocal){
  if(nlocal > s->cap_frac){
    cudaFree(s->d_frac);
    s->cap_frac = nlocal > 0 ? nlocal : 1;
    CK(cudaMalloc(&s->d_frac, (size_t)s->cap_frac * sizeof(FracNode)));
  }
  return static_cast<FracNode*>(s->d_frac);
}

template<int ORD>
__global__ void k_spread_cubes2_warp_fx(int nlocal,const double*x,const double*boxlo,
    double lx,double ly,double lz,const double*q,double rho_scale,
    int nx,int ny,int nz,unsigned long long*rho){
  const int NN=(ORD==4)?32:88;
  __shared__ GpuNode sh_nodes[88];
  for(int t=threadIdx.x;t<NN;t+=blockDim.x)
    sh_nodes[t]=(ORD==4)?c_nodes4s[t]:c_nodes6s[t];
  __syncthreads();
  const int lane=threadIdx.x&31;
  const int i=(blockIdx.x*(blockDim.x>>5))+(threadIdx.x>>5);
  if(i>=nlocal) return;
  const double fx=periodic_frac(x[3*i+0],boxlo[0],lx)*nx;
  const double fy=periodic_frac(x[3*i+1],boxlo[1],ly)*ny;
  const double fz=periodic_frac(x[3*i+2],boxlo[2],lz)*nz;
  const int ix0=(int)floor(fx),iy0=(int)floor(fy),iz0=(int)floor(fz);
  const double tx=fx-ix0,ty=fy-iy0,tz=fz-iz0;
  const double xi=(ORD==4)?kXi4:kXi6;
  // The per-atom 1-D table is filled by every lane of the warp: one lane
  // filling it serializes forty polynomial evaluations behind a barrier while
  // the other thirty-one wait.
  __shared__ Cs2Tab sh_tab[8];
  const int wslot=threadIdx.x>>5;
  if (ORD==6) cs2_fill_tab_par(sh_tab[wslot],(float)tx,(float)ty,(float)tz,(float)kXi6,lane);
  __syncwarp();
  const Cs2Tab& tab=sh_tab[wslot];
  const float qs=(float)rho_scale*(float)q[i];
  // The atomics are what this kernel waits on, so the fixed-point conversion
  // stays on the FP32 pipe: an fp64 multiply plus llrint per node costs 5% of
  // the kernel on a card whose fp64 throughput is 1/64 of fp32.
  const float kfs=(float)kFxScale;
  // Unrolled so the atomics of one lane are independent and in flight together.
  #pragma unroll
  for(int k=lane;k<NN;k+=32){
    const GpuNode& nd=sh_nodes[k];
    const float w=(ORD==4)?cs2_w4(tx,ty,tz,nd,xi):cs2_w6_tab(tab,nd);
    if(w==0.0f) continue;
    const int gx=wrap_near(ix0+nd.dx,nx),gy=wrap_near(iy0+nd.dy,ny),gz=wrap_near(iz0+nd.dz,nz);
    atomicAdd(&rho[midx(gx,gy,gz,nx,ny)],
              (unsigned long long)(long long)__float2ll_rn(qs*w*kfs));
  }
}

// ── Packed spread, four atoms per warp ───────────────────────────────────
// The scatter is not what this kernel waits on: the per-atom weight table is.
// The degree-5 CubeS2 weights are evaluated in fp64 -- their literals are
// `double`, so `(1.0/12.0)*t*t*t*t*t` promotes the whole polynomial -- which on
// a card whose fp64 pipe is 1/64 of fp32 costs about 120 fp64 operations per
// atom.  One warp filling ONE table therefore serializes that behind a warp
// barrier.  Here four atoms share a warp: lanes 0..3 compute the four
// positions, 24 lanes (six per atom, one polynomial each) fill the four tables
// in ONE body, and the 88-entry stencil is then scattered for all four atoms
// from shared memory.  The result is bit-identical to the unpacked kernel.
constexpr int kSpreadPack = 4;
constexpr int kSpreadWarps = 8;

template <int ORD>
__global__ void __launch_bounds__(kSpreadWarps * 32)
    k_spread_cubes2_warp_fx_packed(int nlocal, const FracNode* fr,
                                   const double* q, double rho_scale,
                                   int nx, int ny, int nz,
                                   unsigned long long* rho) {
  constexpr int PACK = kSpreadPack;
  const int NN = (ORD == 4) ? 32 : 88;
  __shared__ GpuNode sh_nodes[88];
  __shared__ long long sh_off[88];  // dx + nx*(dy + ny*dz), no wrap needed
  __shared__ Cs2Tab sh_tab[kSpreadWarps][PACK];
  __shared__ float sh_t[kSpreadWarps][PACK][3];
  __shared__ float sh_qs[kSpreadWarps][PACK];
  __shared__ int sh_ix[kSpreadWarps][PACK];
  __shared__ int sh_iy[kSpreadWarps][PACK];
  __shared__ int sh_iz[kSpreadWarps][PACK];
  __shared__ int sh_intr[kSpreadWarps][PACK];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int t = threadIdx.x; t < NN; t += blockDim.x) {
    sh_nodes[t] = (ORD == 4) ? c_nodes4s[t] : c_nodes6s[t];
    sh_off[t] = static_cast<long long>(sh_nodes[t].dx) +
                static_cast<long long>(nx) *
                    (sh_nodes[t].dy + static_cast<long long>(ny) * sh_nodes[t].dz);
  }
  __syncthreads();

  const int a0 = (blockIdx.x * kSpreadWarps + warp) * PACK;
  // ── the four positions, one lane each ──
  if (lane < PACK) {
    const int i = a0 + lane;
    if (i < nlocal) {
      const FracNode& o = fr[i];
      const int ix0 = o.i[0], iy0 = o.i[1], iz0 = o.i[2];
      sh_ix[warp][lane] = ix0;
      sh_iy[warp][lane] = iy0;
      sh_iz[warp][lane] = iz0;
      sh_t[warp][lane][0] = o.t[0];
      sh_t[warp][lane][1] = o.t[1];
      sh_t[warp][lane][2] = o.t[2];
      sh_qs[warp][lane] = (float)rho_scale * (float)q[i];
      // The stencil reaches -2..+3 on every axis: an atom this far from the
      // boundary never wraps, so its scatter needs no per-node modulo.
      sh_intr[warp][lane] = (ix0 >= 2 && ix0 + 3 < nx && iy0 >= 2 &&
                             iy0 + 3 < ny && iz0 >= 2 && iz0 + 3 < nz)
                                ? 1
                                : 0;
    } else {
      sh_ix[warp][lane] = 0;
      sh_iy[warp][lane] = 0;
      sh_iz[warp][lane] = 0;
      sh_t[warp][lane][0] = 0.f;
      sh_t[warp][lane][1] = 0.f;
      sh_t[warp][lane][2] = 0.f;
      sh_qs[warp][lane] = 0.f;
      // NOT the interior path.  sh_intr means "this atom's stencil is inside
      // the mesh, so base + sh_off[k] needs no wrap"; a padding lane sits at
      // (0,0,0), whose stencil reaches -2, so the interior path would index
      // rho[-2] and rho[-2 - nx], i.e. BEFORE the mesh.  The value written is
      // zero (qs == 0), so no result was ever wrong -- but the address is out
      // of bounds and faulted as soon as the allocation layout moved.
      sh_intr[warp][lane] = 0;
    }
  }
  __syncwarp();
  // ── the four tables, six lanes each, one polynomial per lane ──
  if (lane < 6 * PACK) {
    const int a = lane / 6;
    const int q6 = lane - a * 6;
    const int ax = q6 >> 1;
    const int e = q6 & 1;
    const float tv = sh_t[warp][a][ax];
    const float arg = e ? (1.f - tv) : tv;
    const float xi = (float)kXi6;
    Cs2Tab& T = sh_tab[warp][a];
    T.elt[ax][e] = arg;
    T.L111[ax][e] = cs2_L111_6(arg, xi);
    T.L311[ax][e] = cs2_L311_6(arg, xi);
    T.S6[ax][e] = cs2_S6(arg, xi);
    T.R[ax][e] = cs2_R(arg, xi);
    // The reference combines the already-rounded float L111/L311, in double.
    T.L211[ax][e] =
        (float)(-0.25 * (double)T.L111[ax][e] - 2.5 * (double)T.L311[ax][e]);
    // eta/S6eta are permutations of the values above, not new polynomials.
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int src = (i == 2) ? 0 : 1;
      T.eta[ax][i] = T.elt[ax][src];
      T.S6eta[ax][i] = T.S6[ax][src];
    }
  }
  __syncwarp();

  // ── scatter, four atoms from the shared tables ──
  const float kfs = (float)kFxScale;
  #pragma unroll
  for (int a = 0; a < PACK; ++a) {
    const float qs = sh_qs[warp][a];
    const Cs2Tab& T = sh_tab[warp][a];
    if (sh_intr[warp][a]) {
      const long long base =
          static_cast<long long>(sh_ix[warp][a]) +
          static_cast<long long>(nx) *
              (sh_iy[warp][a] + static_cast<long long>(ny) * sh_iz[warp][a]);
      #pragma unroll
      for (int k = lane; k < NN; k += 32) {
        const float w = cs2_w6_tab(T, sh_nodes[k]);
        if (w == 0.0f) continue;
        atomicAdd(&rho[base + sh_off[k]],
                  (unsigned long long)(long long)__float2ll_rn(qs * w * kfs));
      }
    } else {
      const int ix0 = sh_ix[warp][a], iy0 = sh_iy[warp][a], iz0 = sh_iz[warp][a];
      #pragma unroll
      for (int k = lane; k < NN; k += 32) {
        const GpuNode& nd = sh_nodes[k];
        const float w = cs2_w6_tab(T, nd);
        if (w == 0.0f) continue;
        const int gx = wrap_near(ix0 + nd.dx, nx);
        const int gy = wrap_near(iy0 + nd.dy, ny);
        const int gz = wrap_near(iz0 + nd.dz, nz);
        atomicAdd(&rho[midx(gx, gy, gz, nx, ny)],
                  (unsigned long long)(long long)__float2ll_rn(qs * w * kfs));
      }
    }
  }
}

// Order 4 has no packed kernel: its spread used to evaluate the branchy
// cs2_w4 once per node, per lane, in fp64, which made its per-node cost higher
// than order 6's despite having a third of the nodes.  One warp per atom with
// a per-warp table is enough here, because the 32 nodes are exactly one round
// of a warp.
__global__ void __launch_bounds__(kSpreadWarps * 32)
    k_spread_cubes2_warp_fx_tab4(int nlocal, const FracNode* fr,
                                 const double* q, double rho_scale,
                                 int nx, int ny, int nz,
                                 unsigned long long* rho) {
  constexpr int NN = 32;
  __shared__ GpuNode sh_nodes[NN];
  __shared__ Cs2Tab4 sh_tab[kSpreadWarps];
  for (int t = threadIdx.x; t < NN; t += blockDim.x) {
    sh_nodes[t] = c_nodes4s[t];
  }
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int i = blockIdx.x * (blockDim.x >> 5) + warp;
  if (i >= nlocal) return;
  const FracNode& o = fr[i];
  const int ix0 = o.i[0], iy0 = o.i[1], iz0 = o.i[2];
  const float tf[3] = {o.t[0], o.t[1], o.t[2]};
  cs2_fill_tab4(sh_tab[warp], tf, (float)kXi4, lane);
  __syncwarp();
  const GpuNode& nd = sh_nodes[lane];
  const float w = cs2_w4_tab(sh_tab[warp], nd);
  if (w == 0.f) return;
  const float qs = (float)rho_scale * (float)q[i];
  const int gx = wrap_near(ix0 + nd.dx, nx), gy = wrap_near(iy0 + nd.dy, ny),
            gz = wrap_near(iz0 + nd.dz, nz);
  atomicAdd(&rho[midx(gx, gy, gz, nx, ny)],
            (unsigned long long)(long long)__float2ll_rn(qs * w * (float)kFxScale));
}

// ── Half-spectrum (real-to-complex) path ─────────────────────────────────
// The charge mesh and the potential mesh are real, so the transforms are R2C
// and C2R and the spectrum holds only the positive half of the fastest axis
// (x here, the contiguous one).  The k-space pass then visits half the modes
// and weighs the interior ones twice, which is exactly the full sum.

template<int ORD>
__global__ void k_gather_cubes2_warp_r(int nlocal,const FracNode*fr,
    double ilx,double ily,double ilz,const double*q,double qscale,
    int nx,int ny,int nz,const float*pot,int want_pot,
    double*force,double*vpot,double*red){
  constexpr int NN=(ORD==4)?32:88;
  constexpr int kRounds=(NN+31)/32;
  __shared__ GpuNode sh_nodes[88];
  for(int t=threadIdx.x;t<NN;t+=blockDim.x)
    sh_nodes[t]=(ORD==4)?c_nodes4s[t]:c_nodes6s[t];
  __syncthreads();
  const int lane=threadIdx.x&31;
  const int warp=threadIdx.x>>5;
  const int i=(blockIdx.x*(blockDim.x>>5))+warp;
  if(i>=nlocal) return;
  const FracNode& fo=fr[i];
  const int ix0=fo.i[0],iy0=fo.i[1],iz0=fo.i[2];
  const float tf[3]={fo.t[0],fo.t[1],fo.t[2]};
  double dtx=0,dty=0,dtz=0,gp=0;
  // A lane past the end of the table re-reads the last node with a zero
  // potential instead of breaking out of the loop: the break would stop the
  // compiler from unrolling, which is what keeps the scattered potential reads
  // of one lane in flight together.
  #pragma unroll
  for(int round=0;round<kRounds;++round){
    const int k=lane+32*round;
    const bool live=(k<NN);
    const GpuNode& nd=sh_nodes[live?k:(NN-1)];
    const GpuWeightGrad wg=(ORD==4)
        ?cs2_wg4(tf[0],tf[1],tf[2],nd,(float)kXi4)
        :cs2_wg6(tf[0],tf[1],tf[2],nd,(float)kXi6);
    const int gxi=wrap_near(ix0+nd.dx,nx),gyi=wrap_near(iy0+nd.dy,ny),gzi=wrap_near(iz0+nd.dz,nz);
    const double u=live?(double)pot[midx(gxi,gyi,gzi,nx,ny)]:0.0;
    dtx+=wg.dx*u; dty+=wg.dy*u; dtz+=wg.dz*u; gp+=wg.v*u;
  }
  #pragma unroll
  for(int off=16;off>0;off>>=1){
    dtx+=__shfl_down_sync(0xffffffffu,dtx,off);
    dty+=__shfl_down_sync(0xffffffffu,dty,off);
    dtz+=__shfl_down_sync(0xffffffffu,dtz,off);
    gp +=__shfl_down_sync(0xffffffffu,gp ,off);
  }
  if(lane!=0) return;
  const double qi=q[i];
  const double fxs=-qscale*qi*dtx*nx*ilx, fys=-qscale*qi*dty*ny*ily, fzs=-qscale*qi*dtz*nz*ilz;
  force[3*i+0]=fxs; force[3*i+1]=fys; force[3*i+2]=fzs;
  vpot[i]=qscale*gp;
}

// ── ik-mode (Fourier differentiation) path ───────────────────────────────
// Instead of gathering the weight derivatives (the AD path above), multiply the
// potential half-spectrum by i·k_a and invert, which yields dphi/dr_a on the
// mesh directly.  The gather then reads the three gradient meshes with the
// value-only weights, so the force is -qscale·q_i·(dphi/dr) with no nx/lx
// rescale (the derivative is already in physical units).  p_F = 2ν, one order
// above the AD path, at the cost of three extra C2R transforms.

__global__ void k_mul_ik(int nx,int ny,int nz,double kfx,double kfy,double kfz,
                         const cufftComplex*spec,cufftComplex*out,int axis){
  const int nh=nx/2+1;
  const size_t total=(size_t)nh*ny*nz;
  const size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(idx>=total) return;
  // cuFFT's R2C output of a (nz, ny, nx) real array is (nz, ny, nh) complex,
  // row-major, with the half axis kh CONTIGUOUS.  Same layout k_kspace_h
  // walks, and getting it transposed silently applies the wrong k to every
  // mode.
  const int kh=(int)(idx%nh);
  const int iy=(int)((idx/nh)%ny);
  const int iz=(int)(idx/((size_t)nh*ny));
  const int my=iy-(2*iy>=ny?ny:0);
  const int mz=iz-(2*iz>=nz?nz:0);
  const int even_nx=(2*nh-1!=nx)?1:0;
  // Self-conjugate points (k == -k): there the coefficient of a real field is
  // real, so i*k_a*phi_hat is pure imaginary and cannot stay Hermitian -- the
  // mode must be dropped.  Only these points; every other point of a Nyquist
  // plane is a genuine conjugate pair inside the half spectrum and must stay.
  const bool kx_self=(kh==0)||(even_nx&&kh==nh-1);
  const bool ky_self=(iy==0)||((ny%2)==0&&iy==ny/2);
  const bool kz_self=(iz==0)||((nz%2)==0&&iz==nz/2);
  const bool self=kx_self&&ky_self&&kz_self;
  double ka;
  if(axis==0) ka=kfx*kh;
  else if(axis==1) ka=kfy*my;
  else ka=kfz*mz;
  if(self || ka==0.0){ out[idx].x=0.f; out[idx].y=0.f; return; }
  // (rx + i ry) · i·ka = -ka·ry + i·ka·rx
  const double rx=(double)spec[idx].x, ry=(double)spec[idx].y;
  out[idx].x=(float)(-ka*ry);
  out[idx].y=(float)( ka*rx);
}

template<int ORD>
__global__ void k_gather_cubes2_warp_r_ik(int nlocal,const FracNode*fr,
    const double*q,double qscale,int nx,int ny,int nz,
    const float*pot,const float*gradx,const float*grady,const float*gradz,
    int want_pot,double*force,double*vpot){
  constexpr int NN=(ORD==4)?32:88;
  constexpr int kRounds=(NN+31)/32;
  __shared__ GpuNode sh_nodes[88];
  for(int t=threadIdx.x;t<NN;t+=blockDim.x)
    sh_nodes[t]=(ORD==4)?c_nodes4s[t]:c_nodes6s[t];
  __syncthreads();
  const int lane=threadIdx.x&31;
  const int warp=threadIdx.x>>5;
  const int i=(blockIdx.x*(blockDim.x>>5))+warp;
  if(i>=nlocal) return;
  const FracNode& fo=fr[i];
  const int ix0=fo.i[0],iy0=fo.i[1],iz0=fo.i[2];
  const float tf[3]={fo.t[0],fo.t[1],fo.t[2]};
  double dtx=0,dty=0,dtz=0,gp=0;
  #pragma unroll
  for(int round=0;round<kRounds;++round){
    const int k=lane+32*round;
    const bool live=(k<NN);
    const GpuNode& nd=sh_nodes[live?k:(NN-1)];
    const float w=(ORD==4)?cs2_w4(tf[0],tf[1],tf[2],nd,(float)kXi4)
                          :cs2_w6(tf[0],tf[1],tf[2],nd,(float)kXi6);
    const int gxi=wrap_near(ix0+nd.dx,nx),gyi=wrap_near(iy0+nd.dy,ny),gzi=wrap_near(iz0+nd.dz,nz);
    const int m=live?midx(gxi,gyi,gzi,nx,ny):0;
    const float ux=live?gradx[m]:0.f, uy=live?grady[m]:0.f, uz=live?gradz[m]:0.f;
    const float u=live?pot[m]:0.f;
    dtx+=w*ux; dty+=w*uy; dtz+=w*uz; gp+=w*u;
  }
  #pragma unroll
  for(int off=16;off>0;off>>=1){
    dtx+=__shfl_down_sync(0xffffffffu,dtx,off);
    dty+=__shfl_down_sync(0xffffffffu,dty,off);
    dtz+=__shfl_down_sync(0xffffffffu,dtz,off);
    gp +=__shfl_down_sync(0xffffffffu,gp ,off);
  }
  if(lane!=0) return;
  const double qi=q[i];
  force[3*i+0]=-qscale*qi*dtx; force[3*i+1]=-qscale*qi*dty; force[3*i+2]=-qscale*qi*dtz;
  vpot[i]=qscale*gp;
}

// ── QuadS ik path: the same three Fourier gradients on the FULL complex grid ──
// QuadS spreads on a full complex mesh (no R2C half spectrum), so the ik
// multiply walks all nx*ny*nz modes.  The self-conjugate points (k == -k) are
// the only ones where i*k_a*phi_hat cannot stay Hermitian; they are dropped,
// exactly as k_mul_ik does on the half spectrum.
__global__ void k_mul_ik_full(int nx,int ny,int nz,double kfx,double kfy,
                              double kfz,const cufftComplex*spec,
                              cufftComplex*out,int axis){
  const size_t total=(size_t)nx*ny*nz;
  const size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(idx>=total) return;
  const int ix=(int)(idx%(size_t)nx);
  const int iy=(int)((idx/(size_t)nx)%(size_t)ny);
  const int iz=(int)(idx/((size_t)nx*(size_t)ny));
  const int mx=ix-(2*ix>=nx?nx:0);
  const int my=iy-(2*iy>=ny?ny:0);
  const int mz=iz-(2*iz>=nz?nz:0);
  const bool sx=(ix==0)||((nx%2)==0&&ix==nx/2);
  const bool sy=(iy==0)||((ny%2)==0&&iy==ny/2);
  const bool sz=(iz==0)||((nz%2)==0&&iz==nz/2);
  double ka;
  if(axis==0) ka=kfx*mx;
  else if(axis==1) ka=kfy*my;
  else ka=kfz*mz;
  if((sx&&sy&&sz) || ka==0.0){ out[idx].x=0.f; out[idx].y=0.f; return; }
  const double rx=(double)spec[idx].x, ry=(double)spec[idx].y;
  out[idx].x=(float)(-ka*ry);
  out[idx].y=(float)( ka*rx);
}

// Value-only QuadS gather for the ik path: the gradient meshes already carry
// dphi/dr in physical units, so there is no nx/lx rescale and no weight
// derivative.  The k-space virial comes from k_kspace_f, so no r⊗F loop.
template<int ORD>
__global__ void k_gather_quads_ik(int nlocal,const double*x,const double*boxlo,
    double lx,double ly,double lz,const double*q,double qscale,
    int nx,int ny,int nz,const cufftComplex*pot,const cufftComplex*gx,
    const cufftComplex*gy,const cufftComplex*gz,int want_pot,
    double*force,double*vpot){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=nlocal) return;
  double fx=periodic_frac(x[3*i+0],boxlo[0],lx)*nx;
  double fy=periodic_frac(x[3*i+1],boxlo[1],ly)*ny;
  double fz=periodic_frac(x[3*i+2],boxlo[2],lz)*nz;
  int ix0=(int)floor(fx),iy0=(int)floor(fy),iz0=(int)floor(fz);
  double tx=fx-ix0,ty=fy-iy0,tz=fz-iz0;
  double wx[ORD],wy[ORD],wz[ORD];
  if(ORD==4){ quads4_w(tx,wx); quads4_w(ty,wy); quads4_w(tz,wz); }
  else      { quads6_w(tx,wx); quads6_w(ty,wy); quads6_w(tz,wz); }
  const int off=(ORD==4)?-1:-2;
  double dx=0,dy=0,dz=0,gp=0;
  for(int a=0;a<ORD;++a){int gxi=wrap(ix0+off+a,nx);
    for(int b=0;b<ORD;++b){int gyi=wrap(iy0+off+b,ny);
      for(int c=0;c<ORD;++c){int gzi=wrap(iz0+off+c,nz);
        size_t id=midx(gxi,gyi,gzi,nx,ny);
        double w=wx[a]*wy[b]*wz[c];
        dx+=w*gx[id].x; dy+=w*gy[id].x; dz+=w*gz[id].x;
        gp+=w*pot[id].x; }}}
  double qi=q[i];
  force[3*i+0]=-qscale*qi*dx; force[3*i+1]=-qscale*qi*dy;
  force[3*i+2]=-qscale*qi*dz;
  vpot[i]=qscale*gp;
}
// One pass over the half spectrum: multiply by the influence function, write
// the scaled potential spectrum, and reduce the energy and the k-space virial
// with the Hermitian multiplicity of every mode.
//
// This pass and the spread are the two fp64-bound kernels of the pipeline: on
// a card whose fp64 pipe is 1/64 of fp32, three correctly rounded divisions per
// mode (``kx = 2*pi*mx/lx`` with ``lx`` a runtime value -- nvcc cannot
// reassociate them) plus seven fp64 accumulations cost more than all of the
// memory traffic together.  The reciprocal lengths come from the host, one
// thread owns three adjacent kh columns of one y row and strides z, so kx, ky
// and the six k products leave the inner loop, the 64-bit index multiply
// becomes pointer increments, and the seven sums accumulate in fp32 per mode
// and widen once per thread.  The potential spectrum -- the only output the
// inverse FFT sees -- stays bit-identical; the reduced scalars move by ~5e-8.
__global__ void k_kspace_h(int nx,int ny,int nz,double kfx,double kfy,double kfz,
                           const double*hinv,int metric,
                           const float2*gp,cufftComplex*spec,int want_pot,
                           double scaleinv,double*red){
  constexpr int MS = 3;
  const int nh = nx/2+1;
  // Hermitian weight of each stored column of the half spectrum.  kh == 0 is
  // self-conjugate.  kh == nh-1 is ONLY self-conjugate when nx is even (then it
  // is the Nyquist column); for odd nx its partner nx-(nh-1) = (nx+1)/2 is a
  // different, UNSTORED column, so it needs weight 2 like any interior column.
  // Same rule as k_green_pencil.  Getting this wrong costs one column of nx in
  // the energy/virial reduction (red[1..6]); it is exactly 0 for even nx.
  const int even_nx = (2 * nh - 1 != nx) ? 1 : 0;
  const int npw = (nh + MS - 1) / MS;          // work items per y row
  const int myw = ny;
  // All threads reach the warp shuffles and __syncthreads() below; a thread
  // without a work item contributes exact zeros.
  const int w = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  const bool active = (w < npw * myw);
  const int wc = active ? w : 0;
  const int iy = wc / npw;
  const int khc0 = MS * (wc - iy * npw);
  const int my = iy - (2 * iy >= ny ? ny : 0);
  const int nm = (nh - khc0) < MS ? (nh - khc0) : MS;

  // Per-thread constants: the three columns and their k products.
  float kx2[MS], ky2[MS], kxky[MS], kxzf[MS], kyzf[MS], ms2[MS], kxf[MS];
  const float kfyf = (float)kfy, kfzf = (float)kfz;
  const float kyf = kfyf * my;
  #pragma unroll
  for (int j = 0; j < MS; ++j) {
    const int khc = khc0 + j;
    const bool ok = active && (j < nm);
    const float kx = ok ? (float)kfx * khc : 0.f;
    kxf[j] = kx;
    kx2[j] = kx * kx;
    ky2[j] = kyf * kyf;
    kxky[j] = kx * kyf;
    kxzf[j] = kx * kfzf;
    kyzf[j] = kyf * kfzf;
    const double mult =
        (khc == 0 || (even_nx && khc == nh - 1)) ? 1.0 : 2.0;
    ms2[j] = ok ? (float)(mult * scaleinv * scaleinv) : 0.f;
  }

  float acc[MS][7];
  #pragma unroll
  for (int j = 0; j < MS; ++j) {
    #pragma unroll
    for (int t = 0; t < 7; ++t) acc[j][t] = 0.f;
  }

  const size_t gstride = (size_t)gridDim.y * nx * ny;
  const size_t sstride = (size_t)gridDim.y * nh * ny;
  const float2* gp0 = gp + (size_t)nx * (iy + (size_t)ny * blockIdx.y) + khc0;
  cufftComplex* sp = spec + (size_t)nh * (iy + (size_t)ny * blockIdx.y) + khc0;

  for (int iz = (int)blockIdx.y; iz < nz; iz += (int)gridDim.y) {
    const int mz = iz - (2 * iz >= nz ? nz : 0);
    const float kzf = kfzf * mz;
    const float kz2 = kzf * kzf;
    if (active) {
      if (metric) {
        const double mx0 = khc0, myd = my, mzd = mz;
        #pragma unroll
        for (int j = 0; j < MS; ++j) {
          if (j >= nm) break;
          const double mx = mx0 + j;
          const double kx = 2 * M_PI * (hinv[0] * mx + hinv[1] * myd + hinv[2] * mzd);
          const double ky = 2 * M_PI * (hinv[3] * mx + hinv[4] * myd + hinv[5] * mzd);
          const double kz = 2 * M_PI * (hinv[6] * mx + hinv[7] * myd + hinv[8] * mzd);
          const float2 g = gp0[j];
          const double rr = (double)sp[j].x, ri = (double)sp[j].y;
          if (want_pot) {
            sp[j].x = (float)(scaleinv * (double)g.x * rr);
            sp[j].y = (float)(scaleinv * (double)g.x * ri);  // gee, not gvv
          }
          const float p = ms2[j] * ((float)rr * (float)rr + (float)ri * (float)ri);
          const float pg = p * g.x, pv = p * g.y;
          acc[j][0] += pg;
          acc[j][1] = fmaf(-pv, (float)(kx * kx), acc[j][1] + pg);
          acc[j][2] = fmaf(-pv, (float)(ky * ky), acc[j][2] + pg);
          acc[j][3] = fmaf(-pv, (float)(kz * kz), acc[j][3] + pg);
          acc[j][4] = fmaf(-pv, (float)(kx * ky), acc[j][4]);
          acc[j][5] = fmaf(-pv, (float)(kx * kz), acc[j][5]);
          acc[j][6] = fmaf(-pv, (float)(ky * kz), acc[j][6]);
        }
      } else {
        #pragma unroll
        for (int j = 0; j < MS; ++j) {
          if (j >= nm) break;
          const float2 g = gp0[j];
          const double rr = (double)sp[j].x, ri = (double)sp[j].y;
          if (want_pot) {
            sp[j].x = (float)(scaleinv * (double)g.x * rr);
            sp[j].y = (float)(scaleinv * (double)g.x * ri);  // gee, not gvv
          }
          const float p = ms2[j] * ((float)rr * (float)rr + (float)ri * (float)ri);
          const float pg = p * g.x, pv = p * g.y;
          // The diagonal carries gee; the off-diagonal does not.
          acc[j][0] += pg;
          acc[j][1] = fmaf(-pv, kx2[j], acc[j][1] + pg);
          acc[j][2] = fmaf(-pv, ky2[j], acc[j][2] + pg);
          acc[j][3] = fmaf(-pv, kz2, acc[j][3] + pg);
          acc[j][4] = fmaf(-pv, kxky[j], acc[j][4]);
          // kxzf already carries the kfz factor: multiply by the mode only.
          acc[j][5] = fmaf(-pv, kxzf[j] * mz, acc[j][5]);
          acc[j][6] = fmaf(-pv, kyzf[j] * mz, acc[j][6]);
        }
      }
    }
    gp0 += gstride;
    sp += sstride;
  }

  double a[7];
  #pragma unroll
  for (int t = 0; t < 7; ++t) {
    a[t] = (double)acc[0][t] + (double)acc[1][t] + (double)acc[2][t];
  }
  #pragma unroll
  for(int t=0;t<7;++t){
    #pragma unroll
    for(int o=16;o>0;o>>=1) a[t]+=__shfl_down_sync(0xffffffffu,a[t],o);
  }
  __shared__ double sred[7][8];
  const int lane=(int)(threadIdx.x&31u), warp=(int)(threadIdx.x>>5);
  if(lane==0){ for(int t=0;t<7;++t) sred[t][warp]=a[t]; }
  __syncthreads();
  const int nwarps=(int)(blockDim.x>>5);
  if((int)threadIdx.x<7){
    double v=sred[threadIdx.x][0];
    for(int w2=1;w2<nwarps;++w2) v+=sred[threadIdx.x][w2];
    atomicAdd(&red[threadIdx.x], v);
  }
}

// Warp-cooperative CubeS2 spread: one warp per atom, every lane spreads its
// share of the stencil.  Same reasoning as the gather below: one thread per
// atom leaves the device with a handful of blocks.
template<int ORD, typename Cplx>
__global__ void k_spread_cubes2_warp(int nlocal,const double*x,const double*boxlo,
    double lx,double ly,double lz,const double*q,double rho_scale,
    int nx,int ny,int nz,Cplx*rho){
  const int NN=(ORD==4)?32:88;
  // The node table is __constant__, whose broadcast only works when every lane
  // reads the same index; the lanes here take different nodes, which serializes
  // the constant read up to 32 ways.  Stage it in shared memory once per block.
  __shared__ GpuNode sh_nodes[88];
  for(int t=threadIdx.x;t<NN;t+=blockDim.x) sh_nodes[t]=(ORD==4)?c_nodes4s[t]:c_nodes6s[t];
  __syncthreads();
  const int lane=threadIdx.x&31;
  const int i=(blockIdx.x*(blockDim.x>>5))+(threadIdx.x>>5);
  if(i>=nlocal) return;
  const double fx=periodic_frac(x[3*i+0],boxlo[0],lx)*nx;
  const double fy=periodic_frac(x[3*i+1],boxlo[1],ly)*ny;
  const double fz=periodic_frac(x[3*i+2],boxlo[2],lz)*nz;
  const int ix0=(int)floor(fx),iy0=(int)floor(fy),iz0=(int)floor(fz);
  const double tx=fx-ix0,ty=fy-iy0,tz=fz-iz0;
  const float xi_f=(ORD==4)?(float)kXi4:(float)kXi6;
  Cs2Tab tab;
  if (ORD==6) cs2_fill_tab(tab,(float)tx,(float)ty,(float)tz,xi_f);
  const double qs=rho_scale*q[i];
  for(int k=lane;k<NN;k+=32){
    const GpuNode& nd=sh_nodes[k];
    const double w=(ORD==4)?cs2_w4(tx,ty,tz,nd,(double)xi_f):(double)cs2_w6_tab(tab,nd);
    if(w==0.0) continue;
    const int gx=wrap_near(ix0+nd.dx,nx),gy=wrap_near(iy0+nd.dy,ny),gz=wrap_near(iz0+nd.dz,nz);
    atomicAdd(&rho[midx(gx,gy,gz,nx,ny)].x, qs*w);
  }
}

// Warp-cooperative CubeS2 gather: one warp per atom, every lane takes a share
// of the stencil and the four sums reduce across the warp.  A thread-per-atom
// launch gives only nlocal threads -- a few blocks for a few thousand atoms --
// and each thread then walks the 32/88 nodes through dependent uncached loads,
// so the kernel runs far below the memory system's capability.
template<int ORD, typename Cplx>
__global__ void k_gather_cubes2_warp(int nlocal,const double*x,const double*boxlo,
    double lx,double ly,double lz,const double*q,double qscale,
    int nx,int ny,int nz,const Cplx*pot,int want_pot,
    double*force,double*vpot,double*red){
  const int NN=(ORD==4)?32:88;
  __shared__ GpuNode sh_nodes[88];
  for(int t=threadIdx.x;t<NN;t+=blockDim.x) sh_nodes[t]=(ORD==4)?c_nodes4s[t]:c_nodes6s[t];
  __syncthreads();
  const int lane=threadIdx.x&31;
  const int i=(blockIdx.x*(blockDim.x>>5))+(threadIdx.x>>5);
  if(i>=nlocal) return;
  const double fx=periodic_frac(x[3*i+0],boxlo[0],lx)*nx;
  const double fy=periodic_frac(x[3*i+1],boxlo[1],ly)*ny;
  const double fz=periodic_frac(x[3*i+2],boxlo[2],lz)*nz;
  const int ix0=(int)floor(fx),iy0=(int)floor(fy),iz0=(int)floor(fz);
  const double tx=fx-ix0,ty=fy-iy0,tz=fz-iz0;
  const double xi=(ORD==4)?kXi4:kXi6;
  double dtx=0,dty=0,dtz=0,gp=0;
  for(int k=lane;k<NN;k+=32){
    const GpuNode& nd=sh_nodes[k];
    GpuWeightGrad wg=(ORD==4)?cs2_wg4(tx,ty,tz,nd,xi):cs2_wg6(tx,ty,tz,nd,xi);
    const double w=wg.v; if(w==0.0) continue;
    const int gxi=wrap_near(ix0+nd.dx,nx),gyi=wrap_near(iy0+nd.dy,ny),gzi=wrap_near(iz0+nd.dz,nz);
    const double u=(double)pot[midx(gxi,gyi,gzi,nx,ny)].x;
    dtx+=wg.dx*u; dty+=wg.dy*u; dtz+=wg.dz*u; gp+=w*u;
  }
  #pragma unroll
  for(int off=16;off>0;off>>=1){
    dtx+=__shfl_down_sync(0xffffffffu,dtx,off);
    dty+=__shfl_down_sync(0xffffffffu,dty,off);
    dtz+=__shfl_down_sync(0xffffffffu,dtz,off);
    gp +=__shfl_down_sync(0xffffffffu,gp ,off);
  }
  if(lane!=0) return;
  const double qi=q[i];
  const double fxs=-qscale*qi*dtx*nx/lx, fys=-qscale*qi*dty*ny/ly, fzs=-qscale*qi*dtz*nz/lz;
  force[3*i+0]=fxs; force[3*i+1]=fys; force[3*i+2]=fzs;
  vpot[i]=qscale*gp;
}

// ── kernels ──
// QuadS spread: 1 thread/atom, (norder)^3 tensor-product atomicAdds into the real part of d_rho.
template<int NORD, typename Cplx>
__global__ void k_spread_quads(int nlocal,const double*x,const double*boxlo,
    double lx,double ly,double lz,const double*q,double rho_scale,
    int nx,int ny,int nz,Cplx*rho){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=nlocal) return;
  double fx=periodic_frac(x[3*i+0],boxlo[0],lx)*nx;
  double fy=periodic_frac(x[3*i+1],boxlo[1],ly)*ny;
  double fz=periodic_frac(x[3*i+2],boxlo[2],lz)*nz;
  int ix0=(int)floor(fx), iy0=(int)floor(fy), iz0=(int)floor(fz);
  double tx=fx-ix0, ty=fy-iy0, tz=fz-iz0;
  double wx[NORD],wy[NORD],wz[NORD],dwx[NORD],dwy[NORD],dwz[NORD];
  if(NORD==4){ quads4_wd(tx,wx,dwx); quads4_wd(ty,wy,dwy); quads4_wd(tz,wz,dwz); }
  else       { quads6_wd(tx,wx,dwx); quads6_wd(ty,wy,dwy); quads6_wd(tz,wz,dwz); }
  const int off=(NORD==4)?-1:-2;
  double qs=rho_scale*q[i];
  for(int a=0;a<NORD;++a){int gx=wrap(ix0+off+a,nx); double wa=wx[a]*qs;
    for(int b=0;b<NORD;++b){int gy=wrap(iy0+off+b,ny); double wab=wa*wy[b];
      for(int c=0;c<NORD;++c){int gz=wrap(iz0+off+c,nz);
        atomicAdd(&rho[midx(gx,gy,gz,nx,ny)].x, wab*wz[c]); }}}
}

// ── Warp-cooperative QuadS kernels on the real / half-spectrum path ──
// The full-complex, one-thread-per-atom kernels above are the reference; these
// mirror the B-spline / CubeS2 core: a real density mesh, R2C/C2R transforms,
// and one warp per atom that strides the (2ν)^3 separable stencil across its 32
// lanes.  The stencil node k decodes as (a,b,c) = (k/N², (k/N)%N, k%N) in the
// same order as the triple loop it replaces.
template<int NORD>
__global__ void k_spread_quads_warp_r(int nlocal,const double*x,
    const double*boxlo,double lx,double ly,double lz,const double*q,
    double rho_scale,int nx,int ny,int nz,float*rho){
  // LaneSlots: lanes take (k2,k3) pairs with k3 (x) fastest and each lane walks
  // the k1 (z) axis, so a warp touches contiguous x runs (the B-spline /
  // CubeS2 scheduler).  k3 -> x, k2 -> y, k1 -> z.
  constexpr int P=NORD, kPairs=P*P;
  constexpr int kGroups=(kPairs>=32)?1:(32/kPairs);
  constexpr int kSlots=kGroups*kPairs, kRounds=(kSlots+31)/32;
  const int lane=threadIdx.x&31, warp=threadIdx.x>>5;
  const int i=blockIdx.x*(blockDim.x>>5)+warp; if(i>=nlocal) return;
  double fx=periodic_frac(x[3*i+0],boxlo[0],lx)*nx;
  double fy=periodic_frac(x[3*i+1],boxlo[1],ly)*ny;
  double fz=periodic_frac(x[3*i+2],boxlo[2],lz)*nz;
  int ix0=(int)floor(fx),iy0=(int)floor(fy),iz0=(int)floor(fz);
  double tx=fx-ix0,ty=fy-iy0,tz=fz-iz0;
  double wx[P],wy[P],wz[P];
  bcast_quads_w<NORD>(tx,wx,lane);
  bcast_quads_w<NORD>(ty,wy,lane);
  bcast_quads_w<NORD>(tz,wz,lane);
  const int off=(NORD==4)?-1:-2;
  const double qs=rho_scale*q[i];
  #pragma unroll
  for(int round=0;round<kRounds;++round){
    const int slot=lane+32*round;
    const bool live=(slot<kSlots);
    const int ss=live?slot:(kSlots-1);
    const int g=ss/kPairs, pair=ss%kPairs;
    const int k2=pair/P, k3=pair%P;
    const int k1b=g*(P/kGroups), k1e=k1b+(P/kGroups);
    for(int k1=k1b;k1<k1e;++k1){
      if(!live) break;
      const int gx=wrap(ix0+off+k3,nx),gy=wrap(iy0+off+k2,ny),
                gz=wrap(iz0+off+k1,nz);
      atomicAdd(&rho[midx(gx,gy,gz,nx,ny)],
                (float)(wx[k3]*wy[k2]*wz[k1]*qs));
    }
  }
}

// AD gather: value + weight-derivative accumulation, warp-reduced.  The
// derivative keeps the nx/lx rescale because it differentiates the mesh
// weights, not the potential.
template<int NORD>
__global__ void k_gather_quads_warp_r(int nlocal,const double*x,
    const double*boxlo,double lx,double ly,double lz,const double*q,
    double qscale,int nx,int ny,int nz,const float*pot,int want_pot,
    double*force,double*vpot,double*red){
  constexpr int P=NORD, kPairs=P*P;
  constexpr int kGroups=(kPairs>=32)?1:(32/kPairs);
  constexpr int kSlots=kGroups*kPairs, kRounds=(kSlots+31)/32;
  const int lane=threadIdx.x&31, warp=threadIdx.x>>5;
  const int i=blockIdx.x*(blockDim.x>>5)+warp; if(i>=nlocal) return;
  double fx=periodic_frac(x[3*i+0],boxlo[0],lx)*nx;
  double fy=periodic_frac(x[3*i+1],boxlo[1],ly)*ny;
  double fz=periodic_frac(x[3*i+2],boxlo[2],lz)*nz;
  int ix0=(int)floor(fx),iy0=(int)floor(fy),iz0=(int)floor(fz);
  double tx=fx-ix0,ty=fy-iy0,tz=fz-iz0;
  double wx[P],wy[P],wz[P],dwx[P],dwy[P],dwz[P];
  bcast_quads_wd<NORD>(tx,wx,dwx,lane);
  bcast_quads_wd<NORD>(ty,wy,dwy,lane);
  bcast_quads_wd<NORD>(tz,wz,dwz,lane);
  const int off=(NORD==4)?-1:-2;
  double dtx=0,dty=0,dtz=0,gp=0;
  #pragma unroll
  for(int round=0;round<kRounds;++round){
    const int slot=lane+32*round;
    const bool live=(slot<kSlots);
    const int ss=live?slot:(kSlots-1);
    const int g=ss/kPairs, pair=ss%kPairs;
    const int k2=pair/P, k3=pair%P;
    const int k1b=g*(P/kGroups), k1e=k1b+(P/kGroups);
    for(int k1=k1b;k1<k1e;++k1){
      const int gx=wrap(ix0+off+k3,nx),gy=wrap(iy0+off+k2,ny),
                gz=wrap(iz0+off+k1,nz);
      const double u=live?(double)pot[midx(gx,gy,gz,nx,ny)]:0.0;
      const double wyz=wy[k2]*wz[k1];
      dtx+=dwx[k3]*wyz*u;
      dty+=wx[k3]*dwy[k2]*wz[k1]*u;
      dtz+=wx[k3]*wy[k2]*dwz[k1]*u;
      gp +=wx[k3]*wyz*u;
    }
  }
  #pragma unroll
  for(int off2=16;off2>0;off2>>=1){
    dtx+=__shfl_down_sync(0xffffffffu,dtx,off2);
    dty+=__shfl_down_sync(0xffffffffu,dty,off2);
    dtz+=__shfl_down_sync(0xffffffffu,dtz,off2);
    gp +=__shfl_down_sync(0xffffffffu,gp ,off2);
  }
  if(lane!=0) return;
  const double qi=q[i];
  const double fxs=-qscale*qi*dtx*nx/lx;
  const double fys=-qscale*qi*dty*ny/ly;
  const double fzs=-qscale*qi*dtz*nz/lz;
  force[3*i+0]=fxs; force[3*i+1]=fys; force[3*i+2]=fzs;
  vpot[i]=qscale*gp;
  atomicAdd(&red[14], x[3*i+0]*fxs);
  atomicAdd(&red[15], x[3*i+1]*fys);
  atomicAdd(&red[16], x[3*i+2]*fzs);
  atomicAdd(&red[17], x[3*i+0]*fys);
  atomicAdd(&red[18], x[3*i+0]*fzs);
  atomicAdd(&red[19], x[3*i+1]*fzs);
}

// ik gather: the three gradient meshes already hold dphi/dr, so the gather is
// value-only with no rescale (matching k_gather_cubes2_warp_r_ik).
template<int NORD>
__global__ void k_gather_quads_warp_r_ik(int nlocal,const double*x,
    const double*boxlo,double lx,double ly,double lz,const double*q,
    double qscale,int nx,int ny,int nz,const float*pot,const float*gx,
    const float*gy,const float*gz,int want_pot,double*force,double*vpot){
  constexpr int P=NORD, kPairs=P*P;
  constexpr int kGroups=(kPairs>=32)?1:(32/kPairs);
  constexpr int kSlots=kGroups*kPairs, kRounds=(kSlots+31)/32;
  const int lane=threadIdx.x&31, warp=threadIdx.x>>5;
  const int i=blockIdx.x*(blockDim.x>>5)+warp; if(i>=nlocal) return;
  double fx=periodic_frac(x[3*i+0],boxlo[0],lx)*nx;
  double fy=periodic_frac(x[3*i+1],boxlo[1],ly)*ny;
  double fz=periodic_frac(x[3*i+2],boxlo[2],lz)*nz;
  int ix0=(int)floor(fx),iy0=(int)floor(fy),iz0=(int)floor(fz);
  double tx=fx-ix0,ty=fy-iy0,tz=fz-iz0;
  double wx[P],wy[P],wz[P];
  bcast_quads_w<NORD>(tx,wx,lane);
  bcast_quads_w<NORD>(ty,wy,lane);
  bcast_quads_w<NORD>(tz,wz,lane);
  const int off=(NORD==4)?-1:-2;
  double dx=0,dy=0,dz=0,gp=0;
  #pragma unroll
  for(int round=0;round<kRounds;++round){
    const int slot=lane+32*round;
    const bool live=(slot<kSlots);
    const int ss=live?slot:(kSlots-1);
    const int g=ss/kPairs, pair=ss%kPairs;
    const int k2=pair/P, k3=pair%P;
    const int k1b=g*(P/kGroups), k1e=k1b+(P/kGroups);
    for(int k1=k1b;k1<k1e;++k1){
      const int m=midx(wrap(ix0+off+k3,nx),wrap(iy0+off+k2,ny),
                       wrap(iz0+off+k1,nz),nx,ny);
      const double w=live?(wx[k3]*wy[k2]*wz[k1]):0.0;
      dx+=w*gx[m]; dy+=w*gy[m]; dz+=w*gz[m]; gp+=w*pot[m];
    }
  }
  #pragma unroll
  for(int off2=16;off2>0;off2>>=1){
    dx+=__shfl_down_sync(0xffffffffu,dx,off2);
    dy+=__shfl_down_sync(0xffffffffu,dy,off2);
    dz+=__shfl_down_sync(0xffffffffu,dz,off2);
    gp+=__shfl_down_sync(0xffffffffu,gp,off2);
  }
  if(lane!=0) return;
  const double qi=q[i];
  force[3*i+0]=-qscale*qi*dx; force[3*i+1]=-qscale*qi*dy;
  force[3*i+2]=-qscale*qi*dz;
  vpot[i]=qscale*gp;
}

// green-multiply: 1 thread/grid-point. Builds ik*gf*rho grad meshes + ge*rho potential mesh,
// and reduces energy + 6 virial (block-reduced into d_red via atomicAdd).
//
// The k-space SELF term used to be accumulated here as a Riemann sum over mesh
// modes (`diag_sum`, legacy red[7]) plus six self-stress accumulators
// (legacy red[8..13]).  That sum is NOT volume-extensive: K(k^2) is dominated
// by the few lowest-|k| modes, whose sampling changes with the box, so for a
// 264-atom cell it grew 9.03x (2x replication) and 76.5x (4x) instead of 8x and
// 64x.  It also made sum(K_v k_a k_b) anisotropic -- (0.827, 1.085, 1.339) x
// the diagonal instead of isotropic -- injecting a spurious shear stress.
// Both are replaced by the exact continuum value, which is closed-form and
// needs no mesh at all; see the `self_coeff` handling in the solve functions.
__global__ void k_kspace(int nx,int ny,int nz,double lx,double ly,double lz,
    const double*hinv,int metric,
    const cufftDoubleComplex*rho,const double*ge,const double*gf,const double*gv,
    cufftDoubleComplex*gx,cufftDoubleComplex*gy,cufftDoubleComplex*gz,
    cufftDoubleComplex*pot,int want_pot,double scaleinv,double*red){
  size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x; size_t ngrid=(size_t)nx*ny*nz;
  // Every thread of the block MUST reach the warp shuffles and __syncthreads()
  // below.  An early `return` here made the last (partially filled) block
  // undefined behaviour on three counts: __shfl_down_sync(0xffffffff) asserts
  // that all 32 lanes are active, warps that exited never wrote sred[t][warp]
  // (leaving uninitialised shared memory in the reduction), and
  // __syncthreads() was not reached by every thread.  The visible symptom was
  // a random, system-size-INDEPENDENT additive error in red[0] (energy) and
  // red[1..6] (virial), plus garbage pot/gx/gy/gz for the grid points of that
  // block: 11.4 eV (43 meV/atom) spread over identical calls at 264 atoms,
  // 0.6 eV (35 ueV/atom) at 16896, and a 1e7 atm pressure at 264 atoms.
  // Out-of-range threads now stay in the kernel and contribute exact zeros.
  const bool in_range = (idx < ngrid);
  if(!in_range) idx = 0;
  int ix=idx%nx, iy=(idx/nx)%ny, iz=idx/((size_t)nx*ny);
  int kxm=ix-nx*((2*ix)/nx), kym=iy-ny*((2*iy)/ny), kzm=iz-nz*((2*iz)/nz);
  double kx, ky, kz;
  if(metric){
    // k_a = 2 pi sum_d (A^{-1})_{d a} m_d; hinv is row d, column a.
    const double mx=kxm, my=kym, mz=kzm;
    kx=2*M_PI*(hinv[0]*mx+hinv[1]*my+hinv[2]*mz);
    ky=2*M_PI*(hinv[3]*mx+hinv[4]*my+hinv[5]*mz);
    kz=2*M_PI*(hinv[6]*mx+hinv[7]*my+hinv[8]*mz);
  } else {
    kx=2*M_PI/lx*kxm; ky=2*M_PI/ly*kym; kz=2*M_PI/lz*kzm;
  }
  double geff=gf[idx], gee=ge[idx], gvv=gv[idx];
  double rr=rho[idx].x, ri=rho[idx].y, rho2=rr*rr+ri*ri;
  double s2=scaleinv*scaleinv;
  // ── Reductions ──
  // 7 global accumulators per grid point used to be one atomicAdd each onto the
  // same 20 doubles: at a 125³ mesh that is 27 M contending global atomics per
  // step and it dominated the whole k-space solve (~20 ms).  Reduce inside the
  // warp (shuffles), then per block in shared memory, then one atomicAdd per
  // block -- 8 warps x 7629 blocks instead of 1.95 M x 7.
  double acc[7];
  acc[0]  = s2*gee*rho2;                 // energy
  acc[1]  = s2*rho2*(gee - gvv*kx*kx);   // xx
  acc[2]  = s2*rho2*(gee - gvv*ky*ky);   // yy
  acc[3]  = s2*rho2*(gee - gvv*kz*kz);   // zz
  acc[4]  = s2*rho2*(-gvv*kx*ky);        // xy
  acc[5]  = s2*rho2*(-gvv*kx*kz);        // xz
  acc[6]  = s2*rho2*(-gvv*ky*kz);        // yz
  // Out-of-range lanes took the idx=0 loads above; zero them so they add
  // nothing to the block reduction while still participating in the shuffles.
  if(!in_range){
    for(int t=0;t<7;++t) acc[t]=0.0;
  }
  #pragma unroll
  for(int t=0;t<7;++t){
    #pragma unroll
    for(int o=16;o>0;o>>=1) acc[t]+=__shfl_down_sync(0xffffffffu,acc[t],o);
  }
  __shared__ double sred[7][8];
  const int lane=(int)(threadIdx.x&31u), warp=(int)(threadIdx.x>>5);
  if(lane==0){
    #pragma unroll
    for(int t=0;t<7;++t) sred[t][warp]=acc[t];
  }
  __syncthreads();
  const int nwarps=(int)(blockDim.x>>5);
  if((int)threadIdx.x<7){
    double v=sred[threadIdx.x][0];
    for(int w=1;w<nwarps;++w) v+=sred[threadIdx.x][w];
    atomicAdd(&red[threadIdx.x], v);
  }
  // gradient meshes: grad = i*k * (scaleinv*geff*rho)
  // Only real grid points may be written; out-of-range lanes operate on the
  // idx=0 values and must not store them.
  if(in_range){
    // Only the potential mesh: the gathers differentiate their interpolation
    // weights instead of reading a gradient mesh (see the gather kernels).
    if(want_pot){ pot[idx].x=scaleinv*gee*rr; pot[idx].y=scaleinv*gee*ri; }
  }
  (void)gx; (void)gy; (void)gz;
}

__global__ void k_kspace_f(int nx,int ny,int nz,double lx,double ly,double lz,
    const double*hinv,int metric,
    const cufftComplex*rho,const float2*gp,
        cufftComplex*pot,int want_pot,double scaleinv,double*red){
  size_t idx=(size_t)blockIdx.x*blockDim.x+threadIdx.x; size_t ngrid=(size_t)nx*ny*nz;
  // Every thread of the block MUST reach the warp shuffles and __syncthreads()
  // below.  An early `return` here made the last (partially filled) block
  // undefined behaviour on three counts: __shfl_down_sync(0xffffffff) asserts
  // that all 32 lanes are active, warps that exited never wrote sred[t][warp]
  // (leaving uninitialised shared memory in the reduction), and
  // __syncthreads() was not reached by every thread.  The visible symptom was
  // a random, system-size-INDEPENDENT additive error in red[0] (energy) and
  // red[1..6] (virial), plus garbage pot/gx/gy/gz for the grid points of that
  // block: 11.4 eV (43 meV/atom) spread over identical calls at 264 atoms,
  // 0.6 eV (35 ueV/atom) at 16896, and a 1e7 atm pressure at 264 atoms.
  // Out-of-range threads now stay in the kernel and contribute exact zeros.
  const bool in_range = (idx < ngrid);
  if(!in_range) idx = 0;
  int ix=idx%nx, iy=(idx/nx)%ny, iz=idx/((size_t)nx*ny);
  int kxm=ix-nx*((2*ix)/nx), kym=iy-ny*((2*iy)/ny), kzm=iz-nz*((2*iz)/nz);
  double kx, ky, kz;
  if(metric){
    // k_a = 2 pi sum_d (A^{-1})_{d a} m_d; hinv is row d, column a.
    const double mx=kxm, my=kym, mz=kzm;
    kx=2*M_PI*(hinv[0]*mx+hinv[1]*my+hinv[2]*mz);
    ky=2*M_PI*(hinv[3]*mx+hinv[4]*my+hinv[5]*mz);
    kz=2*M_PI*(hinv[6]*mx+hinv[7]*my+hinv[8]*mz);
  } else {
    kx=2*M_PI/lx*kxm; ky=2*M_PI/ly*kym; kz=2*M_PI/lz*kzm;
  }
  const float2 gpq=gp[idx];
  double gee=(double)gpq.x, geff=gee, gvv=(double)gpq.y;  // ge == gf
  double rr=rho[idx].x, ri=rho[idx].y, rho2=rr*rr+ri*ri;
  double s2=scaleinv*scaleinv;
  // ── Reductions ──
  // 7 global accumulators per grid point used to be one atomicAdd each onto the
  // same 20 doubles: at a 125³ mesh that is 27 M contending global atomics per
  // step and it dominated the whole k-space solve (~20 ms).  Reduce inside the
  // warp (shuffles), then per block in shared memory, then one atomicAdd per
  // block -- 8 warps x 7629 blocks instead of 1.95 M x 7.
  double acc[7];
  acc[0]  = s2*gee*rho2;                 // energy
  acc[1]  = s2*rho2*(gee - gvv*kx*kx);   // xx
  acc[2]  = s2*rho2*(gee - gvv*ky*ky);   // yy
  acc[3]  = s2*rho2*(gee - gvv*kz*kz);   // zz
  acc[4]  = s2*rho2*(-gvv*kx*ky);        // xy
  acc[5]  = s2*rho2*(-gvv*kx*kz);        // xz
  acc[6]  = s2*rho2*(-gvv*ky*kz);        // yz
  // Out-of-range lanes took the idx=0 loads above; zero them so they add
  // nothing to the block reduction while still participating in the shuffles.
  if(!in_range){
    for(int t=0;t<7;++t) acc[t]=0.0;
  }
  #pragma unroll
  for(int t=0;t<7;++t){
    #pragma unroll
    for(int o=16;o>0;o>>=1) acc[t]+=__shfl_down_sync(0xffffffffu,acc[t],o);
  }
  __shared__ double sred[7][8];
  const int lane=(int)(threadIdx.x&31u), warp=(int)(threadIdx.x>>5);
  if(lane==0){
    #pragma unroll
    for(int t=0;t<7;++t) sred[t][warp]=acc[t];
  }
  __syncthreads();
  const int nwarps=(int)(blockDim.x>>5);
  if((int)threadIdx.x<7){
    double v=sred[threadIdx.x][0];
    for(int w=1;w<nwarps;++w) v+=sred[threadIdx.x][w];
    atomicAdd(&red[threadIdx.x], v);
  }
  // gradient meshes: grad = i*k * (scaleinv*geff*rho)
  // Only real grid points may be written; out-of-range lanes operate on the
  // idx=0 values and must not store them.
  if(in_range){
    // Only the potential mesh: the gathers differentiate their interpolation
    // weights instead of reading a gradient mesh (see the gather kernels).
    if(want_pot){ pot[idx].x=scaleinv*gee*rr; pot[idx].y=scaleinv*gee*ri; }
  }
}


// ── Distributed (slab) k-space: green multiply on a local PENCIL ─────────────
// The pencil holds the full z range, all x, and this rank's y-slab, with z
// FASTEST: index = iz + nz*(ix + nx*iy_local).  That layout makes both the
// z-transform a contiguous batch and the global-index arithmetic below cheap.
//
// Every quantity must be GLOBAL (theory doc I1/I2/I3): the green table is
// indexed by the global grid index gi = ix + nx*(iy + ny*iz) with iy = ylo +
// iy_local, and k comes from the global integer mode.  Using a local offset
// instead produces a plausible-looking but wrong energy with no fingerprint
// (measured 1.78x / 13.27x), which is why this is spelled out here.
//
// Reduces 7 accumulators (energy + 6 virial) exactly like k_kspace, so the
// caller's global Allreduce is the same one the replicated path already does.
// R2C pencil: only kh = 0..Nh-1 is stored.  Every internal kh stands for
// itself AND its -kx partner (Hermitian weight 2); kh=0 and, for even nx, the
// Nyquist column kh=Nh-1 have no partner (weight 1).  The green tables remain
// FULL-size in x, so the table index uses the original `nx` stride at column
// kh (the table only depends on kx^2).
// Reads the SAME single-precision representations as the replicated solve: a
// cufftComplex pencil and the packed (green energy, green virial) float2 table.
__global__ void k_green_pencil(int Nh,int nx,int ny,int nz,int ylo,int YL,
    double lx,double ly,double lz,
    const cufftComplex*rho,const float2*gp,
    cufftComplex*pot,double scaleinv,double*red){
  size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  const size_t nloc=(size_t)nz*(size_t)Nh*(size_t)YL;
  const bool in_range = (i < nloc);
  if(!in_range) i = 0;
  int iz=(int)(i % nz);
  int kh=(int)((i/nz) % Nh);
  int iyl=(int)(i/((size_t)nz*Nh));
  int iy=ylo+iyl;
  if(!in_range){ iz=0; kh=0; iyl=0; iy=ylo; }
  // The green tables are this rank's COMPACT x half-spectrum y-slice:
  // kh + Nh*(iyl + YL*iz) (the -kx half is redundant for a real kernel).
  const size_t gi=(size_t)kh+(size_t)Nh*((size_t)iyl+(size_t)YL*(size_t)iz);
  const int even_nx = (2*Nh-1 != nx) ? 1 : 0;
  const double w = (kh==0 || (even_nx && kh==Nh-1)) ? 1.0 : 2.0;
  int kym=iy-ny*((2*iy)/ny), kzm=iz-nz*((2*iz)/nz);
  // Physical k = 2 pi / L * mode  (NOT 2 pi / n: the virial terms need it).
  // kx uses the +kh representative; the green table and every acc term depend
  // on kx^2 (or the pair-symmetric kx*ky / kx*kz), so the sign is immaterial.
  double kx=2*M_PI/lx*kh, ky=2*M_PI/ly*kym, kz=2*M_PI/lz*kzm;
  double gee=(double)gp[gi].x, gvv=(double)gp[gi].y;
  double rr=(double)rho[i].x, ri=(double)rho[i].y, rho2=rr*rr+ri*ri;
  double s2=scaleinv*scaleinv;
  double acc[7];
  acc[0] = s2*gee*rho2;
  acc[1] = s2*rho2*(gee - gvv*kx*kx);
  acc[2] = s2*rho2*(gee - gvv*ky*ky);
  acc[3] = s2*rho2*(gee - gvv*kz*kz);
  acc[4] = s2*rho2*(-gvv*kx*ky);
  acc[5] = s2*rho2*(-gvv*kx*kz);
  acc[6] = s2*rho2*(-gvv*ky*kz);
  if(!in_range){ for(int t=0;t<7;++t) acc[t]=0.0; }
  else { for(int t=0;t<7;++t) acc[t]*=w; }   // half-spectrum Hermitian weight
  #pragma unroll
  for(int t=0;t<7;++t){
    #pragma unroll
    for(int o=16;o>0;o>>=1) acc[t]+=__shfl_down_sync(0xffffffffu,acc[t],o);
  }
  __shared__ double sred[7][8];
  const int lane=(int)(threadIdx.x&31u), warp=(int)(threadIdx.x>>5);
  if(lane==0){
    #pragma unroll
    for(int t=0;t<7;++t) sred[t][warp]=acc[t];
  }
  __syncthreads();
  const int nwarps=(int)(blockDim.x>>5);
  if((int)threadIdx.x<7){
    double v=sred[threadIdx.x][0];
    for(int w=1;w<nwarps;++w) v+=sred[threadIdx.x][w];
    atomicAdd(&red[threadIdx.x], v);
  }
  if(in_range){ pot[i].x=scaleinv*gee*rr; pot[i].y=scaleinv*gee*ri; }
}

// ── Transpose pack/unpack on the device ────────────────────────────────────
// The host used to repack the mesh with three nested loops per direction; at a
// 150^3 mesh that is ~0.5 GB of strided host traffic per step.  These kernels
// do the same permutation on the GPU so only the MPI message itself crosses
// the bus.
//
// Block decomposition (must match the host's scnt/rdisp exactly): destination
// d owns y in [d*ny/np, (d+1)*ny/np); sender r owns z in [r*nz/np, (r+1)*nz/np).
// A block is laid out [yl][ix][z_local] with z fastest.
__global__ void k_pack_xy_to_blocks(int Nh,int ny,int nz,int ZL,int np,
    const cufftComplex*cur,cufftComplex*snd){
  size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  const size_t ntot=(size_t)Nh*ny*ZL;
  if(t>=ntot) return;
  const int yb=(int)(t/((size_t)ZL*Nh));          // global y
  int d=0; while(d+1<np && yb >= ((d+1)*ny)/np) ++d;
  const int dl=(d*ny)/np;
  const size_t b=t-(size_t)Nh*ZL*dl;              // index inside the block
  const int z=(int)(b%ZL), kh=(int)((b/ZL)%Nh);
  const size_t src=(size_t)kh+(size_t)Nh*((size_t)yb+(size_t)ny*z);
  snd[t]=cur[src];
}

__global__ void k_unpack_blocks_to_pen(int Nh,int ny,int nz,int np,int YL,
    const cufftComplex*rcv,cufftComplex*pen){
  size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  const size_t ntot=(size_t)nz*Nh*YL;
  if(t>=ntot) return;
  const int zl=(int)(t%nz), kh=(int)((t/nz)%Nh),
            yl=(int)(t/((size_t)nz*Nh));
  int r=0; while(r+1<np && zl >= ((r+1)*nz)/np) ++r;
  const int rl=(r*nz)/np, rz=((r+1)*nz)/np-rl;
  const size_t start=(size_t)Nh*YL*rl;
  const size_t src=start+(size_t)(zl-rl)+
                   (size_t)rz*((size_t)kh+(size_t)Nh*yl);
  pen[t]=rcv[src];
}

__global__ void k_pack_pen_to_blocks(int Nh,int ny,int nz,int np,int YL,
    const cufftComplex*pen,cufftComplex*snd){
  size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  const size_t ntot=(size_t)nz*Nh*YL;
  if(t>=ntot) return;
  const int zl=(int)(t%nz), kh=(int)((t/nz)%Nh),
            yl=(int)(t/((size_t)nz*Nh));
  int r=0; while(r+1<np && zl >= ((r+1)*nz)/np) ++r;
  const int rl=(r*nz)/np, rz=((r+1)*nz)/np-rl;
  const size_t start=(size_t)Nh*YL*rl;
  const size_t dst=start+(size_t)(zl-rl)+
                   (size_t)rz*((size_t)kh+(size_t)Nh*yl);
  snd[dst]=pen[t];
}

__global__ void k_unpack_blocks_to_xy(int Nh,int ny,int ZL,int np,
    const cufftComplex*back,cufftComplex*cur2){
  size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  const size_t ntot=(size_t)Nh*ny*ZL;
  if(t>=ntot) return;
  // The compact slab is (kh fastest, y, z_local): index = kh + Nh*(y + ny*z).
  // Decoding it as the block layout instead permutes y and z (energy and forces
  // wrong, virial right -- the same trap as the full-spectrum version).
  const int z=(int)(t/((size_t)Nh*ny));
  const int y=(int)((t/(size_t)Nh)%ny);
  const int kh=(int)(t%Nh);
  int d=0; while(d+1<np && y >= ((d+1)*ny)/np) ++d;
  const int dl=(d*ny)/np;
  const size_t src=(size_t)Nh*ZL*dl+(size_t)z+
                   (size_t)ZL*((size_t)kh+(size_t)Nh*(y-dl));
  cur2[t]=back[src];
}

template<int NORD, typename Cplx>
__global__ void k_gather_quads(int nlocal,const double*x,const double*boxlo,
    double lx,double ly,double lz,const double*q,double qscale,
    int nx,int ny,int nz,const Cplx*pot,int want_pot,
    double*force,double*vpot,double*red){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=nlocal) return;
  double fx=periodic_frac(x[3*i+0],boxlo[0],lx)*nx;
  double fy=periodic_frac(x[3*i+1],boxlo[1],ly)*ny;
  double fz=periodic_frac(x[3*i+2],boxlo[2],lz)*nz;
  int ix0=(int)floor(fx),iy0=(int)floor(fy),iz0=(int)floor(fz);
  double tx=fx-ix0,ty=fy-iy0,tz=fz-iz0;
  double wx[NORD],wy[NORD],wz[NORD],dwx[NORD],dwy[NORD],dwz[NORD];
  if(NORD==4){ quads4_wd(tx,wx,dwx); quads4_wd(ty,wy,dwy); quads4_wd(tz,wz,dwz); }
  else       { quads6_wd(tx,wx,dwx); quads6_wd(ty,wy,dwy); quads6_wd(tz,wz,dwz); }
  const int off=(NORD==4)?-1:-2;
  double dtx=0,dty=0,dtz=0,gp=0;
  for(int a=0;a<NORD;++a){int gxi=wrap(ix0+off+a,nx);
    for(int b=0;b<NORD;++b){int gyi=wrap(iy0+off+b,ny); double wab=wx[a]*wy[b];
      for(int c=0;c<NORD;++c){int gzi=wrap(iz0+off+c,nz); double w=wab*wz[c];
        size_t id=midx(gxi,gyi,gzi,nx,ny);
        double u=pot[id].x;
        dtx+=dwx[a]*wy[b]*wz[c]*u; dty+=wx[a]*dwy[b]*wz[c]*u; dtz+=wx[a]*wy[b]*dwz[c]*u;
        gp+=w*u; }}}
  double qi=q[i];
  double fxs=-qscale*qi*dtx*nx/lx, fys=-qscale*qi*dty*ny/ly, fzs=-qscale*qi*dtz*nz/lz;
  force[3*i+0]=fxs; force[3*i+1]=fys; force[3*i+2]=fzs;
  vpot[i]=qscale*gp;
  // r⊗F virial: exact, captures all position-dependent strain effects
  // that the analytic Fourier formula misses (matching CPU vv_rf).
  // red[14..19] holds Σ r_iα · F_iβ.
  atomicAdd(&red[14], x[3*i+0]*fxs);  // r_x * F_x (xx)
  atomicAdd(&red[15], x[3*i+1]*fys);  // r_y * F_y (yy)
  atomicAdd(&red[16], x[3*i+2]*fzs);  // r_z * F_z (zz)
  atomicAdd(&red[17], x[3*i+0]*fys);  // r_x * F_y (xy)
  atomicAdd(&red[18], x[3*i+0]*fzs);  // r_x * F_z (xz)
  atomicAdd(&red[19], x[3*i+1]*fzs);  // r_y * F_z (yz)
}

// ── Distributed Grid3d local-brick kernels ──
// Convert a global periodic grid index to the equivalent unwrapped index
// stored by this rank's owned+ghost brick.
__device__ __forceinline__ int brick_coord(int g,int n,int lo,int hi){
  if(g<lo) g += ((lo-g+n-1)/n)*n;
  if(g>hi) g -= ((g-hi+n-1)/n)*n;
  return (g>=lo && g<=hi) ? g-lo : -1;
}
__device__ __forceinline__ size_t bidx(int ix,int iy,int iz,int nx,int ny){
  return (size_t)iz*ny*nx+(size_t)iy*nx+ix;
}

template<int ORD>
__global__ void k_spread_cubes2_brick(int nlocal,const double*x,
    double xlo,double ylo,double zlo,double lx,double ly,double lz,
    const double*q,double rho_scale,int nx,int ny,int nz,
    int bxlo,int bxhi,int bylo,int byhi,int bzlo,int bzhi,
    unsigned long long*rho){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=nlocal) return;
  double fx=periodic_frac(x[3*i+0],xlo,lx)*nx;
  double fy=periodic_frac(x[3*i+1],ylo,ly)*ny;
  double fz=periodic_frac(x[3*i+2],zlo,lz)*nz;
  int ix0=(int)floor(fx),iy0=(int)floor(fy),iz0=(int)floor(fz);
  double tx=fx-ix0,ty=fy-iy0,tz=fz-iz0;
  // Fixed-point (2^40) integer atomics, exactly as the replicated spread: the
  // accumulation is reproducible and runs on the fp32 pipe.
  const float qs=(float)rho_scale*(float)q[i];
  const float kfs=(float)kFxScale;
  int bnx=bxhi-bxlo+1,bny=byhi-bylo+1;
  const int NN=(ORD==4)?32:88; const double xi=(ORD==4)?kXi4:kXi6;
  for(int k=0;k<NN;++k){
    GpuNode nd=(ORD==4)?c_nodes4[k]:c_nodes6[k];
    double w=(ORD==4)?cs2_w4(tx,ty,tz,nd,xi):cs2_w6(tx,ty,tz,nd,xi);
    if(w==0.0) continue;
    int gx=brick_coord(ix0+nd.dx,nx,bxlo,bxhi);
    int gy=brick_coord(iy0+nd.dy,ny,bylo,byhi);
    int gz=brick_coord(iz0+nd.dz,nz,bzlo,bzhi);
    if(gx>=0&&gy>=0&&gz>=0)
      atomicAdd(&rho[bidx(gx,gy,gz,bnx,bny)],
                (unsigned long long)(long long)__float2ll_rn(qs*(float)w*kfs));
  }
}

template<int NORD>
__global__ void k_spread_quads_brick(int nlocal,const double*x,
    double xlo,double ylo,double zlo,double lx,double ly,double lz,
    const double*q,double rho_scale,int nx,int ny,int nz,
    int bxlo,int bxhi,int bylo,int byhi,int bzlo,int bzhi,
    unsigned long long*rho){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=nlocal) return;
  double fx=periodic_frac(x[3*i+0],xlo,lx)*nx;
  double fy=periodic_frac(x[3*i+1],ylo,ly)*ny;
  double fz=periodic_frac(x[3*i+2],zlo,lz)*nz;
  int ix0=(int)floor(fx),iy0=(int)floor(fy),iz0=(int)floor(fz);
  double tx=fx-ix0,ty=fy-iy0,tz=fz-iz0;
  const float qs=(float)rho_scale*(float)q[i];
  const float kfs=(float)kFxScale;
  double wx[NORD],wy[NORD],wz[NORD];
  if(NORD==4){quads4_w(tx,wx);quads4_w(ty,wy);quads4_w(tz,wz);}
  else {quads6_w(tx,wx);quads6_w(ty,wy);quads6_w(tz,wz);}
  int off=(NORD==4)?-1:-2,bnx=bxhi-bxlo+1,bny=byhi-bylo+1;
  for(int a=0;a<NORD;++a){int gx=brick_coord(ix0+off+a,nx,bxlo,bxhi);if(gx<0)continue;
    for(int b=0;b<NORD;++b){int gy=brick_coord(iy0+off+b,ny,bylo,byhi);if(gy<0)continue;
      for(int c=0;c<NORD;++c){int gz=brick_coord(iz0+off+c,nz,bzlo,bzhi);if(gz<0)continue;
        atomicAdd(&rho[bidx(gx,gy,gz,bnx,bny)],
                  (unsigned long long)(long long)__float2ll_rn(
                      qs*(float)(wx[a]*wy[b]*wz[c])*kfs));}}}
}

template<int ORD>
__global__ void k_gather_cubes2_brick(int nlocal,const double*x,
    double xlo,double ylo,double zlo,double lx,double ly,double lz,
    const double*q,double qscale,int nx,int ny,int nz,
    int bxlo,int bxhi,int bylo,int byhi,int bzlo,int bzhi,const float*u,
    double*force,double*vpot){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=nlocal) return;
  double fx=periodic_frac(x[3*i+0],xlo,lx)*nx;
  double fy=periodic_frac(x[3*i+1],ylo,ly)*ny;
  double fz=periodic_frac(x[3*i+2],zlo,lz)*nz;
  int ix0=(int)floor(fx),iy0=(int)floor(fy),iz0=(int)floor(fz);
  double tx=fx-ix0,ty=fy-iy0,tz=fz-iz0,dtx=0,dty=0,dtz=0,gp=0;
  int bnx=bxhi-bxlo+1,bny=byhi-bylo+1;
  const int NN=(ORD==4)?32:88; const double xi=(ORD==4)?kXi4:kXi6;
  for(int k=0;k<NN;++k){
    GpuNode nd=(ORD==4)?c_nodes4[k]:c_nodes6[k];
    GpuWeightGrad wg=(ORD==4)?cs2_wg4(tx,ty,tz,nd,xi):cs2_wg6(tx,ty,tz,nd,xi);
    if(wg.v==0.0) continue;
    int gx=brick_coord(ix0+nd.dx,nx,bxlo,bxhi);
    int gy=brick_coord(iy0+nd.dy,ny,bylo,byhi);
    int gz=brick_coord(iz0+nd.dz,nz,bzlo,bzhi);
    if(gx<0||gy<0||gz<0) continue;
    double up=(double)u[bidx(gx,gy,gz,bnx,bny)];
    dtx+=wg.dx*up;dty+=wg.dy*up;dtz+=wg.dz*up;gp+=wg.v*up;
  }
  double qi=q[i];
  force[3*i+0]=-qscale*qi*dtx*nx/lx;
  force[3*i+1]=-qscale*qi*dty*ny/ly;
  force[3*i+2]=-qscale*qi*dtz*nz/lz;
  vpot[i]=qscale*gp;
}

template<int NORD>
__global__ void k_gather_quads_brick(int nlocal,const double*x,
    double xlo,double ylo,double zlo,double lx,double ly,double lz,
    const double*q,double qscale,int nx,int ny,int nz,
    int bxlo,int bxhi,int bylo,int byhi,int bzlo,int bzhi,const float*u,
    double*force,double*vpot){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=nlocal) return;
  double fx=periodic_frac(x[3*i+0],xlo,lx)*nx;
  double fy=periodic_frac(x[3*i+1],ylo,ly)*ny;
  double fz=periodic_frac(x[3*i+2],zlo,lz)*nz;
  int ix0=(int)floor(fx),iy0=(int)floor(fy),iz0=(int)floor(fz);
  double tx=fx-ix0,ty=fy-iy0,tz=fz-iz0,dtx=0,dty=0,dtz=0,gp=0;
  double wx[NORD],wy[NORD],wz[NORD],dwx[NORD],dwy[NORD],dwz[NORD];
  if(NORD==4){quads4_wd(tx,wx,dwx);quads4_wd(ty,wy,dwy);quads4_wd(tz,wz,dwz);}
  else {quads6_wd(tx,wx,dwx);quads6_wd(ty,wy,dwy);quads6_wd(tz,wz,dwz);}
  int off=(NORD==4)?-1:-2,bnx=bxhi-bxlo+1,bny=byhi-bylo+1;
  for(int a=0;a<NORD;++a){int gx=brick_coord(ix0+off+a,nx,bxlo,bxhi);if(gx<0)continue;
    for(int b=0;b<NORD;++b){int gy=brick_coord(iy0+off+b,ny,bylo,byhi);if(gy<0)continue;
      for(int c=0;c<NORD;++c){int gz=brick_coord(iz0+off+c,nz,bzlo,bzhi);if(gz<0)continue;
        double up=(double)u[bidx(gx,gy,gz,bnx,bny)];
        dtx+=dwx[a]*wy[b]*wz[c]*up;dty+=wx[a]*dwy[b]*wz[c]*up;
        dtz+=wx[a]*wy[b]*dwz[c]*up;gp+=wx[a]*wy[b]*wz[c]*up;}}}
  double qi=q[i];
  force[3*i+0]=-qscale*qi*dtx*nx/lx;
  force[3*i+1]=-qscale*qi*dty*ny/ly;
  force[3*i+2]=-qscale*qi*dtz*nz/lz;
  vpot[i]=qscale*gp;
}

__global__ void k_zero_complex(cufftDoubleComplex*a,size_t n){
  size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n){a[i].x=0;a[i].y=0;}
}

// ── B-spline brick stencils (cardinal B-spline of order NORD) ──────────────
// Ported from the deployed LAMMPS GPU B-spline path; same Cox-de Boor
// recurrence as the CPU mesh_sum.cu weights, so the slab solve and the
// replicated SPME evaluate the same assignment.  The B-spline reaches NORD
// points per axis starting at floor(u) - (NORD-1)/2, matching the
// assignment factor |D_d(m)|^-2 that build_green_slab_half inverts.
template<int N>
__device__ __forceinline__ void bspline_w1d(double frac,double*w){
  w[0]=1.0-frac; w[1]=frac;
  #pragma unroll
  for(int j=2;j<N;++j) w[j]=0.0;
  #pragma unroll
  for(int k=3;k<=N;++k){
    const double inv=1.0/(double)(k-1);
    double wp[8];
    #pragma unroll
    for(int j=0;j<N;++j) wp[j]=w[j];
    w[k-1]=frac*wp[k-2]*inv;
    #pragma unroll
    for(int j=1;j<=k-2;++j)
      w[k-1-j]=((frac+(double)j)*wp[k-2-j]+((double)(k-j)-frac)*wp[k-1-j])*inv;
    w[0]=(1.0-frac)*wp[0]*inv;
  }
}

template<int N>
__device__ __forceinline__ void bspline_w1d_d(double frac,double*w,double*dw){
  w[0]=1.0-frac; dw[0]=-1.0;
  w[1]=frac;     dw[1]=1.0;
  #pragma unroll
  for(int j=2;j<N;++j){ w[j]=0.0; dw[j]=0.0; }
  #pragma unroll
  for(int k=3;k<=N;++k){
    const double inv=1.0/(double)(k-1);
    double wp[8],dwp[8];
    #pragma unroll
    for(int j=0;j<N;++j){ wp[j]=w[j]; dwp[j]=dw[j]; }
    w[k-1]=frac*wp[k-2]*inv;
    dw[k-1]=(wp[k-2]+frac*dwp[k-2])*inv;
    #pragma unroll
    for(int j=1;j<=k-2;++j){
      w[k-1-j]=((frac+(double)j)*wp[k-2-j]+((double)(k-j)-frac)*wp[k-1-j])*inv;
      dw[k-1-j]=(wp[k-2-j]+(frac+(double)j)*dwp[k-2-j]
                 -wp[k-1-j]+((double)(k-j)-frac)*dwp[k-1-j])*inv;
    }
    w[0]=(1.0-frac)*wp[0]*inv;
    dw[0]=(-wp[0]+(1.0-frac)*dwp[0])*inv;
  }
}

template<int NORD>
__global__ void k_spread_bspline_brick(int nlocal,const double*x,
    double xlo,double ylo,double zlo,double lx,double ly,double lz,
    const double*q,double rho_scale,int nx,int ny,int nz,
    int bxlo,int bxhi,int bylo,int byhi,int bzlo,int bzhi,
    unsigned long long*rho){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=nlocal) return;
  double fx=periodic_frac(x[3*i+0],xlo,lx)*nx;
  double fy=periodic_frac(x[3*i+1],ylo,ly)*ny;
  double fz=periodic_frac(x[3*i+2],zlo,lz)*nz;
  int ix0=(int)floor(fx),iy0=(int)floor(fy),iz0=(int)floor(fz);
  double tx=fx-ix0,ty=fy-iy0,tz=fz-iz0;
  const float qs=(float)rho_scale*(float)q[i];
  const float kfs=(float)kFxScale;
  double wx[NORD],wy[NORD],wz[NORD];
  bspline_w1d<NORD>(tx,wx); bspline_w1d<NORD>(ty,wy); bspline_w1d<NORD>(tz,wz);
  const int half=(NORD-1)/2, bnx=bxhi-bxlo+1, bny=byhi-bylo+1;
  for(int a=0;a<NORD;++a){ int gx=brick_coord(ix0-half+a,nx,bxlo,bxhi); if(gx<0)continue;
    for(int b=0;b<NORD;++b){ int gy=brick_coord(iy0-half+b,ny,bylo,byhi); if(gy<0)continue;
      for(int c=0;c<NORD;++c){ int gz=brick_coord(iz0-half+c,nz,bzlo,bzhi); if(gz<0)continue;
        atomicAdd(&rho[bidx(gx,gy,gz,bnx,bny)],
                  (unsigned long long)(long long)__float2ll_rn(
                      qs*(float)(wx[a]*wy[b]*wz[c])*kfs)); } } }
}

template<int NORD>
__global__ void k_gather_bspline_brick(int nlocal,const double*x,
    double xlo,double ylo,double zlo,double lx,double ly,double lz,
    const double*q,double qscale,int nx,int ny,int nz,
    int bxlo,int bxhi,int bylo,int byhi,int bzlo,int bzhi,const float*u,
    double*force,double*vpot){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=nlocal) return;
  double fx=periodic_frac(x[3*i+0],xlo,lx)*nx;
  double fy=periodic_frac(x[3*i+1],ylo,ly)*ny;
  double fz=periodic_frac(x[3*i+2],zlo,lz)*nz;
  int ix0=(int)floor(fx),iy0=(int)floor(fy),iz0=(int)floor(fz);
  double tx=fx-ix0,ty=fy-iy0,tz=fz-iz0;
  double wx[NORD],wy[NORD],wz[NORD],dwx[NORD],dwy[NORD],dwz[NORD];
  bspline_w1d_d<NORD>(tx,wx,dwx); bspline_w1d_d<NORD>(ty,wy,dwy);
  bspline_w1d_d<NORD>(tz,wz,dwz);
  const int half=(NORD-1)/2, bnx=bxhi-bxlo+1, bny=byhi-bylo+1;
  double dtx=0,dty=0,dtz=0,gp=0;
  for(int a=0;a<NORD;++a){ int gx=brick_coord(ix0-half+a,nx,bxlo,bxhi); if(gx<0)continue;
    for(int b=0;b<NORD;++b){ int gy=brick_coord(iy0-half+b,ny,bylo,byhi); if(gy<0)continue;
      for(int c=0;c<NORD;++c){ int gz=brick_coord(iz0-half+c,nz,bzlo,bzhi); if(gz<0)continue;
        double up=(double)u[bidx(gx,gy,gz,bnx,bny)];
        double wyz=wy[b]*wz[c], wxz=wx[a]*wz[c], wxy=wx[a]*wy[b];
        gp+=wx[a]*wyz*up; dtx+=dwx[a]*wyz*up; dty+=dwy[b]*wxz*up; dtz+=dwz[c]*wxy*up; } } }
  double qi=q[i];
  force[3*i+0]=-qscale*qi*dtx*nx/lx;
  force[3*i+1]=-qscale*qi*dty*ny/ly;
  force[3*i+2]=-qscale*qi*dtz*nz/lz;
  vpot[i]=qscale*gp;
}


// ── host API ──
extern "C" SogGpuState *sog_gpu_create(){
  // upload CubeS2 node tables to constant memory (layout GpuNode == CubeS2Node4)
  cudaMemcpyToSymbol(c_nodes4, LAMMPS_NS::kCubes2Nodes4, 32*sizeof(GpuNode));
  cudaMemcpyToSymbol(c_nodes6, LAMMPS_NS::kCubes2Nodes6, 88*sizeof(GpuNode));
  {
    auto key = [](const GpuNode& n) {
      return (long long)(n.dz + 4) * 10000LL + (long long)(n.dy + 4) * 100LL +
             (long long)(n.dx + 4);
    };
    std::vector<GpuNode> s4(LAMMPS_NS::kCubes2Nodes4,
                            LAMMPS_NS::kCubes2Nodes4 + 32);
    std::vector<GpuNode> s6(LAMMPS_NS::kCubes2Nodes6,
                            LAMMPS_NS::kCubes2Nodes6 + 88);
    std::sort(s4.begin(), s4.end(),
              [&](const GpuNode& a, const GpuNode& b) { return key(a) < key(b); });
    std::sort(s6.begin(), s6.end(),
              [&](const GpuNode& a, const GpuNode& b) { return key(a) < key(b); });
    cudaMemcpyToSymbol(c_nodes4s, s4.data(), 32*sizeof(GpuNode));
    cudaMemcpyToSymbol(c_nodes6s, s6.data(), 88*sizeof(GpuNode));
  }
  // QuadS tables: (2nu)^3 nodes, x fastest, offsets 1-nu..nu.
  {
    std::vector<QNodeS> q4(64), q6(216);
    int k=0;
    for(int c=0;c<4;++c) for(int b=0;b<4;++b) for(int a=0;a<4;++a)
      q4[k++]={ (signed char)(a-1),(signed char)(b-1),(signed char)(c-1),
                (signed char)a,(signed char)b,(signed char)c };
    k=0;
    for(int c=0;c<6;++c) for(int b=0;b<6;++b) for(int a=0;a<6;++a)
      q6[k++]={ (signed char)(a-2),(signed char)(b-2),(signed char)(c-2),
                (signed char)a,(signed char)b,(signed char)c };
    cudaMemcpyToSymbol(c_qnodes4s, q4.data(), 64*sizeof(QNodeS));
    cudaMemcpyToSymbol(c_qnodes6s, q6.data(), 216*sizeof(QNodeS));
  }
  return new SogGpuState();
}
static void free_grid_buffers(SogGpuState*s);  // defined with the slab setup below

extern "C" void sog_gpu_destroy(SogGpuState*s){
  if(!s) return;
  if(s->plan_xy_rc) cufftDestroy(s->plan_xy_rc);
  if(s->plan_xy_cr) cufftDestroy(s->plan_xy_cr);
  if(s->plan_z) cufftDestroy(s->plan_z);
  free_grid_buffers(s);  // plan, d_* and the float mesh set; nulls them
  cudaFree(s->d_x);cudaFree(s->d_q);cudaFree(s->d_force);cudaFree(s->d_vpot);cudaFree(s->d_red);
  cudaFree(s->d_boxlo);
  cudaFree(s->d_brick);
  cudaFree(s->s_brick_fx);
  cudaFree(s->d_halo);
  cudaFree(s->d_frac);
  delete s;
}


// ── Distributed (slab) k-space: device side ────────────────────────────────
// The host orchestrator (sog_kspace_op.cc) owns the decomposition, the halo
// exchange and the transpose; these entry points only move one buffer to the
// device, transform it, and bring it back.  Correctness first: no device-side
// MPI (this machine has neither CUDA-aware MPI nor a working NCCL inside the
// op), so every exchange is host-staged by the caller.
//
// Layouts (both keep x in the middle, which is what cuFFT wants):
//   "xy": (nx, ny, ZL)         z slowest, contiguous (nx,ny) planes, one per
//                              local z.  rank-2 plan over (x,y), batch ZL.
//   "z" : (nz, nx, YL)         z FASTEST, so the z transform is a plain
//                              contiguous batch of nx*YL vectors of length nz.
extern "C" int sog_gpu_slab_setup(SogGpuState*s,int nx,int ny,int nz,
    int ZL,int YL){
  const int Nh=nx/2+1;
  // The plans depend on the GRID (nx,ny,nz), not just the slab shapes: a stale
  // reuse after a grid change was a latent bug even before R2C.
  if(s->slab_ZL==ZL && s->slab_YL==YL && s->slab_Nh==Nh && s->plan_xy_rc &&
     s->plan_xy_cr && s->plan_z) return 0;
  if(s->plan_xy_rc) cufftDestroy(s->plan_xy_rc);
  if(s->plan_xy_cr) cufftDestroy(s->plan_xy_cr);
  if(s->plan_z) cufftDestroy(s->plan_z);
  s->plan_xy_rc=0; s->plan_xy_cr=0; s->plan_z=0;
  // cuFFT takes sizes SLOWEST -> FASTEST.  Getting this backwards still looks
  // right on cubic grids and aliases once nx != ny; see
  // sog_work/dist_mesh/README.md.
  //
  // SINGLE PRECISION, like the replicated solve: cufftComplex spectra and
  // cufftReal meshes.  The brick is float32 and the density accumulation is
  // 64-bit fixed point, so the whole slab pipeline shares the replicated
  // path's precision contract instead of running a fp64 mesh beside it.
  int n2[2]={ny,nx}, rin[2]={ny,nx}, cout[2]={ny,Nh};
  if(cufftPlanMany(&s->plan_xy_rc,2,n2,rin,1,(size_t)nx*ny,cout,1,
                   (size_t)ny*Nh,CUFFT_R2C,ZL)!=CUFFT_SUCCESS) return -1;
  if(cufftPlanMany(&s->plan_xy_cr,2,n2,cout,1,(size_t)ny*Nh,rin,1,
                   (size_t)nx*ny,CUFFT_C2R,ZL)!=CUFFT_SUCCESS){
    cufftDestroy(s->plan_xy_rc); s->plan_xy_rc=0; return -1;
  }
  int n1[1]={nz}, e1[1]={nz};
  if(cufftPlanMany(&s->plan_z,1,n1,e1,1,nz,e1,1,nz,CUFFT_C2C,Nh*YL)
     !=CUFFT_SUCCESS){
    cufftDestroy(s->plan_xy_rc); cufftDestroy(s->plan_xy_cr);
    s->plan_xy_rc=0; s->plan_xy_cr=0; return -1;
  }
  s->slab_ZL=ZL; s->slab_YL=YL; s->slab_Nh=Nh;
  return 0;
}

// green multiply on the local pencil; returns the 7 accumulators in red[0..6]
// ── Device-resident transpose stages ───────────────────────────────────────
// Everything between the MPI messages stays on the GPU: the host only sees the
// packed block matrices.  This removes ~0.9 GB/step of strided host traffic at
// a 150^3 mesh (four pack/unpack traversals plus four extra H<->D copies).
//
//   forward : cur (xy slab) -> FFT(x,y) -> pack -> [MPI] -> unpack -> FFT(z)
//             -> green -> FFT^-1(z) -> pack -> [MPI] -> unpack -> FFT^-1(x,y)

// Device-resident variant: the received blocks are ALREADY in d_rho; the
// packed inverse blocks are left in d_rho and the local accumulators in d_red.
extern "C" int sog_gpu_slab_unpack_green_izpack_dev(SogGpuState*s,int nx,int ny,
    int nz,int ylo,int YL,double lx,double ly,double lz,double scaleinv,int np){
  if(s->s_rho==nullptr||s->s_pot==nullptr||s->plan_z==0) return -1;
  const int Nh=nx/2+1;
  const size_t nc=(size_t)nz*Nh*YL;              // complex elements each way
  int tb=256; size_t gg=(nc+tb-1)/tb;
  k_unpack_blocks_to_pen<<<(int)gg,tb>>>(Nh,ny,nz,np,YL,s->s_rho,s->s_pot);
  CK(cudaGetLastError());
  if(cufftExecC2C(s->plan_z,s->s_pot,s->s_pot,CUFFT_FORWARD)!=CUFFT_SUCCESS)
    return -2;
  if(s->d_red==nullptr) CK(cudaMalloc(&s->d_red,20*sizeof(double)));
  CK(cudaMemset(s->d_red,0,20*sizeof(double)));
  k_green_pencil<<<(int)gg,tb>>>(Nh,nx,ny,nz,ylo,YL,lx,ly,lz,s->s_pot,
                                s->s_gp,s->s_pot,scaleinv,s->d_red);
  CK(cudaGetLastError());
  if(cufftExecC2C(s->plan_z,s->s_pot,s->s_pot,CUFFT_INVERSE)!=CUFFT_SUCCESS)
    return -3;
  k_pack_pen_to_blocks<<<(int)gg,tb>>>(Nh,ny,nz,np,YL,s->s_pot,s->s_rho);
  CK(cudaGetLastError());
  return 0;
}
// Host-buffer wrapper (MPI fallback path).
extern "C" int sog_gpu_slab_unpack_green_izpack(SogGpuState*s,int nx,int ny,
    int nz,int ylo,int YL,double lx,double ly,double lz,double scaleinv,int np,
    const double*host_rcv,double*host_snd_back,double*red7){
  const size_t nc=(size_t)nz*(size_t)(nx/2+1)*YL;
  CK(cudaMemcpy(s->s_rho,host_rcv,2*nc*sizeof(float),cudaMemcpyHostToDevice));
  const int rc=sog_gpu_slab_unpack_green_izpack_dev(s,nx,ny,nz,ylo,YL,lx,ly,lz,
                                                    scaleinv,np);
  if(rc!=0) return rc;
  double red[20]; CK(cudaMemcpy(red,s->d_red,20*sizeof(double),cudaMemcpyDeviceToHost));
  for(int t=0;t<7;++t) red7[t]=red[t];
  CK(cudaMemcpy(host_snd_back,s->s_rho,2*nc*sizeof(float),cudaMemcpyDeviceToHost));
  return 0;
}

// unpack [MPI] -> FFT^-1(x,y) -> cur2 (xy slab)


// ── NCCL device collectives (dlopen, no build/link dependency) ─────────────
// The transpose all-to-all and the halo exchanges are device-to-device with
// NCCL, so the packed messages never cross the bus.  The op library is also
// compiled into the LAMMPS plugin (torch-free), so NCCL is resolved at run
// time instead of being a link dependency: if it cannot be loaded or the
// communicator cannot be created, every entry returns non-zero and the caller
// keeps using the host-staged MPI path.
namespace {
// ncclUniqueId is a 128-byte struct passed BY VALUE to ncclCommInitRank; using
// a pointer in the function-pointer cast changes the ABI and NCCL answers
// "invalid argument".
struct NcclUniqueId{ char internal[128]; };
struct NcclApi {
  bool loaded=false, ok=false;
  int (*get_id)(NcclUniqueId*)=nullptr;
  int (*init_rank)(void**,int,NcclUniqueId,int)=nullptr;
  int (*destroy)(void*)=nullptr;
  int (*group_start)()=nullptr;
  int (*group_end)()=nullptr;
  int (*send)(const void*,size_t,int,int,void*,cudaStream_t)=nullptr;
  int (*recv)(void*,size_t,int,int,void*,cudaStream_t)=nullptr;
  int (*all_reduce)(const void*,void*,size_t,int,int,void*,cudaStream_t)=nullptr;
  const char* (*err_str)(int)=nullptr;
};
NcclApi &nccl_api(){
  static NcclApi a;
  if(a.loaded) return a;
  a.loaded=true;
  const char *cands[5];
  int nc=0;
  const char *e=getenv("SOG_NCCL_LIB");
  if(e) cands[nc++]=e;
  cands[nc++]="libnccl.so.2";
  cands[nc++]="libnccl.so";
  // Last resort for this deployment: the NCCL that ships inside the torch
  // wheel's nvidia/ tree, which is not on LD_LIBRARY_PATH by default.
  cands[nc++]="/data/home/jiangzhen/.conda/envs/lr/lib/python3.11/"
              "site-packages/nvidia/nccl/lib/libnccl.so.2";
  for(int i=0;i<nc && !a.ok;++i){
    void *lib=dlopen(cands[i],RTLD_NOW|RTLD_LOCAL);
    if(!lib) continue;
    a.get_id=(int(*)(NcclUniqueId*))dlsym(lib,"ncclGetUniqueId");
    a.init_rank=(int(*)(void**,int,NcclUniqueId,int))dlsym(lib,"ncclCommInitRank");
    a.destroy=(int(*)(void*))dlsym(lib,"ncclCommDestroy");
    a.group_start=(int(*)())dlsym(lib,"ncclGroupStart");
    a.group_end=(int(*)())dlsym(lib,"ncclGroupEnd");
    a.send=(int(*)(const void*,size_t,int,int,void*,cudaStream_t))dlsym(lib,"ncclSend");
    a.recv=(int(*)(void*,size_t,int,int,void*,cudaStream_t))dlsym(lib,"ncclRecv");
    a.all_reduce=(int(*)(const void*,void*,size_t,int,int,void*,cudaStream_t))dlsym(lib,"ncclAllReduce");
    a.err_str=(const char*(*)(int))dlsym(lib,"ncclGetErrorString");
    a.ok=a.get_id&&a.init_rank&&a.destroy&&a.group_start&&a.group_end&&a.send&&
         a.recv&&a.all_reduce;
  }
  return a;
}
constexpr int kNcclDouble=8;    // ncclDataType_t: ncclFloat64/ncclDouble
constexpr int kNcclFloat32=7;   // ncclDataType_t: ncclFloat32/ncclFloat
constexpr int kNcclUint64=5;    // ncclDataType_t: ncclUint64 (exact integer sum)
constexpr int kNcclSum=0;       // ncclRedOp_t

// One communicator per process: neutralize/qmean (which have no handle) and
// solve_auto share it, so the first collective can be the neutralization.
void *g_comm=nullptr;
int g_rank=-1, g_np=-1, g_state=0;   // 0 = not tried, 1 = usable, -1 = unavailable
}  // namespace

extern "C" int sog_gpu_nccl_get_id(void *id){
  NcclApi &a=nccl_api();
  return (a.ok && a.get_id(static_cast<NcclUniqueId*>(id))==0) ? 0 : -1;
}

extern "C" int sog_gpu_nccl_init(int np,int me,const void *id){
  if(g_state==1 && g_np==np && g_rank==me) return 0;
  if(g_state==-1) return -1;
  if(getenv("SOG_NCCL")!=nullptr && atoi(getenv("SOG_NCCL"))==0){
    g_state=-1; return -1;
  }
  NcclApi &a=nccl_api();
  if(!a.ok){ g_state=-1; return -1; }
  void *comm=nullptr;
  const NcclUniqueId uid=*static_cast<const NcclUniqueId*>(id);
  const int rc=a.init_rank(&comm,np,uid,me);
  if(rc!=0){
    if(getenv("SOG_DIST_DEBUG"))
      fprintf(stderr,"[sog-nccl] init failed (%s); falling back to MPI\n",
              a.err_str?a.err_str(rc):"error");
    g_state=-1; return -1;
  }
  g_comm=comm; g_np=np; g_rank=me; g_state=1;
  return 0;
}
extern "C" int sog_gpu_nccl_ok(){ return g_state==1 ? 1 : 0; }
extern "C" int sog_gpu_nccl_available(){ return nccl_api().ok ? 1 : 0; }

// The packed blocks are cufftComplex (8 bytes).  scnt/rcnt/rdisp count in
// FLOAT elements (2 per complex), so the byte count and the offsets are the
// same numbers the fp64 version used: `send + disp` advances disp floats =
// 4*disp bytes = 8*complex, exactly one cufftComplex per step.
extern "C" int sog_gpu_nccl_alltoall(SogGpuState*s,const float*send,
    const int*scnt,const int*sdisp,float*recv,const int*rcnt,
    const int*rdisp,int np){
  NcclApi &a=nccl_api();
  if(!a.ok||g_state!=1) return -1;
  if(a.group_start()!=0) return -2;
  for(int d=0;d<np;++d)
    a.send(send+sdisp[d],(size_t)scnt[d],kNcclFloat32,d,g_comm,(cudaStream_t)0);
  for(int r=0;r<np;++r)
    a.recv(recv+rdisp[r],(size_t)rcnt[r],kNcclFloat32,r,g_comm,(cudaStream_t)0);
  return a.group_end()==0 ? 0 : -3;
}

// Two-direction halo without tags.  NCCL pairs messages per peer by posting
// order, so the order below is chosen to be correct when prev == next (np=2):
//   send->prev, recv<-next, send->next, recv<-prev
// At np=2 the 1st send pairs with the peer's 1st recv and the 2nd with the 2nd;
// at np>2 each direction has a distinct peer, so the order is irrelevant.
// `dtype` selects the element width: kNcclDouble for the 8-byte fixed-point
// density halo, kNcclFloat32 for the 4-byte potential halo.
extern "C" int sog_gpu_nccl_halo2(SogGpuState*s,int prev,int next,
    const void*send_prev,int cnt_prev,const void*send_next,int cnt_next,
    void*recv_prev,void*recv_next,int dtype){
  NcclApi &a=nccl_api();
  if(!a.ok||g_state!=1) return -1;
  if(a.group_start()!=0) return -2;
  a.send(send_prev,(size_t)cnt_prev,dtype,prev,g_comm,(cudaStream_t)0);
  a.recv(recv_next,(size_t)cnt_next,dtype,next,g_comm,(cudaStream_t)0);
  a.send(send_next,(size_t)cnt_next,dtype,next,g_comm,(cudaStream_t)0);
  a.recv(recv_prev,(size_t)cnt_prev,dtype,prev,g_comm,(cudaStream_t)0);
  return a.group_end()==0 ? 0 : -3;
}

__global__ void k_add_into(double*dst,const double*src,size_t n);  // defined below
__global__ void k_add_into_u64(unsigned long long*dst,
                               const unsigned long long*src,size_t n);

// Forward all-to-all: packed forward blocks in s_pot -> received blocks in s_rho.
extern "C" int sog_gpu_nccl_alltoall_fwd(SogGpuState*s,const int*scnt,
    const int*sdisp,const int*rcnt,const int*rdisp,int np){
  return sog_gpu_nccl_alltoall(s,(const float*)s->s_pot,scnt,sdisp,
                               (float*)s->s_rho,rcnt,rdisp,np);
}
// The np == 1 "all-to-all" is the identity permutation: there is one block, it
// must move s_pot -> s_rho (forward) or s_rho -> s_pot (inverse), and no rank
// has to talk to anyone -- least of all to itself through NCCL, a path the
// reference never exercises.  n is in FLOAT units (2 per cufftComplex), as
// scnt/rcnt are.
extern "C" int sog_gpu_slab_copy_packed(SogGpuState*s,int forward,size_t n){
  if(s->s_rho==nullptr||s->s_pot==nullptr) return -1;
  const size_t bytes=n*sizeof(float);
  CK(cudaMemcpy(forward?(void*)s->s_rho:(void*)s->s_pot,
                forward?(const void*)s->s_pot:(const void*)s->s_rho,bytes,
                cudaMemcpyDeviceToDevice));
  return 0;
}
// Inverse: packed inverse blocks in s_rho -> received blocks in s_pot.
extern "C" int sog_gpu_nccl_alltoall_bwd(SogGpuState*s,const int*scnt,
    const int*sdisp,const int*rcnt,const int*rdisp,int np){
  return sog_gpu_nccl_alltoall(s,(const float*)s->s_rho,scnt,sdisp,
                               (float*)s->s_pot,rcnt,rdisp,np);
}

// Reverse halo (density overlap-add) on the FIXED-POINT device brick: the
// integer planes are summed exactly, matching the replicated spread's
// reproducibility.  Requires gh_lo == gh_hi (both are the interpolation order).
extern "C" int sog_gpu_nccl_halo_rev(SogGpuState*s,int prev,int next,
    size_t plane,int zoff,int ZL,int gh_lo,int gh_hi){
  if(gh_lo!=gh_hi) return -4;
  const size_t n=(size_t)plane*(size_t)gh_lo;
  if(2*n>s->cap_halo){
    if(s->d_halo) cudaFree(s->d_halo);
    CK(cudaMalloc(&s->d_halo,2*n*sizeof(unsigned long long)));
    s->cap_halo=2*n;
  }
  unsigned long long *recv_prev=(unsigned long long*)s->d_halo,
                     *recv_next=recv_prev+n;
  const unsigned long long *send_lo=s->s_brick_fx;                    // lower ghost
  const unsigned long long *send_hi=s->s_brick_fx+(size_t)(zoff+ZL)*plane;  // upper ghost
  if(sog_gpu_nccl_halo2(s,prev,next,send_lo,(int)n,send_hi,(int)n,recv_prev,
                        recv_next,kNcclDouble)!=0) return -1;
  int tb=256; size_t gg=(n+tb-1)/tb;
  k_add_into_u64<<<(int)gg,tb>>>(s->s_brick_fx+(size_t)zoff*plane,recv_prev,n);
  k_add_into_u64<<<(int)gg,tb>>>(s->s_brick_fx+(size_t)(zoff+ZL-gh_lo)*plane,
                                 recv_next,n);
  CK(cudaGetLastError());
  return 0;
}
// Forward halo (potential ghosts) on the FLOAT device brick.
extern "C" int sog_gpu_nccl_halo_fwd(SogGpuState*s,int prev,int next,
    size_t plane,int zoff,int ZL,int gh_lo,int gh_hi){
  if(gh_lo!=gh_hi) return -4;
  const size_t n=(size_t)plane*(size_t)gh_lo;
  if(n>s->cap_halo){
    if(s->d_halo) cudaFree(s->d_halo);
    CK(cudaMalloc(&s->d_halo,2*n*sizeof(float)));
    s->cap_halo=n;
  }
  float *recv_prev=(float*)s->d_halo, *recv_next=recv_prev+n;
  const float *send_first=s->d_brick+(size_t)zoff*plane;           // first owned
  const float *send_last=s->d_brick+(size_t)(zoff+ZL-gh_lo)*plane; // last owned
  if(sog_gpu_nccl_halo2(s,prev,next,send_first,(int)n,send_last,(int)n,
                        recv_prev,recv_next,kNcclFloat32)!=0) return -1;
  CK(cudaMemcpy(s->d_brick,recv_prev,n*sizeof(float),cudaMemcpyDeviceToDevice));
  CK(cudaMemcpy(s->d_brick+(size_t)(zoff+ZL)*plane,recv_next,n*sizeof(float),
                cudaMemcpyDeviceToDevice));
  return 0;
}

// Generic in-place device all-reduce with an explicit element width.  The
// full-brick path needs the 64-bit fixed-point density (ncclUint64: an exact
// integer sum, so the global mesh does not depend on the reduction order) and
// the float potential (ncclFloat32); neutralize/qmean use doubles.
extern "C" int sog_gpu_nccl_allreduce_dev_dtype(void *buf,size_t n,int dtype){
  NcclApi &a=nccl_api();
  if(!a.ok||g_state!=1) return -1;
  return a.all_reduce(buf,buf,n,dtype,kNcclSum,g_comm,(cudaStream_t)0)==0
             ? 0 : -2;
}
extern "C" int sog_gpu_nccl_allreduce_dev(void *buf,size_t n){
  return sog_gpu_nccl_allreduce_dev_dtype(buf,n,kNcclDouble);
}

// full_brick: every rank spread its own atoms over the WHOLE z range, so the
// rank sum is the global mesh.  `off`/`n` are element counts inside the brick
// (the owned planes, [gh*plane, (gh+nz)*plane)); the ghost planes are not part
// of the sum.
extern "C" int sog_gpu_brick_allreduce_fx(SogGpuState*s,size_t off,size_t n){
  return sog_gpu_nccl_allreduce_dev_dtype(s->s_brick_fx+off,n,kNcclUint64);
}
extern "C" int sog_gpu_brick_allreduce_f(SogGpuState*s,size_t off,size_t n){
  return sog_gpu_nccl_allreduce_dev_dtype(s->d_brick+off,n,kNcclFloat32);
}

extern "C" int sog_gpu_nccl_allreduce_red(SogGpuState*s){
  NcclApi &a=nccl_api();
  if(!a.ok||g_state!=1||s->d_red==nullptr) return -1;
  return a.all_reduce(s->d_red,s->d_red,(size_t)7,kNcclDouble,kNcclSum,
                      g_comm,(cudaStream_t)0)==0 ? 0 : -2;
}

extern "C" void sog_gpu_red_download(SogGpuState*s,double*host){
  if(s->d_red==nullptr) return;
  CK(cudaMemcpy(host,s->d_red,7*sizeof(double),cudaMemcpyDeviceToHost));
}

// cuFFT wants a suitably aligned pointer for R2C/C2R.  The brick's slab offset
// zoff*plane floats is 16-byte aligned only when the float offset is a
// multiple of 4; otherwise stage the contiguous plane run through this scratch.
static float *ensure_real_f(SogGpuState*s,size_t n){
  if(n>s->cap_real){
    if(s->s_real) cudaFree(s->s_real);
    CK(cudaMalloc(&s->s_real,n*sizeof(float)));
    s->cap_real=n;
  }
  return s->s_real;
}

// ── Device-resident brick staging ──────────────────────────────────────────
// The distributed solve keeps the owned+ghost brick on the GPU: the slab
// conversions and the halo overlap-adds run as kernels, and only the few ghost
// planes cross the bus.  This removes the full-brick D2H/H2D per step and the
// host rho_ext/u_ext/cur/cur2 buffers.
//   k_add_into      : fp64 scratch, legacy host-add helper
//   k_add_into_f    : float potential brick, legacy host-add helper
//   k_add_into_u64  : fixed-point density brick, the np==1 / NCCL reverse halo
__global__ void k_add_into(double*dst,const double*src,size_t n){
  size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) dst[i]+=src[i];
}
__global__ void k_add_into_f(float*dst,const float*src,size_t n){
  size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) dst[i]+=src[i];
}
__global__ void k_add_into_u64(unsigned long long*dst,
                               const unsigned long long*src,size_t n){
  size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) dst[i]+=src[i];
}

// plane ranges are in elements: off = plane_index * nx*ny.  The device brick is
// float; these legacy host helpers convert to/from the fp64 host API.
extern "C" void sog_gpu_brick_get(SogGpuState*s,double*host,size_t off,size_t n){
  std::vector<float> h(n);
  CK(cudaMemcpy(h.data(),s->d_brick+off,n*sizeof(float),cudaMemcpyDeviceToHost));
  for(size_t i=0;i<n;++i) host[i]=(double)h[i];
}
extern "C" void sog_gpu_brick_set(SogGpuState*s,const double*host,size_t off,
    size_t n){
  std::vector<float> h(n);
  for(size_t i=0;i<n;++i) h[i]=(float)host[i];
  CK(cudaMemcpy(s->d_brick+off,h.data(),n*sizeof(float),cudaMemcpyHostToDevice));
}
extern "C" void sog_gpu_brick_add(SogGpuState*s,const double*host,size_t off,
    size_t n){
  std::vector<float> h(n);
  for(size_t i=0;i<n;++i) h[i]=(float)host[i];
  if(n>s->cap_halo){
    if(s->d_halo) cudaFree(s->d_halo);
    CK(cudaMalloc(&s->d_halo,n*sizeof(float)));
    s->cap_halo=n;
  }
  float *hbuf=(float*)s->d_halo;
  CK(cudaMemcpy(hbuf,h.data(),n*sizeof(float),cudaMemcpyHostToDevice));
  int tb=256; size_t gg=(n+tb-1)/tb;
  k_add_into_f<<<(int)gg,tb>>>(s->d_brick+off,hbuf,n);
  CK(cudaGetLastError());
}
extern "C" void sog_gpu_brick_zero(SogGpuState*s,size_t off,size_t n){
  CK(cudaMemset(s->d_brick+off,0,n*sizeof(float)));
}
// Device-only plane moves inside the brick.  The np == 1 halo folds used the
// host get/add pairs, i.e. 4*gh synchronising bus round trips per call: at a
// 98^2 plane that alone was ~1.9 ms of a 2.2 ms solve.  These keep the data on
// the device.  add_planes is the DENSITY fold (fixed-point brick); copy_planes
// is the POTENTIAL fill (float brick).
extern "C" void sog_gpu_brick_add_planes(SogGpuState*s,size_t src_off,
    size_t dst_off,size_t n){
  int tb=256; size_t gg=(n+tb-1)/tb;
  k_add_into_u64<<<(int)gg,tb>>>(s->s_brick_fx+dst_off,s->s_brick_fx+src_off,n);
  CK(cudaGetLastError());
}
extern "C" void sog_gpu_brick_copy_planes(SogGpuState*s,size_t src_off,
    size_t dst_off,size_t n){
  CK(cudaMemcpy(s->d_brick+dst_off,s->d_brick+src_off,n*sizeof(float),
                cudaMemcpyDeviceToDevice));
}

// brick (owned planes) -> xy slab complex, then FFT(x,y), pack.  The packed
// blocks are LEFT IN s_pot so a device all-to-all can send them directly.
// The brick's owned planes hold the fixed-point (2^40) density; they are
// converted to float32 on the fly, exactly as the replicated solve converts
// f_rho_fx -> f_rho_r before its R2C.
extern "C" int sog_gpu_slab_xyfft_pack_brick_dev(SogGpuState*s,int nx,int ny,
    int nz,int ZL,int np,int zoff){
  if(s->s_rho==nullptr||s->s_pot==nullptr||s->plan_xy_rc==0) return -1;
  const int Nh=nx/2+1;
  const size_t nc=(size_t)Nh*ny*ZL;
  // R2C out of the owned brick planes (x fastest, ny rows, ZL batches).  zoff
  // is even in the 1-D slab path, but the full-brick path can land the slab at
  // an offset that is not 16-byte aligned; then stage it through the scratch.
  const size_t plane=(size_t)nx*ny;
  const size_t doff=(size_t)zoff*plane;
  const size_t nown=(size_t)ZL*plane;
  float *rin;
  if((doff%4)==0){
    rin=s->d_brick+doff;
  }else{
    rin=ensure_real_f(s,nown);
  }
  int tb=256; size_t gg=(nc+tb-1)/tb, gc=(nown+tb-1)/tb;
  k_fx_to_float<<<(int)gc,tb>>>(nown,s->s_brick_fx+doff,rin,1.0/kFxScale);
  if(cufftExecR2C(s->plan_xy_rc,rin,s->s_rho)!=CUFFT_SUCCESS) return -2;
  k_pack_xy_to_blocks<<<(int)gg,tb>>>(Nh,ny,nz,ZL,np,s->s_rho,s->s_pot);
  CK(cudaGetLastError());
  return 0;
}
extern "C" int sog_gpu_slab_xyfft_pack_brick(SogGpuState*s,int nx,int ny,
    int nz,int ZL,int np,int zoff,double*host_snd){
  const int rc=sog_gpu_slab_xyfft_pack_brick_dev(s,nx,ny,nz,ZL,np,zoff);
  if(rc!=0) return rc;
  const size_t nc=(size_t)(nx/2+1)*ny*ZL;
  CK(cudaMemcpy(host_snd,s->s_pot,2*nc*sizeof(float),cudaMemcpyDeviceToHost));
  return 0;
}

// blocks -> xy slab, inverse FFT(x,y), then slab -> brick owned planes; the
// ghost planes are zeroed so the forward halo starts from a clean (float)
// potential brick.  Device-resident variant: the received blocks are ALREADY
// in s_pot.
extern "C" int sog_gpu_slab_unpack_ixyfft_to_brick_dev(SogGpuState*s,int nx,
    int ny,int nz,int ZL,int np,int zoff,int ZB){
  if(s->s_rho==nullptr||s->s_pot==nullptr||s->plan_xy_cr==0) return -1;
  const int Nh=nx/2+1;
  const size_t nc=(size_t)Nh*ny*ZL;
  int tb=256; size_t gg=(nc+tb-1)/tb;
  k_unpack_blocks_to_xy<<<(int)gg,tb>>>(Nh,ny,ZL,np,s->s_pot,s->s_rho);
  CK(cudaGetLastError());
  const size_t plane=(size_t)nx*ny;
  const size_t doff=(size_t)zoff*plane;
  const size_t nown=(size_t)ZL*plane;
  float *rout;
  if((doff%4)==0){
    rout=s->d_brick+doff;
  }else{
    rout=ensure_real_f(s,nown);
  }
  if(cufftExecC2R(s->plan_xy_cr,s->s_rho,rout)!=CUFFT_SUCCESS) return -2;
  if((doff%4)!=0)
    CK(cudaMemcpy(s->d_brick+doff,s->s_real,nown*sizeof(float),
                  cudaMemcpyDeviceToDevice));
  if(zoff>0) sog_gpu_brick_zero(s,0,(size_t)zoff*plane);
  const size_t upper=(size_t)(ZB-zoff-ZL);
  if(upper>0) sog_gpu_brick_zero(s,(size_t)(zoff+ZL)*plane,upper*plane);
  return 0;
}
// Host-buffer wrapper (MPI fallback path).
extern "C" int sog_gpu_slab_unpack_ixyfft_to_brick(SogGpuState*s,int nx,int ny,
    int nz,int ZL,int np,int zoff,int ZB,const double*host_back){
  const size_t nc=(size_t)(nx/2+1)*ny*ZL;
  CK(cudaMemcpy(s->s_pot,host_back,2*nc*sizeof(float),cudaMemcpyHostToDevice));
  return sog_gpu_slab_unpack_ixyfft_to_brick_dev(s,nx,ny,nz,ZL,np,zoff,ZB);
}

// Free the grid-sized buffers (full or slab) and the 3D plan.
static void free_grid_buffers(SogGpuState*s){
  if(s->plan){cufftDestroy(s->plan);s->plan=0;}
  if(s->f_plan){cufftDestroy(s->f_plan);s->f_plan=0;}
  if(s->f_plan_r2c){cufftDestroy(s->f_plan_r2c);s->f_plan_r2c=0;}
  if(s->f_plan_c2r){cufftDestroy(s->f_plan_c2r);s->f_plan_c2r=0;}
  if(s->d_rho){cudaFree(s->d_rho);s->d_rho=nullptr;}
  if(s->d_gx){cudaFree(s->d_gx);s->d_gx=nullptr;}
  if(s->d_gy){cudaFree(s->d_gy);s->d_gy=nullptr;}
  if(s->d_gz){cudaFree(s->d_gz);s->d_gz=nullptr;}
  if(s->d_pot){cudaFree(s->d_pot);s->d_pot=nullptr;}
  if(s->d_ge){cudaFree(s->d_ge);s->d_ge=nullptr;}
  if(s->d_gf){cudaFree(s->d_gf);s->d_gf=nullptr;}
  if(s->d_gv){cudaFree(s->d_gv);s->d_gv=nullptr;}
  // The single-precision replicated set.  sog_gpu_setup re-allocates these with
  // a bare cudaMalloc on every rebuild (grid change under NPT, and every
  // slab<->replicated switch), so they MUST be released here or each rebuild
  // leaks ~48 B per grid point plus three plans -- 162 MB per switch at 150^3.
  if(s->f_rho){cudaFree(s->f_rho);s->f_rho=nullptr;}
  if(s->f_pot){cudaFree(s->f_pot);s->f_pot=nullptr;}
  if(s->f_gp){cudaFree(s->f_gp);s->f_gp=nullptr;}
  if(s->f_rho_r){cudaFree(s->f_rho_r);s->f_rho_r=nullptr;}
  if(s->f_rho_fx){cudaFree(s->f_rho_fx);s->f_rho_fx=nullptr;}
  if(s->f_pot_r){cudaFree(s->f_pot_r);s->f_pot_r=nullptr;}
  if(s->f_spec){cudaFree(s->f_spec);s->f_spec=nullptr;}
  if(s->f_gradx_r){cudaFree(s->f_gradx_r);s->f_gradx_r=nullptr;}
  if(s->f_grady_r){cudaFree(s->f_grady_r);s->f_grady_r=nullptr;}
  if(s->f_gradz_r){cudaFree(s->f_gradz_r);s->f_gradz_r=nullptr;}
  if(s->f_grad_spec){cudaFree(s->f_grad_spec);s->f_grad_spec=nullptr;}
  if(s->f_qgx){cudaFree(s->f_qgx);s->f_qgx=nullptr;}
  if(s->f_qgy){cudaFree(s->f_qgy);s->f_qgy=nullptr;}
  if(s->f_qgz){cudaFree(s->f_qgz);s->f_qgz=nullptr;}
  if(s->f_qgspec){cudaFree(s->f_qgspec);s->f_qgspec=nullptr;}
  if(s->f_fft_work){cudaFree(s->f_fft_work);s->f_fft_work=nullptr;}
  // The slab's own single-precision set (see the state struct): complex
  // spectra, the packed float2 green table and the R2C/C2R alignment scratch.
  if(s->s_rho){cudaFree(s->s_rho);s->s_rho=nullptr;}
  if(s->s_pot){cudaFree(s->s_pot);s->s_pot=nullptr;}
  if(s->s_gp){cudaFree(s->s_gp);s->s_gp=nullptr;}
  if(s->s_real){cudaFree(s->s_real);s->s_real=nullptr;s->cap_real=0;}
  s->slab_cap=0;s->slab_gn=0;s->slab_mode=false;s->slab_Y=0;
}

// Slab-sized setup for the distributed path.  s_rho / s_pot hold the largest
// slab this rank ever transforms, s_gp holds only this rank's y-slice of the
// green table (indexed kh + Nh*(iyl + YL*iz)) packed as (ge, gv) floats, and
// the gradient meshes d_gx/d_gy/d_gz are NOT allocated -- the distributed solve
// never uses them.  Nothing here creates the 3D plan; the slab plans are built
// by sog_gpu_slab_setup, which needs the shapes.
extern "C" void sog_gpu_setup_slab(SogGpuState*s,int nx,int ny,int nz,int YL,
    int ZL,const double*ge,const double*gvv,size_t gn){
  const int Nh=nx/2+1;
  const size_t cap=(size_t)((size_t)Nh*ny*ZL > (size_t)nz*Nh*YL
                                ? (size_t)Nh*ny*ZL : (size_t)nz*Nh*YL);
  const bool realloc = !s->slab_mode || s->nx!=nx || s->ny!=ny || s->nz!=nz ||
                       s->slab_cap!=cap || s->slab_gn!=gn || s->slab_Y!=YL;
  if(realloc){
    free_grid_buffers(s);
    CK(cudaMalloc(&s->s_rho,cap*sizeof(cufftComplex)));
    CK(cudaMalloc(&s->s_pot,cap*sizeof(cufftComplex)));
    CK(cudaMalloc(&s->s_gp,gn*sizeof(float2)));
    s->nx=nx;s->ny=ny;s->nz=nz;s->ngrid=(size_t)nx*ny*nz;
    s->slab_cap=cap;s->slab_gn=gn;s->slab_Y=YL;s->slab_mode=true;
  }
  // Pack (green energy, green virial) -- the pair the replicated f_gp holds.
  // The slab's gf == ge by construction (both are kfac*dc), so the third table
  // is dropped instead of uploaded.
  std::vector<float2> fgp(gn);
  for(size_t q=0;q<gn;++q) fgp[q]=make_float2((float)ge[q],(float)gvv[q]);
  CK(cudaMemcpy(s->s_gp,fgp.data(),gn*sizeof(float2),cudaMemcpyHostToDevice));
  if(getenv("SOG_DIST_DEBUG")){
    size_t fb=0,tb=0; cudaMemGetInfo(&fb,&tb);
    fprintf(stderr,"[sog-mem] slab grid buffers: scratch %.1f MB + green %.1f MB "
            "(full grid would be %.1f MB); device free %.1f MB\n",
            2.0*cap*8/1048576.0, 1.0*gn*8/1048576.0,
            (5.0*nx*ny*nz*16 + 3.0*nx*ny*nz*8)/1048576.0, fb/1048576.0);
    fflush(stderr);
  }
}

// `want_double` allocates the fp64 mesh set (d_rho/d_pot and the d_ge/gf/gv
// tables).  Nothing in the replicated or local-spread solve reads them -- those
// run entirely on the single-precision f_* set -- so a caller that never runs
// the fp64 solver can skip 56 B per grid point (53 MiB at 100^3, 400 MiB at
// 196^3).  The default entry point keeps them, so every existing caller is
// unaffected.
static void setup_replicated(SogGpuState*s,int nx,int ny,int nz,const double*ge,
    const double*gf,const double*gvv,int want_double){
  size_t ng=(size_t)nx*ny*nz;
  const int want = want_double ? 1 : 0;
  if(s->slab_mode || s->double_mesh!=want || nx!=s->nx||ny!=s->ny||nz!=s->nz){
    free_grid_buffers(s);
    if(want){
      CK(cudaMalloc(&s->d_rho,ng*sizeof(cufftDoubleComplex)));
      // The gradient meshes d_gx/d_gy/d_gz are not allocated: the gathers read
      // only the potential mesh and differentiate the interpolation weights, so
      // three complex grids (48 B per grid point) would be pure overhead.
      CK(cudaMalloc(&s->d_pot,ng*sizeof(cufftDoubleComplex)));
      CK(cudaMalloc(&s->d_ge,ng*sizeof(double)));
      CK(cudaMalloc(&s->d_gf,ng*sizeof(double)));
      CK(cudaMalloc(&s->d_gv,ng*sizeof(double)));
      // cuFFT 3D expects (nz,ny,nx) with x fastest -> pass dims as (nz,ny,nx)
      cufftPlan3d(&s->plan,nz,ny,nx,CUFFT_Z2Z);
    }
    s->double_mesh=want;
    // f_rho / f_pot (16 B per grid point, a third of this set) and the C2C plan
    // belong to the QuadS branch alone; the CubeS₂ kernel this operator always
    // selects runs on f_rho_fx / f_rho_r / f_spec / f_pot_r / f_gp.  They are
    // now allocated on first use by ensure_quads_mesh().
    CK(cudaMalloc(&s->f_gp,ng*sizeof(float2)));
    CK(cudaMalloc(&s->f_rho_r,ng*sizeof(float)));
    CK(cudaMalloc(&s->f_rho_fx,ng*sizeof(unsigned long long)));
    // f_pot_r (4 B per grid point) is not allocated here: the replicated
    // solve reuses f_rho_r for the potential, which is dead once the
    // forward R2C has consumed the density.  The one caller that can skip
    // the density rebuild (sog_gpu_solve_f with density_ready) allocates it
    // on demand through ensure_pot_r().
    cufftPlan3d(&s->f_plan_r2c,nz,ny,nx,CUFFT_R2C);
    cufftPlan3d(&s->f_plan_c2r,nz,ny,nx,CUFFT_C2R);
    {
      size_t w1=0,w2=0;
      cufftGetSize(s->f_plan_r2c,&w1);
      cufftGetSize(s->f_plan_c2r,&w2);
      const size_t wmax=w1>w2?w1:w2;
      const char *sw=std::getenv("SOG_FFT_SHARED_WORK");
      if(wmax>0 && !(sw && sw[0]=='0')){
        CK(cudaMalloc(&s->f_fft_work,wmax));
        cufftSetAutoAllocation(s->f_plan_r2c,0);
        cufftSetAutoAllocation(s->f_plan_c2r,0);
        cufftSetWorkArea(s->f_plan_r2c,s->f_fft_work);
        cufftSetWorkArea(s->f_plan_c2r,s->f_fft_work);
      }
    }
    s->nx=nx;s->ny=ny;s->nz=nz;s->ngrid=ng;
    if(std::getenv("SOG_MEM_BREAKDOWN")){
      size_t w_c2c=0,w_r2c=0,w_c2r=0,w_z2z=0;
      if(s->f_plan) cufftGetSize(s->f_plan,&w_c2c);
      cufftGetSize(s->f_plan_r2c,&w_r2c);
      cufftGetSize(s->f_plan_c2r,&w_c2r);
      if(s->plan) cufftGetSize(s->plan,&w_z2z);
      const double gb=1.0/1073741824.0;
      fprintf(stderr,"[sog-mem] grid %dx%dx%d = %.2f Mpts  want_double=%d\n",
              nx,ny,nz,(double)ng/1e6,want);
      fprintf(stderr,"[sog-mem]   mesh buffers GB: f_gp %.2f  f_rho_r %.2f"
             "  f_rho_fx %.2f (= mesh + half spectrum)  | sum %.2f B/pt=24\n",
             ng*8.0*gb, ng*4.0*gb, ng*8.0*gb, ng*24.0*gb);
      fprintf(stderr,"[sog-mem]   deferred: f_rho/f_pot (QuadS only) %.2f,"
             " f_pot_r %.2f, f_spec %.2f, C2C plan\n",
             ng*16.0*gb, ng*4.0*gb, ng*8.0*gb);
      if(want)
        fprintf(stderr,"[sog-mem]   fp64 set GB: d_rho %.2f  d_pot %.2f  d_ge/gf/gv %.2f\n",
                ng*16.0*gb, ng*16.0*gb, ng*24.0*gb);
      fprintf(stderr,"[sog-mem]   cuFFT work GB: C2C %.2f  R2C %.2f  C2R %.2f  Z2Z %.2f"
             "  | sum %.2f\n", w_c2c*gb, w_r2c*gb, w_c2r*gb, w_z2z*gb,
             (w_c2c+w_r2c+w_c2r+w_z2z)*gb);
      fflush(stderr);
    }
  }
  if(want){
    CK(cudaMemcpy(s->d_ge,ge,ng*sizeof(double),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s->d_gf,gf,ng*sizeof(double),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s->d_gv,gvv,ng*sizeof(double),cudaMemcpyHostToDevice));
  }
  // The packed float table the replicated solve reads (built once per rebuild).
  {
    std::vector<float2> fgp(ng);
    for(size_t q=0;q<ng;++q){fgp[q]=make_float2((float)ge[q],(float)gvv[q]);}
    CK(cudaMemcpy(s->f_gp,fgp.data(),ng*sizeof(float2),cudaMemcpyHostToDevice));
  }
}

extern "C" void sog_gpu_setup(SogGpuState*s,int nx,int ny,int nz,
    const double*ge,const double*gf,const double*gvv){
  setup_replicated(s,nx,ny,nz,ge,gf,gvv,/*want_double=*/1);
}

// Single-precision only: skips the fp64 mesh set.  Safe for any caller whose
// solve goes through sog_gpu_compute* or sog_gpu_solve_f (both read only f_*).
extern "C" void sog_gpu_setup_f32(SogGpuState*s,int nx,int ny,int nz,
    const double*ge,const double*gf,const double*gvv){
  setup_replicated(s,nx,ny,nz,ge,gf,gvv,/*want_double=*/0);
}

// v_i self-energy correction, on the device:
//   vpot[i] -= qscale * q[i] * (rsi + sc)
__global__ void k_vpot_self(int n,double*vpot,const double*q,double qscale,double rsi,double sc){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) vpot[i]-=qscale*q[i]*(rsi+sc);
}

// Device-resident core: every per-atom array (x, q in; force, vpot out) already
// lives on the GPU, so a runner that keeps coordinates/charges/forces in device
// arrays (LAMMPS Kokkos views) pays no host round trip.  ``sog_gpu_compute``
// below is the host-buffer wrapper around it.
// ── Form A: the exact assignment-window symbol, measured on the device ──────
// The mesh influence function must invert the assignment window.  Form B
// approximates that inverse by a Gaussian variance subtraction,
// exp(xi^2 Delta^2 k^2); Form A divides by the true <|W(k)|^2>.
//
// W is theta-dependent (a non-separable window is a function of the distance
// to the nodes, so the symbol carries the sub-cell position), and what a
// single global deconvolution can undo is the incoherent average <|W|^2>,
// i.e. the average over charge positions of the one-charge spectrum.  A
// coherent average <W> cancels and is meaningless here.
//
// So: spread one unit charge at ntheta^3 sub-cell positions, transform, and
// accumulate |W|^2.  Uses the same spread kernel and cuFFT plan as the solve,
// so no separate implementation of the weights is needed.  Once per box.
__global__ void k_acc_spec_abs2(size_t n, const cufftComplex* spec, double* acc){
  size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n){
    const double re=(double)spec[i].x, im=(double)spec[i].y;
    acc[i]+=re*re+im*im;
  }
}
// Divide the packed green table (ge, gv) by <|W|^2>.  f_gp is indexed with a
// full-nx stride (kh + nx*(iy + ny*iz)); the accumulation buffer is the
// contiguous half spectrum cuFFT writes (kh + nh*(iy + ny*iz)).
__global__ void k_gp_divide_form_a(int nh,int nx,int ny,int nz,float2*gp,
    const double*w2,double inv_theta,double rel_floor,
    double xi,double dx,double dy,double dz,double kfx,double kfy,double kfz){
  size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  const size_t ntot=(size_t)nh*ny*nz;
  if(i>=ntot) return;
  const int kh=(int)(i%nh);
  const size_t r=i/(size_t)nh;
  const int iy=(int)(r%ny), iz=(int)(r/ny);
  const size_t gi=(size_t)kh+(size_t)nx*((size_t)iy+(size_t)ny*(size_t)iz);
  const double w=w2[i]*inv_theta;
  // A window near-zero at the Nyquist corner would blow the division up.  Those
  // modes carry almost no kernel weight; drop them exactly like the B-spline
  // path drops its Nyquist planes rather than amplify noise.
  if(!(w>rel_floor)){ gp[gi].x=0.f; gp[gi].y=0.f; return; }
  // The incoming table is Form B: ge_B = kfac * D with
  // D = exp(xi^2 (dx^2 kx^2 + dy^2 ky^2 + dz^2 kz^2)) already standing in for
  // 1/|W|^2.  The exact table is kfac/|W|^2 = ge_B / (D * <|W|^2>), so BOTH
  // factors must be divided out -- dividing by <|W|^2> alone applies the
  // deconvolution twice (measured: the reciprocal energy came out 1.7x too
  // large and the total energy changed sign).
  const int my=iy-ny*(2*iy/ny), mz=iz-nz*(2*iz/nz);
  const double kx=kfx*(double)kh, ky=kfy*(double)my, kz=kfz*(double)mz;
  const double D=std::exp(xi*xi*(dx*dx*kx*kx+dy*dy*ky*ky+dz*dz*kz*kz));
  const double f=w*D;
  gp[gi].x=(float)((double)gp[gi].x/f);
  gp[gi].y=(float)((double)gp[gi].y/f);
}

// Convert the green table in place from Form B to Form A.  `nord` is the
// CubeS2 order (4 or 6); `nth` the number of sub-cell samples per axis.
extern "C" int sog_gpu_green_form_a(SogGpuState*s,int nord,int nth,double lx,
    double ly,double lz){
  if(s->f_gp==nullptr||s->f_plan_r2c==0) return -1;
  // f_rho_r / f_spec may be allocated lazily by the solve; this runs first.
  if(!s->f_rho_r) CK(cudaMalloc(&s->f_rho_r,s->ngrid*sizeof(float)));
  if(!s->f_spec)  CK(cudaMalloc(&s->f_spec,s->ngrid*sizeof(cufftComplex)));
  const int nx=s->nx, ny=s->ny, nz=s->nz, nh=nx/2+1;
  const size_t nspec=(size_t)nh*ny*nz;
  const size_t ng=s->ngrid;
  const double xi=(nord==6)?LAMMPS_NS::kCubes2Xi6:LAMMPS_NS::kCubes2Xi4;
  const int nn=(nord==6)?LAMMPS_NS::kCubes2NumNodes6:LAMMPS_NS::kCubes2NumNodes4;
  const LAMMPS_NS::CubeS2Node4* nodes=(nord==6)?LAMMPS_NS::kCubes2Nodes6:LAMMPS_NS::kCubes2Nodes4;

  double* acc=nullptr;
  if(cudaMalloc(&acc,nspec*sizeof(double))!=cudaSuccess) return -2;
  CK(cudaMemset(acc,0,nspec*sizeof(double)));
  std::vector<float> rho(ng);
  const int tb=256;
  for(int a=0;a<nth;++a){
    for(int b=0;b<nth;++b){
      for(int c=0;c<nth;++c){
        const double tx=(a+0.5)/nth, ty=(b+0.5)/nth, tz=(c+0.5)/nth;
        std::fill(rho.begin(),rho.end(),0.0f);
        // One unit charge sits in cell (0,0,0); the assignment is periodic, so
        // the node offsets are taken mod n.  The sub-cell position supplies the
        // theta that the symbol depends on for a non-separable window.
        for(int d=0;d<nn;++d){
          const LAMMPS_NS::CubeS2Node4& nd=nodes[d];
          const double w=(nord==6)?cubes2_weight_6(tx,ty,tz,nd,xi)
                                  :cubes2_weight_4(tx,ty,tz,nd,xi);
          if(w==0.0) continue;
          const int ix=((nd.dx%nx)+nx)%nx;
          const int iy=((nd.dy%ny)+ny)%ny;
          const int iz=((nd.dz%nz)+nz)%nz;
          rho[(size_t)ix+(size_t)nx*((size_t)iy+(size_t)ny*(size_t)iz)] +=
              (float)w;
        }
        CK(cudaMemcpy(s->f_rho_r,rho.data(),ng*sizeof(float),
                      cudaMemcpyHostToDevice));
        {
          const cufftResult cr=cufftExecR2C(s->f_plan_r2c,s->f_rho_r,s->f_spec);
          if(cr!=CUFFT_SUCCESS){ cudaFree(acc); return -1000+(int)cr; }
        }
        k_acc_spec_abs2<<<(int)((nspec+tb-1)/tb),tb>>>(nspec,s->f_spec,acc);
        {
          const cudaError_t ce=cudaGetLastError();
          if(ce!=cudaSuccess){ cudaFree(acc); return -2000+(int)ce; }
        }
      }
    }
  }
    const double inv_theta=1.0/((double)nth*nth*nth);
  // <|W|^2> <= 1 (it is the mean of |W|^2 over a unit charge), so the floor is
  // inherently RELATIVE to the k=0 value.  An absolute floor lets the Nyquist
  // corner, where the window is ~1e-8, be amplified by 1e8 -- measured symptom:
  // the Form-A error GREW from 6e-3 to 1.1 as the grid refined.
  const char* fl = std::getenv("SOG_MIDTOWN_FORM_A_FLOOR");
  const double floor_rel = fl ? std::atof(fl) : 1.0e-4;
      k_gp_divide_form_a<<<(int)((nspec+tb-1)/tb),tb>>>(nh,nx,ny,nz,s->f_gp,acc,
      inv_theta,floor_rel,xi,lx/nx,ly/ny,lz/nz,
      2.0*M_PI/lx,2.0*M_PI/ly,2.0*M_PI/lz);
  CK(cudaGetLastError());
  cudaFree(acc);
  return 0;
}

// Measure the CubeS2 assignment-window symbol <|W(k)|^2> on the full grid, into
// a host buffer of the compact half spectrum (kh + nh*(iy + ny*iz), nh = nx/2+1).
// Same measurement as the replicated Form-A conversion (spread one unit charge
// at nth^3 sub-cell positions, transform with a full-grid R2C plan, accumulate
// |W|^2), but self-contained: it allocates its own full-grid plan and buffers
// so it runs while the state is in slab mode (whose f_rho_r / f_spec /
// f_plan_r2c are slab-sized or absent).
//
// The window is a global property of the spline and the grid -- identical on
// every rank -- so the z-slab solve measures it redundantly per rank and feeds
// it to build_green_slab_half, which divides that rank's Form-B slice by
// D * <|W|^2> to obtain Form A.  Returns 0 on success, a negative code on error.
extern "C" int sog_gpu_measure_window(int nx,int ny,int nz,int nord,int nth,
    double lx,double ly,double lz,double*out){
  const size_t ng=(size_t)nx*ny*nz;
  const int nh=nx/2+1;
  const size_t nspec=(size_t)nh*ny*nz;
  const double xi=(nord==6)?LAMMPS_NS::kCubes2Xi6:LAMMPS_NS::kCubes2Xi4;
  const int nn=(nord==6)?LAMMPS_NS::kCubes2NumNodes6:LAMMPS_NS::kCubes2NumNodes4;
  const LAMMPS_NS::CubeS2Node4* nodes=(nord==6)?LAMMPS_NS::kCubes2Nodes6:LAMMPS_NS::kCubes2Nodes4;
  (void)lx;(void)ly;(void)lz;
  float* rho=nullptr; cufftComplex* spec=nullptr; cufftHandle plan=0;
  double* acc=nullptr;
  if(cudaMalloc((void**)&rho,ng*sizeof(float))!=cudaSuccess) return -1;
  if(cudaMalloc((void**)&spec,nspec*sizeof(cufftComplex))!=cudaSuccess){
    cudaFree(rho); return -1;}
  if(cudaMalloc((void**)&acc,nspec*sizeof(double))!=cudaSuccess){
    cudaFree(rho); cudaFree(spec); return -1;}
  if(cufftPlan3d(&plan,nz,ny,nx,CUFFT_R2C)!=CUFFT_SUCCESS){
    cudaFree(rho); cudaFree(spec); cudaFree(acc); return -2;}
  CK(cudaMemset(acc,0,nspec*sizeof(double)));
  std::vector<float> h_rho(ng);
  const int tb=256;
  bool ok=true;
  for(int a=0;a<nth&&ok;++a){
    for(int b=0;b<nth&&ok;++b){
      for(int c=0;c<nth&&ok;++c){
        const double tx=(a+0.5)/nth, ty=(b+0.5)/nth, tz=(c+0.5)/nth;
        std::fill(h_rho.begin(),h_rho.end(),0.0f);
        for(int d=0;d<nn;++d){
          const LAMMPS_NS::CubeS2Node4& nd=nodes[d];
          const double w=(nord==6)?cubes2_weight_6(tx,ty,tz,nd,xi)
                                  :cubes2_weight_4(tx,ty,tz,nd,xi);
          if(w==0.0) continue;
          const int ix=((nd.dx%nx)+nx)%nx;
          const int iy=((nd.dy%ny)+ny)%ny;
          const int iz=((nd.dz%nz)+nz)%nz;
          h_rho[(size_t)ix+(size_t)nx*((size_t)iy+(size_t)ny*(size_t)iz)] +=
              (float)w;
        }
        CK(cudaMemcpy(rho,h_rho.data(),ng*sizeof(float),cudaMemcpyHostToDevice));
        if(cufftExecR2C(plan,rho,spec)!=CUFFT_SUCCESS){ ok=false; break; }
        k_acc_spec_abs2<<<(int)((nspec+tb-1)/tb),tb>>>(nspec,spec,acc);
        if(cudaGetLastError()!=cudaSuccess){ ok=false; break; }
      }
    }
  }
  if(ok){
    std::vector<double> h_acc(nspec);
    CK(cudaMemcpy(h_acc.data(),acc,nspec*sizeof(double),cudaMemcpyDeviceToHost));
    const double inv_theta=1.0/((double)nth*nth*nth);
    for(size_t i=0;i<nspec;++i) out[i]=h_acc[i]*inv_theta;
  }
  cudaFree(rho); cudaFree(spec); cudaFree(acc); cufftDestroy(plan);
  return ok?0:-3;
}

// QuadS (separable, full-complex C2C) is the only consumer of the f_rho/f_pot
// meshes and the C2C plan.  The CubeS₂ path this operator selects never touches
// them, so they are allocated here instead of in setup: that removes 16 B per
// grid point (a third of the single-precision set) from every CubeS₂ run.
static void ensure_quads_mesh(SogGpuState*s){
  const size_t ng=s->ngrid;
  if(s->f_rho && s->f_pot && s->f_plan) return;
  if(!s->f_rho) CK(cudaMalloc(&s->f_rho,ng*sizeof(cufftComplex)));
  if(!s->f_pot) CK(cudaMalloc(&s->f_pot,ng*sizeof(cufftComplex)));
  if(!s->f_plan) cufftPlan3d(&s->f_plan,s->nz,s->ny,s->nx,CUFFT_C2C);
}

// QuadS ik-mode scratch: three full-complex gradient meshes and the spectrum
// they are transformed from.  Allocated on demand, once per grid.
static void ensure_quads_ik(SogGpuState*s){
  const size_t ng=s->ngrid;
  if(!s->f_qgx) CK(cudaMalloc(&s->f_qgx,ng*sizeof(cufftComplex)));
  if(!s->f_qgy) CK(cudaMalloc(&s->f_qgy,ng*sizeof(cufftComplex)));
  if(!s->f_qgz) CK(cudaMalloc(&s->f_qgz,ng*sizeof(cufftComplex)));
  if(!s->f_qgspec) CK(cudaMalloc(&s->f_qgspec,ng*sizeof(cufftComplex)));
}

// Allocated on demand by sog_gpu_solve_f, which may skip the density rebuild
// between calls and therefore cannot share f_rho_r with the potential.
// The half spectrum (8 B per grid point) is borrowed from the fixed-point
// charge mesh: the density has already been converted to f_rho_r before the
// forward transform, and the mesh is rebuilt from scratch by the next spread.
// Only sog_gpu_solve_f, which may skip that rebuild, needs a buffer of its own.
static void ensure_spec(SogGpuState*s){
  if(!s->f_spec) CK(cudaMalloc(&s->f_spec,s->ngrid*sizeof(cufftComplex)));
}

static void ensure_pot_r(SogGpuState*s){
  if(!s->f_pot_r) CK(cudaMalloc(&s->f_pot_r,s->ngrid*sizeof(float)));
}

// The ik-mode gradient meshes and their shared half-spectrum scratch.  Three
// extra float grids (12 B per grid point together) plus one half spectrum.
static void ensure_ik_grads(SogGpuState*s){
  if(!s->f_gradx_r) CK(cudaMalloc(&s->f_gradx_r,s->ngrid*sizeof(float)));
  if(!s->f_grady_r) CK(cudaMalloc(&s->f_grady_r,s->ngrid*sizeof(float)));
  if(!s->f_gradz_r) CK(cudaMalloc(&s->f_gradz_r,s->ngrid*sizeof(float)));
  const size_t nsp=(size_t)(s->nx/2+1)*s->ny*s->nz;
  if(!s->f_grad_spec) CK(cudaMalloc(&s->f_grad_spec,nsp*sizeof(cufftComplex)));
}

static double sog_gpu_compute_core(SogGpuState*s,int nlocal,const double*d_x,const double*d_q,
    const double*boxlo,double lx,double ly,double lz,double qscale,int spline,int want_pot,
    double*d_force,double*d_vpot,double self_coeff,double qsqsum,int remove_self,int force_mode,
    double*virial6,int host_reduction){
  size_t ng=s->ngrid;
  (void)nlocal;
  double boxlo3[3]={boxlo[0],boxlo[1],boxlo[2]};
  const bool new_boxlo = (s->d_boxlo==nullptr);
  if(new_boxlo) CK(cudaMalloc(&s->d_boxlo,3*sizeof(double)));
  double *d_boxlo=s->d_boxlo;
  if(s->d_red==nullptr) CK(cudaMalloc(&s->d_red,20*sizeof(double)));
  // A zero origin is the common case: the upload only travels when the origin
  // actually moves (or on the first call), not on every solve.
  if(new_boxlo || boxlo[0]!=0.0 || boxlo[1]!=0.0 || boxlo[2]!=0.0)
    CK(cudaMemcpy(d_boxlo,boxlo3,3*sizeof(double),cudaMemcpyHostToDevice));
  CK(cudaMemset(s->d_red,0,20*sizeof(double)));
  double volume=s->metric?s->volume:lx*ly*lz, rho_scale=(double)ng/volume, scaleinv=1.0/(double)ng;
  // 2*pi/L on the host: three fp64 divisions per mode are the single largest
  // cost of the k-space pass on a 1/64-rate fp64 pipe.
  const double kfx=2*M_PI/lx, kfy=2*M_PI/ly, kfz=2*M_PI/lz;

  int tb=256, gg=(ng+tb-1)/tb, ga=(nlocal+tb-1)/tb;
  // Multiprocessors of this device, for the occupancy of the k-space pass.
  static int multiprocessors = [] {
    int device = 0;
    cudaGetDevice(&device);
    int count = 0;
    cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device);
    return count > 0 ? count : 32;
  }();
  // The phase timers cost five synchronous event waits per call, so they are
  // armed only when the caller asks for them (SOG_GPU_TIMING).
  const bool timing = (std::getenv("SOG_GPU_TIMING") != nullptr);
  // ── Create timing events lazily ──
  if (timing && !s->evt_spread_start) {
    cudaEventCreate(&s->evt_spread_start); cudaEventCreate(&s->evt_spread_stop);
    cudaEventCreate(&s->evt_fftfwd_start); cudaEventCreate(&s->evt_fftfwd_stop);
    cudaEventCreate(&s->evt_kspace_start); cudaEventCreate(&s->evt_kspace_stop);
    cudaEventCreate(&s->evt_fftinv_start); cudaEventCreate(&s->evt_fftinv_stop);
    cudaEventCreate(&s->evt_gather_start); cudaEventCreate(&s->evt_gather_stop);
  }
  bool quads = (spline==SOG_QUADS_4||spline==SOG_QUADS_6);
  int nord = (spline==SOG_CUBES2_6||spline==SOG_QUADS_6)?6:4;
  const int gw=(nlocal+(tb/32)-1)/(tb/32);
  if(quads){
    // Real density and half spectrum, like the B-spline / CubeS2 core: a warp
    // spreads and gathers the (2nu)^3 separable stencil, R2C/C2R carry the
    // mesh.  This halves the transform work and the density/potential buffers
    // relative to the original full-complex path.
    CK(cudaMemsetAsync(s->f_rho_r,0,ng*sizeof(float),0));
    if(nord==4) k_spread_quads_warp_r<4><<<gw,tb>>>(nlocal,d_x,d_boxlo,lx,ly,lz,d_q,rho_scale,s->nx,s->ny,s->nz,s->f_rho_r);
    else        k_spread_quads_warp_r<6><<<gw,tb>>>(nlocal,d_x,d_boxlo,lx,ly,lz,d_q,rho_scale,s->nx,s->ny,s->nz,s->f_rho_r);
    ensure_spec(s);
    cufftExecR2C(s->f_plan_r2c,s->f_rho_r,s->f_spec);
    {
      const int nh=s->nx/2+1;
      const int npw=(nh+2)/3;
      const int gx=(npw*s->ny+63)/64;
      int gz=(16*multiprocessors + gx - 1)/gx;
      if(gz<1) gz=1; if(gz>s->nz) gz=s->nz;
      k_kspace_h<<<dim3(gx,gz),64>>>(s->nx,s->ny,s->nz,kfx,kfy,kfz,s->d_hinv,
                                     s->metric,s->f_gp,s->f_spec,want_pot,
                                     scaleinv,s->d_red);
    }
    if(force_mode==1){
      // Fourier differentiation on the half spectrum: three C2R gradients of
      // i*k_a*phi_hat, then a value-only warp gather.
      ensure_ik_grads(s);
      k_mul_ik<<<gg,tb>>>(s->nx,s->ny,s->nz,kfx,kfy,kfz,s->f_spec,s->f_grad_spec,0);
      cufftExecC2R(s->f_plan_c2r,s->f_grad_spec,s->f_gradx_r);
      k_mul_ik<<<gg,tb>>>(s->nx,s->ny,s->nz,kfx,kfy,kfz,s->f_spec,s->f_grad_spec,1);
      cufftExecC2R(s->f_plan_c2r,s->f_grad_spec,s->f_grady_r);
      k_mul_ik<<<gg,tb>>>(s->nx,s->ny,s->nz,kfx,kfy,kfz,s->f_spec,s->f_grad_spec,2);
      cufftExecC2R(s->f_plan_c2r,s->f_grad_spec,s->f_gradz_r);
      cufftExecC2R(s->f_plan_c2r,s->f_spec,s->f_rho_r);
      if(nord==4) k_gather_quads_warp_r_ik<4><<<gw,tb>>>(nlocal,d_x,d_boxlo,lx,ly,lz,d_q,qscale,s->nx,s->ny,s->nz,s->f_rho_r,s->f_gradx_r,s->f_grady_r,s->f_gradz_r,want_pot,d_force,d_vpot);
      else        k_gather_quads_warp_r_ik<6><<<gw,tb>>>(nlocal,d_x,d_boxlo,lx,ly,lz,d_q,qscale,s->nx,s->ny,s->nz,s->f_rho_r,s->f_gradx_r,s->f_grady_r,s->f_gradz_r,want_pot,d_force,d_vpot);
    } else {
      cufftExecC2R(s->f_plan_c2r,s->f_spec,s->f_rho_r);
      if(nord==4) k_gather_quads_warp_r<4><<<gw,tb>>>(nlocal,d_x,d_boxlo,lx,ly,lz,d_q,qscale,s->nx,s->ny,s->nz,s->f_rho_r,want_pot,d_force,d_vpot,s->d_red);
      else        k_gather_quads_warp_r<6><<<gw,tb>>>(nlocal,d_x,d_boxlo,lx,ly,lz,d_q,qscale,s->nx,s->ny,s->nz,s->f_rho_r,want_pot,d_force,d_vpot,s->d_red);
    }
  } else {
    // CubeS2: real meshes and a half spectrum.
    // The fractional coordinate of every atom, once, one thread per atom.
    FracNode* frac = ensure_frac(s,nlocal);
    if(nlocal>0)
      k_midtown_frac<<<ga,tb>>>(nlocal,d_x,d_boxlo,1.0/lx,1.0/ly,1.0/lz,
                                s->nx,s->ny,s->nz,frac);
    k_zero_fx<<<gg,tb>>>(ng,s->f_rho_fx);
    if(nord==4) k_spread_cubes2_warp_fx_tab4<<<gw,tb>>>(nlocal,frac,d_q,rho_scale,s->nx,s->ny,s->nz,s->f_rho_fx);
    else {
      const int gwp=(nlocal+kSpreadWarps*kSpreadPack-1)/(kSpreadWarps*kSpreadPack);
      k_spread_cubes2_warp_fx_packed<6><<<gwp,tb>>>(nlocal,frac,d_q,rho_scale,s->nx,s->ny,s->nz,s->f_rho_fx);
    }
    k_fx_to_float<<<gg,tb>>>(ng,s->f_rho_fx,s->f_rho_r,1.0/kFxScale);
    cufftComplex *spec = reinterpret_cast<cufftComplex*>(s->f_rho_fx);
    cufftExecR2C(s->f_plan_r2c,s->f_rho_r,spec);
    {
      const int nh=s->nx/2+1;
      const int npw=(nh+2)/3;
      const int gx=(npw*s->ny+63)/64;
      // The pass is fp32-accumulate over the modes of a column and fp64 only
      // for the three wave numbers of a metric cell; with a handful of blocks
      // (the old 240-block target gave gz == 1 at every production mesh) each
      // thread walks the whole z column and the machine sits idle.  Spread the
      // work over the z axis so the launch fills the device instead.
      int gz=(16*multiprocessors + gx - 1)/gx;
      if(gz<1) gz=1; if(gz>s->nz) gz=s->nz;
      k_kspace_h<<<dim3(gx,gz),64>>>(s->nx,s->ny,s->nz,kfx,kfy,kfz,s->d_hinv,
                                     s->metric,s->f_gp,spec,want_pot,
                                     scaleinv,s->d_red);
    }
    // f_rho_r is free here (the density went into f_spec), so the potential
    // mesh replaces it instead of taking a buffer of its own.
    if(force_mode==1){
      // Fourier differentiation: three inverse transforms of i·k_a·φ̂ into the
      // gradient meshes, then a value-only gather.  CubeS2 only (QuadS keeps the
      // AD path); the caller rejects ik on the bspline backend.
      //
      // The three transforms run BEFORE the potential's C2R below: `spec`
      // aliases f_rho_fx, the buffer the potential's own inverse reads and
      // may reuse, so the gradient spectra must be taken off the intact
      // potential spectrum first.
      ensure_ik_grads(s);
      k_mul_ik<<<gg,tb>>>(s->nx,s->ny,s->nz,kfx,kfy,kfz,spec,s->f_grad_spec,0);
      cufftExecC2R(s->f_plan_c2r,s->f_grad_spec,s->f_gradx_r);
      k_mul_ik<<<gg,tb>>>(s->nx,s->ny,s->nz,kfx,kfy,kfz,spec,s->f_grad_spec,1);
      cufftExecC2R(s->f_plan_c2r,s->f_grad_spec,s->f_grady_r);
      k_mul_ik<<<gg,tb>>>(s->nx,s->ny,s->nz,kfx,kfy,kfz,spec,s->f_grad_spec,2);
      cufftExecC2R(s->f_plan_c2r,s->f_grad_spec,s->f_gradz_r);
      cufftExecC2R(s->f_plan_c2r,spec,s->f_rho_r);
      if(nord==4) k_gather_cubes2_warp_r_ik<4><<<gw,tb>>>(nlocal,frac,d_q,qscale,s->nx,s->ny,s->nz,s->f_rho_r,s->f_gradx_r,s->f_grady_r,s->f_gradz_r,want_pot,d_force,d_vpot);
      else        k_gather_cubes2_warp_r_ik<6><<<gw,tb>>>(nlocal,frac,d_q,qscale,s->nx,s->ny,s->nz,s->f_rho_r,s->f_gradx_r,s->f_grady_r,s->f_gradz_r,want_pot,d_force,d_vpot);
    } else {
      cufftExecC2R(s->f_plan_c2r,spec,s->f_rho_r);
      if(nord==4) k_gather_cubes2_warp_r<4><<<gw,tb>>>(nlocal,frac,1.0/lx,1.0/ly,1.0/lz,d_q,qscale,s->nx,s->ny,s->nz,s->f_rho_r,want_pot,d_force,d_vpot,s->d_red);
      else        k_gather_cubes2_warp_r<6><<<gw,tb>>>(nlocal,frac,1.0/lx,1.0/ly,1.0/lz,d_q,qscale,s->nx,s->ny,s->nz,s->f_rho_r,want_pot,d_force,d_vpot,s->d_red);
    }
  }
  if (timing) cudaEventRecord(s->evt_gather_stop, 0);
  // ── Accumulate timing ──
  if (timing) { float ms;
    cudaEventSynchronize(s->evt_spread_stop); cudaEventElapsedTime(&ms,s->evt_spread_start,s->evt_spread_stop); s->acc_spread += ms;
    cudaEventSynchronize(s->evt_fftfwd_stop); cudaEventElapsedTime(&ms,s->evt_fftfwd_start,s->evt_fftfwd_stop); s->acc_fftfwd += ms;
    cudaEventSynchronize(s->evt_kspace_stop); cudaEventElapsedTime(&ms,s->evt_kspace_start,s->evt_kspace_stop); s->acc_kspace += ms;
    cudaEventSynchronize(s->evt_fftinv_stop); cudaEventElapsedTime(&ms,s->evt_fftinv_start,s->evt_fftinv_stop); s->acc_fftinv += ms;
    cudaEventSynchronize(s->evt_gather_stop); cudaEventElapsedTime(&ms,s->evt_gather_start,s->evt_gather_stop); s->acc_gather += ms;
  }
  if (timing) ++s->timing_count;
  if (timing && s->timing_count >= s->timing_interval && s->timing_count > 0) {
    if (getenv("SOG_GPU_TIMING")) {
      int n = s->timing_count;
      printf("SOG_GPU_TIMING (n=%d): spread=%.3f fft_fwd=%.3f kspace=%.3f fft_inv=%.3f gather=%.3f ms/step  grid=%dx%dx%d=%zu spline=%s\n",
             n, s->acc_spread/n, s->acc_fftfwd/n, s->acc_kspace/n, s->acc_fftinv/n, s->acc_gather/n,
             s->nx, s->ny, s->nz, ng, quads ? (nord==6?"QuadS-6":"QuadS-4") : (nord==6?"CubeS2-6":"CubeS2-4"));
    }
    s->acc_spread=s->acc_fftfwd=s->acc_kspace=s->acc_fftinv=s->acc_gather=0;
    s->timing_count=0;
  }

  // A caller that reduces on the device skips the blocking copy: the 20 sums
  // stay in s->d_red and are read there (sog_gpu_red_device_ptr).
  if (!host_reduction) {
    return 0.0;
  }
  double red[20]; CK(cudaMemcpy(red,s->d_red,20*sizeof(double),cudaMemcpyDeviceToHost));
  // ── Self-energy: closed-form in real space, NO mesh sum ──────────────────
  // The k-space energy is the Fourier form of a PAIR potential.  Summing over
  // the complete reciprocal lattice reproduces the pair sum exactly, and its
  // (i == j, R == 0) term is ½·qsum_sq·g(0), with g(0) the multi-Gaussian
  // real-space kernel at zero separation:
  //     g(0) = sum_m A_m,   A_m = amp_m (2 pi bw_m)^(-3/2).
  // So "remove self interaction" is exactly ½·qsum_sq·g(0) -- a real-space
  // quantity, independent of any k-truncation, and exactly proportional to q^2.
  // Algebraically this equals the RBSOG closed form that sog.cpp already builds
  // for internally-generated kernels (`log(b)/(sqrt(2 pi) sigma)(w0 + sum b^-m)`),
  // verified to all 15 digits.  Callers therefore pass it in `self_coeff`; it
  // must NOT be zeroed for externally-supplied amp/bandwidth (the DeepMD path).
  //
  // The old mesh sum was NOT extensive (9.03x / 76.5x instead of 8x / 64x for
  // 2x / 4x replication): several bands are extremely narrow in k (bw up to
  // 2.2e5) and need a k-spacing far below 2 pi / L to resolve, so a finite
  // mesh under-counts the i == j term by a box-dependent amount.
  if(want_pot){
    // v_i = dE/dq_i, applied on the device so the potential never leaves the
    // GPU between the gather and the consumer:
    //   vpot[i] -= qscale*q_i*(rsi + sc),  rsi = g(0),  sc = 0
    // (rsi is exactly 2*self_coeff because self_coeff = ½ g(0)).
    const double rsi = remove_self ? 2.0*self_coeff : 0.0;
    if(nlocal>0) k_vpot_self<<<ga,tb>>>(nlocal,d_vpot,d_q,qscale,rsi,0.0);
  }
  // Energy = qscale * [ 0.5*V*Σ(s2·ge·|ρ|²)  −  remove_self·self_coeff·qsqsum ]
  double e_kneq0 = 0.5*volume*red[0];
  double e_self  = remove_self ? self_coeff*qsqsum : 0.0;
  double energy = qscale*(e_kneq0 - e_self);
  // ── Analytic Fourier virial: the COMPLETE strain-derivative of E_k ──
  // W_αβ = ½·V·qscale · Σ s²·|ρ̂(k)|²·(G_E·δ_αβ − K_v·k_α·k_β)
  // where G_E = Σ a_m e^{−½β_m k²}, K_v = Σ a_m·β_m e^{−½β_m k²}.
  // The strain-derivative of |ρ̂(k)|² is identically ZERO under fractional-
  // coordinate charge spreading + Form-B deconvolution. The r⊗F approach
  // (red[14..19]) is INCOMPLETE — it omits the box-explicit strain term.
  // This matches fastsog.cpp, PPPM vg, and the CPU sog.cpp analytic path.
  // red[1..6] = fv_local; red[14..19] = r⊗F (retained for diagnostics).
  for(int j=0;j<6;++j) virial6[j]=0.5*volume*qscale*red[1+j];
  // No self-energy virial.  The exact self term is qsum_sq * sum_m A_m / 2,
  // independent of V and of the cell, so its strain derivative is identically
  // zero.  (Analytically sum_mesh K_v k_a k_b = diag_sum * delta_ab; the old
  // mesh version was anisotropic -- 0.827/1.085/1.339 x the diagonal at a
  // 264-atom cell -- and injected a spurious shear that this removal deletes.)
  (void)remove_self;
  (void)quads;
  return energy;
}

static void ensure_local_atoms(SogGpuState*s,int nlocal){
  if(nlocal<=s->cap_atoms) return;
  cudaFree(s->d_x);cudaFree(s->d_q);cudaFree(s->d_force);cudaFree(s->d_vpot);
  CK(cudaMalloc(&s->d_x,3*(size_t)nlocal*sizeof(double)));
  CK(cudaMalloc(&s->d_q,(size_t)nlocal*sizeof(double)));
  CK(cudaMalloc(&s->d_force,3*(size_t)nlocal*sizeof(double)));
  CK(cudaMalloc(&s->d_vpot,(size_t)nlocal*sizeof(double)));
  s->cap_atoms=nlocal;
}

static void ensure_local_brick(SogGpuState*s,size_t nbrick){
  if(nbrick<=s->cap_brick) return;
  cudaFree(s->d_brick);
  cudaFree(s->s_brick_fx);
  CK(cudaMalloc(&s->d_brick,nbrick*sizeof(float)));
  CK(cudaMalloc(&s->s_brick_fx,nbrick*sizeof(unsigned long long)));
  s->cap_brick=nbrick;
}

// Spread the local atoms into the device FIXED-POINT brick.  No download: the
// distributed solve keeps working on s_brick_fx and only fetches the ghost
// planes; the float potential brick (d_brick) is written by the inverse FFT.
extern "C" void sog_gpu_spread_brick_dev(SogGpuState*s,int nlocal,
    const double*x,const double*boxlo,double lx,double ly,double lz,
    const double*q,double rho_scale,int spline,int nx,int ny,int nz,
    int nxlo,int nxhi,int nylo,int nyhi,int nzlo,int nzhi){
  size_t nb=(size_t)(nxhi-nxlo+1)*(nyhi-nylo+1)*(nzhi-nzlo+1);
  ensure_local_brick(s,nb);
  CK(cudaMemset(s->s_brick_fx,0,nb*sizeof(unsigned long long)));
  if(nlocal>0){
    ensure_local_atoms(s,nlocal);
    CK(cudaMemcpy(s->d_x,x,3*(size_t)nlocal*sizeof(double),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s->d_q,q,(size_t)nlocal*sizeof(double),cudaMemcpyHostToDevice));
    int tb=256,ga=(nlocal+tb-1)/tb;
    bool quads=(spline==SOG_QUADS_4||spline==SOG_QUADS_6);
    bool bspline=(spline>=200);
    int nord=(spline==SOG_CUBES2_6||spline==SOG_QUADS_6||spline==SOG_BSPLINE_6)?6:4;
    if(bspline){
      const int bo=spline-200;
      if(bo==4)      k_spread_bspline_brick<4><<<ga,tb>>>(nlocal,s->d_x,boxlo[0],boxlo[1],boxlo[2],lx,ly,lz,s->d_q,rho_scale,nx,ny,nz,nxlo,nxhi,nylo,nyhi,nzlo,nzhi,s->s_brick_fx);
      else if(bo==5) k_spread_bspline_brick<5><<<ga,tb>>>(nlocal,s->d_x,boxlo[0],boxlo[1],boxlo[2],lx,ly,lz,s->d_q,rho_scale,nx,ny,nz,nxlo,nxhi,nylo,nyhi,nzlo,nzhi,s->s_brick_fx);
      else           k_spread_bspline_brick<6><<<ga,tb>>>(nlocal,s->d_x,boxlo[0],boxlo[1],boxlo[2],lx,ly,lz,s->d_q,rho_scale,nx,ny,nz,nxlo,nxhi,nylo,nyhi,nzlo,nzhi,s->s_brick_fx);
    }else if(quads){
      if(nord==4) k_spread_quads_brick<4><<<ga,tb>>>(nlocal,s->d_x,boxlo[0],boxlo[1],boxlo[2],lx,ly,lz,s->d_q,rho_scale,nx,ny,nz,nxlo,nxhi,nylo,nyhi,nzlo,nzhi,s->s_brick_fx);
      else k_spread_quads_brick<6><<<ga,tb>>>(nlocal,s->d_x,boxlo[0],boxlo[1],boxlo[2],lx,ly,lz,s->d_q,rho_scale,nx,ny,nz,nxlo,nxhi,nylo,nyhi,nzlo,nzhi,s->s_brick_fx);
    }else{
      if(nord==4) k_spread_cubes2_brick<4><<<ga,tb>>>(nlocal,s->d_x,boxlo[0],boxlo[1],boxlo[2],lx,ly,lz,s->d_q,rho_scale,nx,ny,nz,nxlo,nxhi,nylo,nyhi,nzlo,nzhi,s->s_brick_fx);
      else k_spread_cubes2_brick<6><<<ga,tb>>>(nlocal,s->d_x,boxlo[0],boxlo[1],boxlo[2],lx,ly,lz,s->d_q,rho_scale,nx,ny,nz,nxlo,nxhi,nylo,nyhi,nzlo,nzhi,s->s_brick_fx);
    }
    CK(cudaGetLastError());
  }
}

extern "C" void sog_gpu_spread_brick(SogGpuState*s,int nlocal,
    const double*x,const double*boxlo,double lx,double ly,double lz,
    const double*q,double rho_scale,int spline,int nx,int ny,int nz,
    int nxlo,int nxhi,int nylo,int nyhi,int nzlo,int nzhi,double*rho_brick){
  size_t nb=(size_t)(nxhi-nxlo+1)*(nyhi-nylo+1)*(nzhi-nzlo+1);
  sog_gpu_spread_brick_dev(s,nlocal,x,boxlo,lx,ly,lz,q,rho_scale,spline,nx,ny,
                           nz,nxlo,nxhi,nylo,nyhi,nzlo,nzhi);
  // Fixed-point brick -> fp64 host API (the legacy callers see a double mesh).
  std::vector<unsigned long long> fx(nb);
  CK(cudaMemcpy(fx.data(),s->s_brick_fx,nb*sizeof(unsigned long long),
                cudaMemcpyDeviceToHost));
  const double inv=1.0/kFxScale;
  for(size_t i=0;i<nb;++i)
    rho_brick[i]=inv*(double)(long long)fx[i];
}

// Gather from the brick ALREADY on the device (the distributed solve fills it
// with the potential and the forward halo); only x/q go up and F/v come down.
extern "C" void sog_gpu_gather_brick_dev(SogGpuState*s,int nlocal,
    const double*x,const double*boxlo,double lx,double ly,double lz,
    const double*q,double qscale,int spline,int nx,int ny,int nz,
    int nxlo,int nxhi,int nylo,int nyhi,int nzlo,int nzhi,
    double*force,double*vpot){
  if(nlocal<=0) return;
  size_t nb=(size_t)(nxhi-nxlo+1)*(nyhi-nylo+1)*(nzhi-nzlo+1);
  ensure_local_brick(s,nb);ensure_local_atoms(s,nlocal);
  CK(cudaMemcpy(s->d_x,x,3*(size_t)nlocal*sizeof(double),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(s->d_q,q,(size_t)nlocal*sizeof(double),cudaMemcpyHostToDevice));
  int tb=256,ga=(nlocal+tb-1)/tb;
  bool quads=(spline==SOG_QUADS_4||spline==SOG_QUADS_6);
  bool bspline=(spline>=200);
  int nord=(spline==SOG_CUBES2_6||spline==SOG_QUADS_6||spline==SOG_BSPLINE_6)?6:4;
  if(bspline){
    const int bo=spline-200;
    if(bo==4)      k_gather_bspline_brick<4><<<ga,tb>>>(nlocal,s->d_x,boxlo[0],boxlo[1],boxlo[2],lx,ly,lz,s->d_q,qscale,nx,ny,nz,nxlo,nxhi,nylo,nyhi,nzlo,nzhi,s->d_brick,s->d_force,s->d_vpot);
    else if(bo==5) k_gather_bspline_brick<5><<<ga,tb>>>(nlocal,s->d_x,boxlo[0],boxlo[1],boxlo[2],lx,ly,lz,s->d_q,qscale,nx,ny,nz,nxlo,nxhi,nylo,nyhi,nzlo,nzhi,s->d_brick,s->d_force,s->d_vpot);
    else           k_gather_bspline_brick<6><<<ga,tb>>>(nlocal,s->d_x,boxlo[0],boxlo[1],boxlo[2],lx,ly,lz,s->d_q,qscale,nx,ny,nz,nxlo,nxhi,nylo,nyhi,nzlo,nzhi,s->d_brick,s->d_force,s->d_vpot);
  }else if(quads){
    if(nord==4) k_gather_quads_brick<4><<<ga,tb>>>(nlocal,s->d_x,boxlo[0],boxlo[1],boxlo[2],lx,ly,lz,s->d_q,qscale,nx,ny,nz,nxlo,nxhi,nylo,nyhi,nzlo,nzhi,s->d_brick,s->d_force,s->d_vpot);
    else k_gather_quads_brick<6><<<ga,tb>>>(nlocal,s->d_x,boxlo[0],boxlo[1],boxlo[2],lx,ly,lz,s->d_q,qscale,nx,ny,nz,nxlo,nxhi,nylo,nyhi,nzlo,nzhi,s->d_brick,s->d_force,s->d_vpot);
  }else{
    if(nord==4) k_gather_cubes2_brick<4><<<ga,tb>>>(nlocal,s->d_x,boxlo[0],boxlo[1],boxlo[2],lx,ly,lz,s->d_q,qscale,nx,ny,nz,nxlo,nxhi,nylo,nyhi,nzlo,nzhi,s->d_brick,s->d_force,s->d_vpot);
    else k_gather_cubes2_brick<6><<<ga,tb>>>(nlocal,s->d_x,boxlo[0],boxlo[1],boxlo[2],lx,ly,lz,s->d_q,qscale,nx,ny,nz,nxlo,nxhi,nylo,nyhi,nzlo,nzhi,s->d_brick,s->d_force,s->d_vpot);
  }
  CK(cudaGetLastError());
  CK(cudaMemcpy(force,s->d_force,3*(size_t)nlocal*sizeof(double),cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(vpot,s->d_vpot,(size_t)nlocal*sizeof(double),cudaMemcpyDeviceToHost));
}

extern "C" void sog_gpu_gather_brick(SogGpuState*s,int nlocal,
    const double*x,const double*boxlo,double lx,double ly,double lz,
    const double*q,double qscale,int spline,int nx,int ny,int nz,
    int nxlo,int nxhi,int nylo,int nyhi,int nzlo,int nzhi,const double*u_brick,
    double*force,double*vpot){
  size_t nb=(size_t)(nxhi-nxlo+1)*(nyhi-nylo+1)*(nzhi-nzlo+1);
  ensure_local_brick(s,nb);
  // fp64 host potential -> float device brick (the legacy callers pass doubles).
  std::vector<float> uf(nb);
  for(size_t i=0;i<nb;++i) uf[i]=(float)u_brick[i];
  CK(cudaMemcpy(s->d_brick,uf.data(),nb*sizeof(float),cudaMemcpyHostToDevice));
  sog_gpu_gather_brick_dev(s,nlocal,x,boxlo,lx,ly,lz,q,qscale,spline,nx,ny,nz,
                           nxlo,nxhi,nylo,nyhi,nzlo,nzhi,force,vpot);
}

// ── Host-buffer wrapper (kept for the non-Kokkos call sites in sog.cpp) ──
extern "C" double sog_gpu_compute(SogGpuState*s,int nlocal,const double*x,const double*boxlo,
    double lx,double ly,double lz,const double*q,double qscale,int spline,int want_pot,
    double*force,double*virial6,double*vpot,double self_coeff,double qsqsum,int remove_self){
  if(nlocal>s->cap_atoms){
    cudaFree(s->d_x);cudaFree(s->d_q);cudaFree(s->d_force);cudaFree(s->d_vpot);
    CK(cudaMalloc(&s->d_x,3*nlocal*sizeof(double)));CK(cudaMalloc(&s->d_q,nlocal*sizeof(double)));
    CK(cudaMalloc(&s->d_force,3*nlocal*sizeof(double)));CK(cudaMalloc(&s->d_vpot,nlocal*sizeof(double)));
    s->cap_atoms=nlocal;
  }
  CK(cudaMemcpy(s->d_x,x,3*(size_t)nlocal*sizeof(double),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(s->d_q,q,(size_t)nlocal*sizeof(double),cudaMemcpyHostToDevice));
  double e = sog_gpu_compute_core(s,nlocal,s->d_x,s->d_q,boxlo,lx,ly,lz,qscale,spline,
                                  want_pot,s->d_force,s->d_vpot,self_coeff,qsqsum,
                                  remove_self,0,virial6,1);
  CK(cudaMemcpy(force,s->d_force,3*(size_t)nlocal*sizeof(double),cudaMemcpyDeviceToHost));
  if(want_pot) CK(cudaMemcpy(vpot,s->d_vpot,(size_t)nlocal*sizeof(double),cudaMemcpyDeviceToHost));
  return e;
}

// ── Device-resident entry point (LAMMPS Kokkos) ──
// x, q, force and vpot are device pointers (e.g. Kokkos views).  Only the
// energy/virial reductions (160 bytes) cross back to the host.
extern "C" double sog_gpu_compute_dev(SogGpuState*s,int nlocal,const double*d_x,const double*boxlo,
    double lx,double ly,double lz,const double*d_q,double qscale,int spline,int want_pot,
    double*d_force,double*virial6,double*d_vpot,double self_coeff,double qsqsum,int remove_self){
  return sog_gpu_compute_core(s,nlocal,d_x,d_q,boxlo,lx,ly,lz,qscale,spline,want_pot,
                              d_force,d_vpot,self_coeff,qsqsum,remove_self,0,virial6,1);
}

// Deferred-reduction variant: the pipeline runs exactly as above, but the 20
// double sums stay in the device buffer instead of crossing to the host, so a
// caller that assembles its outputs on the device pays no synchronization.
// Read them with sog_gpu_red_device_ptr; virial6 and the return value are not
// written.
extern "C" void sog_gpu_compute_dev_defer(SogGpuState*s,int nlocal,const double*d_x,
    const double*boxlo,double lx,double ly,double lz,const double*d_q,double qscale,
    int spline,int want_pot,double*d_force,double*d_vpot,double self_coeff,
    double qsqsum,int remove_self,int force_mode){
  sog_gpu_compute_core(s,nlocal,d_x,d_q,boxlo,lx,ly,lz,qscale,spline,want_pot,
                       d_force,d_vpot,self_coeff,qsqsum,remove_self,force_mode,nullptr,0);
}

extern "C" double* sog_gpu_red_device_ptr(SogGpuState*s){ return s->d_red; }

// Split entry points for a distributed solve: the spread of this rank's atoms
// into the full mesh, an accessor for that mesh, and the solve that consumes a
// mesh already summed over the ranks.
extern "C" void sog_gpu_spread_f(SogGpuState*s,int nlocal,const double*x,
    const double*boxlo,double lx,double ly,double lz,const double*q,
    double rho_scale,int spline){
  size_t ng=s->ngrid;
  int tb=256, gg=(int)((ng+tb-1)/tb), gw=(nlocal+(tb/32)-1)/(tb/32);
  if(s->d_boxlo==nullptr) CK(cudaMalloc(&s->d_boxlo,3*sizeof(double)));
  CK(cudaMemcpy(s->d_boxlo,boxlo,3*sizeof(double),cudaMemcpyHostToDevice));
  k_zero_fx<<<gg,tb>>>(ng,s->f_rho_fx);
  // The fixed-point mesh is the CubeS2 path; the order selects the stencil.
  const int nord=(spline==SOG_CUBES2_6)?6:4;
  if(nlocal>0){
    if(nord==6) k_spread_cubes2_warp_fx<6><<<gw,tb>>>(nlocal,x,s->d_boxlo,lx,ly,lz,q,rho_scale,s->nx,s->ny,s->nz,s->f_rho_fx);
    else        k_spread_cubes2_warp_fx<4><<<gw,tb>>>(nlocal,x,s->d_boxlo,lx,ly,lz,q,rho_scale,s->nx,s->ny,s->nz,s->f_rho_fx);
  }
}

extern "C" unsigned long long* sog_gpu_fx_device_ptr(SogGpuState*s){ return s->f_rho_fx; }

extern "C" void sog_gpu_fx_to_real(SogGpuState*s){
  size_t ng=s->ngrid;
  int tb=256, gg=(int)((ng+tb-1)/tb);
  k_fx_to_float<<<gg,tb>>>(ng,s->f_rho_fx,s->f_rho_r,1.0/kFxScale);
}

extern "C" float* sog_gpu_real_device_ptr(SogGpuState*s){ return s->f_rho_r; }

// ``density_ready`` skips the fixed-point to float conversion, for a caller
// that summed the real density over the ranks first.
extern "C" void sog_gpu_solve_f(SogGpuState*s,int nlocal,const double*x,
    const double*boxlo,double lx,double ly,double lz,const double*q,double qscale,
    int want_pot,int density_ready,double*d_force,double*d_vpot,int spline){
  size_t ng=s->ngrid;
  int tb=256, gg=(int)((ng+tb-1)/tb), gw=(nlocal+(tb/32)-1)/(tb/32);
  double volume=lx*ly*lz, scaleinv=1.0/(double)ng;
  const double kfx=2*M_PI/lx, kfy=2*M_PI/ly, kfz=2*M_PI/lz;
  if(s->d_red==nullptr) CK(cudaMalloc(&s->d_red,20*sizeof(double)));
  CK(cudaMemset(s->d_red,0,20*sizeof(double)));
  if(s->d_boxlo==nullptr) CK(cudaMalloc(&s->d_boxlo,3*sizeof(double)));
  CK(cudaMemcpy(s->d_boxlo,boxlo,3*sizeof(double),cudaMemcpyHostToDevice));
  ensure_pot_r(s);
  ensure_spec(s);
  if(!density_ready) k_fx_to_float<<<gg,tb>>>(ng,s->f_rho_fx,s->f_rho_r,1.0/kFxScale);
  cufftExecR2C(s->f_plan_r2c,s->f_rho_r,s->f_spec);
  {
    const int nh=s->nx/2+1;
    const int npw=(nh+2)/3;
    const int gx=(npw*s->ny+63)/64;
    int gz=240/gx; if(gz<1) gz=1; if(gz>s->nz) gz=s->nz;
    k_kspace_h<<<dim3(gx,gz),64>>>(s->nx,s->ny,s->nz,kfx,kfy,kfz,s->d_hinv,
                                   s->metric,s->f_gp,s->f_spec,want_pot,
                                   scaleinv,s->d_red);
  }
  cufftExecC2R(s->f_plan_c2r,s->f_spec,s->f_pot_r);
  FracNode* frac=ensure_frac(s,nlocal);
  if(nlocal>0) k_midtown_frac<<<(nlocal+tb-1)/tb,tb>>>(nlocal,x,s->d_boxlo,1.0/lx,1.0/ly,1.0/lz,s->nx,s->ny,s->nz,frac);
  const int nord=(spline==SOG_CUBES2_6)?6:4;
  if(nlocal>0){
    if(nord==6) k_gather_cubes2_warp_r<6><<<gw,tb>>>(nlocal,frac,1.0/lx,1.0/ly,1.0/lz,q,qscale,s->nx,s->ny,s->nz,s->f_pot_r,want_pot,d_force,d_vpot,s->d_red);
    else        k_gather_cubes2_warp_r<4><<<gw,tb>>>(nlocal,frac,1.0/lx,1.0/ly,1.0/lz,q,qscale,s->nx,s->ny,s->nz,s->f_pot_r,want_pot,d_force,d_vpot,s->d_red);
  }
  (void)volume;
}

// Triclinic mode: the kernels then work in fractional coordinates on a unit
// cell, k comes from the metric A^{-1}, and the volume is |det A|.  Called
// only when the cell or the mesh changes, never per step.
extern "C" void sog_gpu_set_metric(SogGpuState*s,const double*hinv,double volume,int on){
  if(s->d_hinv==nullptr) CK(cudaMalloc(&s->d_hinv,9*sizeof(double)));
  if(on) CK(cudaMemcpy(s->d_hinv,hinv,9*sizeof(double),cudaMemcpyHostToDevice));
  s->metric=on?1:0;
  s->volume=volume;
}

// ── Multi-rank replicated-grid path ─────────────────────────────────────────
// The distributed (Grid3d/Remap) mesh keeps the FFT on the host, and at np=2
// the KISS pencils alone cost ~86 ms/step for a 125^3 grid -- 25x the whole
// single-GPU solve.  Instead every rank spreads its own atoms into a FULL grid,
// the grids are summed with one MPI_Allreduce, and each rank then runs the
// same device cuFFT/green/gather pipeline it would run alone.  Correct by
// construction: the summed density is the global one, so the potential is
// identical to the single-rank solve, and each rank gathers only its local
// atoms.  The price is one grid-sized reduction per step; the benefit is a
// device speed FFT at any rank count.

// Zero the device mesh and spread ``nlocal`` atoms into it.
extern "C" void sog_gpu_spread_full_dev(SogGpuState*s,int nlocal,const double*d_x,
    const double*boxlo,double lx,double ly,double lz,const double*d_q,
    double rho_scale,int spline){
  size_t ng=s->ngrid;
  if(s->d_rho==nullptr) return;
  int tb=256, gg=(int)((ng+tb-1)/tb), ga=(nlocal+tb-1)/tb;
  if(s->d_boxlo==nullptr) CK(cudaMalloc(&s->d_boxlo,3*sizeof(double)));
  CK(cudaMemcpy(s->d_boxlo,boxlo,3*sizeof(double),cudaMemcpyHostToDevice));
  k_zero_complex<<<gg,tb>>>(s->d_rho,ng);
  bool quads=(spline==SOG_QUADS_4||spline==SOG_QUADS_6);
  int nord=(spline==SOG_CUBES2_6||spline==SOG_QUADS_6)?6:4;
  if(nlocal<=0) return;
  if(quads){
    if(nord==4) k_spread_quads<4,cufftDoubleComplex><<<ga,tb>>>(nlocal,d_x,s->d_boxlo,lx,ly,lz,d_q,rho_scale,s->nx,s->ny,s->nz,s->d_rho);
    else        k_spread_quads<6,cufftDoubleComplex><<<ga,tb>>>(nlocal,d_x,s->d_boxlo,lx,ly,lz,d_q,rho_scale,s->nx,s->ny,s->nz,s->d_rho);
  } else {
    const int gw=(nlocal+(tb/32)-1)/(tb/32);
    if(nord==4) k_spread_cubes2_warp<4,cufftDoubleComplex><<<gw,tb>>>(nlocal,d_x,s->d_boxlo,lx,ly,lz,d_q,rho_scale,s->nx,s->ny,s->nz,s->d_rho);
    else        k_spread_cubes2_warp<6,cufftDoubleComplex><<<gw,tb>>>(nlocal,d_x,s->d_boxlo,lx,ly,lz,d_q,rho_scale,s->nx,s->ny,s->nz,s->d_rho);
  }
}

// Device mesh <-> host buffer (the reduction runs on the host; OpenMPI here is
// not GPU-aware, so the grid is staged through host memory).
// Raw device mesh pointer, for a CUDA-aware MPI reduction (no host staging).
extern "C" double* sog_gpu_rho_device_ptr(SogGpuState*s){ return s->d_rho ? (double*)s->d_rho : nullptr; }

extern "C" void sog_gpu_rho_download(SogGpuState*s,double*host){
  CK(cudaMemcpy(host,s->d_rho,2*s->ngrid*sizeof(double),cudaMemcpyDeviceToHost));
}
extern "C" void sog_gpu_rho_upload(SogGpuState*s,const double*host){
  CK(cudaMemcpy(s->d_rho,host,2*s->ngrid*sizeof(double),cudaMemcpyHostToDevice));
}

// FFT -> green multiply -> inverse, then gather force/potential for ``nlocal``
// atoms.  The density must already be the summed global mesh.
extern "C" double sog_gpu_solve_dev(SogGpuState*s,int nlocal,const double*d_x,
    const double*boxlo,double lx,double ly,double lz,const double*d_q,
    double qscale,int spline,int want_pot,double*d_force,double*virial6,
    double*d_vpot,double self_coeff,double qsqsum,int remove_self){
  size_t ng=s->ngrid;
  if(s->d_boxlo==nullptr) CK(cudaMalloc(&s->d_boxlo,3*sizeof(double)));
  CK(cudaMemcpy(s->d_boxlo,boxlo,3*sizeof(double),cudaMemcpyHostToDevice));
  if(s->d_red==nullptr) CK(cudaMalloc(&s->d_red,20*sizeof(double)));
  CK(cudaMemset(s->d_red,0,20*sizeof(double)));
  double volume=lx*ly*lz, scaleinv=1.0/(double)ng;
  int tb=256, gg=(int)((ng+tb-1)/tb), ga=(nlocal+tb-1)/tb;
  bool quads=(spline==SOG_QUADS_4||spline==SOG_QUADS_6);
  int nord=(spline==SOG_CUBES2_6||spline==SOG_QUADS_6)?6:4;
  cufftExecZ2Z(s->plan,s->d_rho,s->d_rho,CUFFT_FORWARD);
  k_kspace<<<gg,tb>>>(s->nx,s->ny,s->nz,lx,ly,lz,s->d_hinv,s->metric,s->d_rho,s->d_ge,s->d_gf,s->d_gv,
                      s->d_gx,s->d_gy,s->d_gz,s->d_pot,want_pot,
                      scaleinv,s->d_red);
  cufftExecZ2Z(s->plan,s->d_pot,s->d_pot,CUFFT_INVERSE);
  if(nlocal>0){
    if(quads){
      if(nord==4) k_gather_quads<4,cufftDoubleComplex><<<ga,tb>>>(nlocal,d_x,s->d_boxlo,lx,ly,lz,d_q,qscale,s->nx,s->ny,s->nz,s->d_pot,want_pot,d_force,d_vpot,s->d_red);
      else        k_gather_quads<6,cufftDoubleComplex><<<ga,tb>>>(nlocal,d_x,s->d_boxlo,lx,ly,lz,d_q,qscale,s->nx,s->ny,s->nz,s->d_pot,want_pot,d_force,d_vpot,s->d_red);
    } else {
      const int gw=(nlocal+(tb/32)-1)/(tb/32);
      if(nord==4) k_gather_cubes2_warp<4,cufftDoubleComplex><<<gw,tb>>>(nlocal,d_x,s->d_boxlo,lx,ly,lz,d_q,qscale,s->nx,s->ny,s->nz,s->d_pot,want_pot,d_force,d_vpot,s->d_red);
      else        k_gather_cubes2_warp<6,cufftDoubleComplex><<<gw,tb>>>(nlocal,d_x,s->d_boxlo,lx,ly,lz,d_q,qscale,s->nx,s->ny,s->nz,s->d_pot,want_pot,d_force,d_vpot,s->d_red);
    }
  }
  double red[20]; CK(cudaMemcpy(red,s->d_red,20*sizeof(double),cudaMemcpyDeviceToHost));
  // Exact closed-form self term; see sog_gpu_compute_dev for the derivation.
  // rsi = g(0) = 2*self_coeff, and the self term contributes no virial.
  if(want_pot){
    const double rsi=remove_self?2.0*self_coeff:0.0;
    if(nlocal>0) k_vpot_self<<<ga,tb>>>(nlocal,d_vpot,d_q,qscale,rsi,0.0);
  }
  double e_kneq0=0.5*volume*red[0];
  double e_self=remove_self?self_coeff*qsqsum:0.0;
  for(int j=0;j<6;++j) virial6[j]=0.5*volume*qscale*red[1+j];
  return qscale*(e_kneq0-e_self);
}
