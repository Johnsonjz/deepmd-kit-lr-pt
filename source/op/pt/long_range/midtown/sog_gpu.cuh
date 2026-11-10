// sog_gpu.cuh — plugin-internal GPU SOG kspace (raw CUDA + cuFFT).
// Called from SOGKSpace::compute_single when `use_gpu` is set. Keeping the GPU path inside
// SOGKSpace leaves the fix sog/response interface (dynamic_cast<SOGKSpace*>, get_potential())
// untouched — no in-tree LAMMPS build, no lib/gpu. See papers/sog/sog-progress.md §8.
#ifndef SOG_GPU_CUH
#define SOG_GPU_CUH

struct SogGpuState;  // opaque device state (buffers + cuFFT plans + green tables)

// Spline selector for the GPU spread/gather.  CubeS2 keeps its legacy 4/6
// codes; every other family is 100 + order (QuadS) or 200 + order (B-spline),
// so the order is `code - family_base` and the family is the hundreds digit.
enum SogSpline {
  SOG_CUBES2_4 = 4,
  SOG_CUBES2_6 = 6,
  SOG_QUADS_4 = 104,
  SOG_QUADS_6 = 106,
  SOG_BSPLINE_4 = 204,
  SOG_BSPLINE_5 = 205,
  SOG_BSPLINE_6 = 206,
};

extern "C" {
SogGpuState *sog_gpu_create();
void sog_gpu_destroy(SogGpuState *s);

// (Re)allocate device mesh buffers, upload the host-computed green tables, and (re)plan cuFFT
// for grid nx*ny*nz. Called from ensure_fft_plan whenever the grid changes.
// (The bare K / K_v "self" tables are gone: the self-energy term is closed-form
//  and needs no per-mode table, see sog_gpu_solve_dev.)
void sog_gpu_setup_slab(SogGpuState *s, int nx, int ny, int nz, int YL,
                        int ZL, const double *ge, const double *gv, size_t gn);
void sog_gpu_setup(SogGpuState *s, int nx, int ny, int nz,
                   const double *green_energy, const double *green_force,
                   const double *green_virial);
// Same, but without the fp64 mesh set (d_rho/d_pot/d_ge/gf/gv, 56 B per grid
// point).  Only for callers whose solve reads f_* alone: sog_gpu_compute* and
// sog_gpu_solve_f qualify; sog_gpu_solve_dev and the slab path do not.
void sog_gpu_setup_f32(SogGpuState *s, int nx, int ny, int nz,
                       const double *green_energy, const double *green_force,
                       const double *green_virial);

// Convert the packed green table of the replicated solve in place from Form B
// (variance subtraction) to Form A (divide by the measured assignment-window
// symbol <|W|^2>).  `nord` is the CubeS2 order (4 or 6), `nth` the number of
// sub-cell samples per axis.  Returns 0 on success.  Once per box.
int sog_gpu_green_form_a(SogGpuState *s, int nord, int nth, double lx,
                         double ly, double lz);

// Measure the CubeS2 assignment-window symbol <|W(k)|^2> on the full grid into
// a host buffer of the compact half spectrum (kh + nh*(iy + ny*iz), nh =
// nx/2+1), for the z-slab Form-A conversion.  Self-contained (own full-grid
// R2C plan + buffers), so it runs in slab mode.  Returns 0 on success.
int sog_gpu_measure_window(int nx, int ny, int nz, int nord, int nth, double lx,
                           double ly, double lz, double *out);

// One SOG kspace step, entirely on device:
//   spread(q@x) -> cuFFT fwd -> green-multiply (grad meshes + potential + energy/virial reductions)
//   -> cuFFT inv -> gather force (+ potential v_i if want_potential).
// x is nlocal*3 wrapped coords (box origin already handled by caller via boxlo/lx..lz);
// qscale = force->qqrd2e*scale. Fills force[nlocal*3], virial6[6]; returns energy.
// If want_potential, fills vpot[nlocal].
double sog_gpu_compute(SogGpuState *s, int nlocal,
                       const double *x, const double *boxlo,
                       double lx, double ly, double lz,
                       const double *q, double qscale,
                       int spline, int want_potential,
                       double *force, double *virial6, double *vpot,
                       double self_coeff, double qsqsum, int remove_self);

// Device-resident variant: x, q, force and vpot are device pointers (Kokkos
// views). Only the energy/virial reductions return to the host.
// ── distributed (slab) k-space, host-buffer entry points ──
// The orchestrator (sog_kspace_op.cc) owns the decomposition, the halo
// exchange and the transposes; these only move one buffer to the device,
// transform it, and bring it back.
int sog_gpu_slab_setup(SogGpuState *s, int nx, int ny, int nz, int ZL, int YL);
int sog_gpu_slab_unpack_green_izpack(SogGpuState *s, int nx, int ny, int nz,
                                     int ylo, int YL, double lx, double ly,
                                     double lz, double scaleinv, int np,
                                     const double *host_rcv,
                                     double *host_snd_back, double *red7);

double sog_gpu_compute_dev(SogGpuState *s, int nlocal,
                           const double *d_x, const double *boxlo,
                           double lx, double ly, double lz,
                           const double *d_q, double qscale,
                           int spline, int want_potential,
                           double *d_force, double *virial6, double *d_vpot,
                           double self_coeff, double qsqsum, int remove_self);

// Deferred-reduction variant: identical pipeline, but the 20 reduction sums
// stay in the device buffer (sog_gpu_red_device_ptr) and no host copy runs.
// `force_mode` 0 is the AD gather (weight derivatives); 1 is the ik path
// (three inverse transforms of i·k_a·φ̂ and a value-only gather).
void sog_gpu_compute_dev_defer(SogGpuState *s, int nlocal, const double *d_x,
                               const double *boxlo, double lx, double ly,
                               double lz, const double *d_q, double qscale,
                               int spline, int want_pot, double *d_force,
                               double *d_vpot, double self_coeff,
                               double qsqsum, int remove_self, int force_mode);
double *sog_gpu_red_device_ptr(SogGpuState *s);
// Distributed split: spread this rank's atoms into the full mesh, expose the
// fixed-point mesh for a rank sum, then solve with the summed mesh.
void sog_gpu_spread_f(SogGpuState *s, int nlocal, const double *x,
                      const double *boxlo, double lx, double ly, double lz,
                      const double *q, double rho_scale, int spline);
unsigned long long *sog_gpu_fx_device_ptr(SogGpuState *s);
void sog_gpu_fx_to_real(SogGpuState *s);
float *sog_gpu_real_device_ptr(SogGpuState *s);
void sog_gpu_solve_f(SogGpuState *s, int nlocal, const double *x,
                     const double *boxlo, double lx, double ly, double lz,
                     const double *q, double qscale, int want_pot,
                     int density_ready, double *d_force, double *d_vpot,
                     int spline);
// Triclinic: fractional coordinates on a unit cell + metric k vectors.
void sog_gpu_set_metric(SogGpuState *s, const double *hinv, double volume, int on);

// Multi-rank replicated-grid path (see sog_gpu.cu). The density mesh is the
// summed global grid; each rank then runs the device FFT/green/gather itself.
void sog_gpu_spread_full_dev(SogGpuState *s, int nlocal, const double *d_x,
                             const double *boxlo, double lx, double ly,
                             double lz, const double *d_q, double rho_scale,
                             int spline);
double *sog_gpu_rho_device_ptr(SogGpuState *s);  // for CUDA-aware MPI
void sog_gpu_rho_download(SogGpuState *s, double *host);
void sog_gpu_rho_upload(SogGpuState *s, const double *host);
double sog_gpu_solve_dev(SogGpuState *s, int nlocal, const double *d_x,
                         const double *boxlo, double lx, double ly, double lz,
                         const double *d_q, double qscale, int spline,
                         int want_pot, double *d_force, double *virial6,
                         double *d_vpot, double self_coeff, double qsqsum,
                         int remove_self);

// Local brick operations used by the PPPM-style distributed backend.  Grid
// indices are global and may extend below zero or above N-1 in the ghost
// layers.  The host owns Grid3d communication and the distributed FFT; CUDA
// performs only particle-local spreading and adjoint gathering.
void sog_gpu_spread_brick_dev(SogGpuState *s, int nlocal, const double *x,
                              const double *boxlo, double lx, double ly,
                              double lz, const double *q, double rho_scale,
                              int spline, int nx, int ny, int nz, int nxlo,
                              int nxhi, int nylo, int nyhi, int nzlo, int nzhi);
void sog_gpu_gather_brick_dev(SogGpuState *s, int nlocal, const double *x,
                              const double *boxlo, double lx, double ly,
                              double lz, const double *q, double qscale,
                              int spline, int nx, int ny, int nz, int nxlo,
                              int nxhi, int nylo, int nyhi, int nzlo, int nzhi,
                              double *force, double *vpot);
void sog_gpu_brick_get(SogGpuState *s, double *host, size_t off, size_t n);
void sog_gpu_brick_set(SogGpuState *s, const double *host, size_t off,
                       size_t n);
void sog_gpu_brick_add(SogGpuState *s, const double *host, size_t off,
                       size_t n);
void sog_gpu_brick_zero(SogGpuState *s, size_t off, size_t n);
int sog_gpu_slab_copy_packed(SogGpuState *s, int forward, size_t n);
void sog_gpu_brick_add_planes(SogGpuState *s, size_t src_off, size_t dst_off,
                              size_t n);
void sog_gpu_brick_copy_planes(SogGpuState *s, size_t src_off, size_t dst_off,
                               size_t n);
int sog_gpu_slab_xyfft_pack_brick_dev(SogGpuState *s, int nx, int ny, int nz,
                                      int ZL, int np, int zoff);
int sog_gpu_slab_unpack_green_izpack_dev(SogGpuState *s, int nx, int ny, int nz,
                                         int ylo, int YL, double lx, double ly,
                                         double lz, double scaleinv, int np);
int sog_gpu_slab_unpack_ixyfft_to_brick_dev(SogGpuState *s, int nx, int ny,
                                            int nz, int ZL, int np, int zoff,
                                            int ZB);
int sog_gpu_nccl_get_id(void *id);
int sog_gpu_nccl_init(int np, int me, const void *id);
int sog_gpu_nccl_ok();
int sog_gpu_nccl_available();
int sog_gpu_nccl_allreduce_dev(void *buf, size_t n);
// Same, with an explicit element width (kNcclDouble / kNcclFloat32 /
// kNcclUint64).  The full-brick path sums the fixed-point density as uint64.
int sog_gpu_nccl_allreduce_dev_dtype(void *buf, size_t n, int dtype);
// full_brick: make the owned planes of the brick global (rank sum).  `off`/`n`
// are element counts inside the brick.
int sog_gpu_brick_allreduce_fx(SogGpuState *s, size_t off, size_t n);
int sog_gpu_brick_allreduce_f(SogGpuState *s, size_t off, size_t n);
int sog_gpu_nccl_alltoall_fwd(SogGpuState *s, const int *scnt,
                              const int *sdisp, const int *rcnt,
                              const int *rdisp, int np);
int sog_gpu_nccl_alltoall_bwd(SogGpuState *s, const int *scnt,
                              const int *sdisp, const int *rcnt,
                              const int *rdisp, int np);
int sog_gpu_nccl_halo_rev(SogGpuState *s, int prev, int next, size_t plane,
                          int zoff, int ZL, int gh_lo, int gh_hi);
int sog_gpu_nccl_halo_fwd(SogGpuState *s, int prev, int next, size_t plane,
                          int zoff, int ZL, int gh_lo, int gh_hi);
int sog_gpu_nccl_allreduce_red(SogGpuState *s);
void sog_gpu_red_download(SogGpuState *s, double *host);
int sog_gpu_slab_xyfft_pack_brick(SogGpuState *s, int nx, int ny, int nz,
                                  int ZL, int np, int zoff, double *host_snd);
int sog_gpu_slab_unpack_ixyfft_to_brick(SogGpuState *s, int nx, int ny, int nz,
                                        int ZL, int np, int zoff, int ZB,
                                        const double *host_back);
void sog_gpu_spread_brick(SogGpuState *s, int nlocal,
                          const double *x, const double *boxlo,
                          double lx, double ly, double lz,
                          const double *q, double rho_scale, int spline,
                          int nx, int ny, int nz,
                          int nxlo, int nxhi, int nylo, int nyhi,
                          int nzlo, int nzhi, double *rho_brick);

void sog_gpu_gather_brick(SogGpuState *s, int nlocal,
                          const double *x, const double *boxlo,
                          double lx, double ly, double lz,
                          const double *q, double qscale, int spline,
                          int nx, int ny, int nz,
                          int nxlo, int nxhi, int nylo, int nyhi,
                          int nzlo, int nzhi, const double *u_brick,
                          double *force, double *vpot);
}
#endif  // SOG_GPU_CUH
