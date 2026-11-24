// SPDX-License-Identifier: LGPL-3.0-or-later
/**
 * See https://docs.lammps.org/Developer_plugins.html
 */
#include <string>

#include "c_api.h"
#include "compute_deepmd_fparam_dedn.h"
#include "compute_deeptensor_atom.h"
#include "deepmd_version.h"
#include "error.h"
#include "fix_dplr.h"
#include "lammps.h"
#include "lammpsplugin.h"
#ifdef LMP_KOKKOS
#include "pair_deepmd_kokkos.h"
#include "pair_dpa4spin_kokkos.h"
#endif
#include "pair_deepmd.h"
#include "pair_deepspin.h"
#include "pair_dpa4spin.h"
#include "version.h"
#if LAMMPS_VERSION_NUMBER >= 20220328
#include "pppm_dplr.h"
#endif

using namespace LAMMPS_NS;

static Pair* pairdeepmd(LAMMPS* lmp) { return new PairDeepMD(lmp); }
static Pair* pairdeepspin(LAMMPS* lmp) { return new PairDeepSpin(lmp); }
static Pair* pairdpa4spin(LAMMPS* lmp) { return new PairDPA4Spin(lmp); }

#ifdef LMP_KOKKOS
// Runtime plugins do not consume the PairStyle declarations used by LAMMPS'
// built-in package machinery, so register each Kokkos alias explicitly.
static Pair* pairdeepmdkokkosdevice(LAMMPS* lmp) {
  return new PairDeepMDKokkos<LMPDeviceType>(lmp);
}
static Pair* pairdeepmdkokkoshost(LAMMPS* lmp) {
  return new PairDeepMDKokkos<LMPHostType>(lmp);
}
static Pair* pairdpa4spinkokkosdevice(LAMMPS* lmp) {
  return new PairDPA4SpinKokkos<LMPDeviceType>(lmp);
}
static Pair* pairdpa4spinkokkoshost(LAMMPS* lmp) {
  return new PairDPA4SpinKokkos<LMPHostType>(lmp);
}
#endif

static Compute* computedeepmdtensoratom(LAMMPS* lmp, int narg, char** arg) {
  return new ComputeDeeptensorAtom(lmp, narg, arg);
}

static Compute* computedeepmdfparamdedn(LAMMPS* lmp, int narg, char** arg) {
  return new ComputeDeepmdFparamDedn(lmp, narg, arg);
}

static Fix* fixdplr(LAMMPS* lmp, int narg, char** arg) {
  return new FixDPLR(lmp, narg, arg);
}

#if LAMMPS_VERSION_NUMBER >= 20220328
static KSpace* pppmdplr(LAMMPS* lmp) { return new PPPMDPLR(lmp); }
#endif

extern "C" void lammpsplugin_init(void* lmp, void* handle, void* regfunc) {
  // A C symbol hides a changed argument list: a binary built against another
  // DeePMD-kit C API calls this library with shifted arguments instead of
  // failing to load, and an old canonical call then hands its edge count to
  // the stream argument. Compare the versions before registering any style.
  const int loaded_api = DP_GetApiVersion();
  if (loaded_api != DP_C_API_VERSION) {
    static_cast<LAMMPS*>(lmp)->error->all(
        FLERR,
        std::string("DeePMD-kit C API version mismatch: this plugin was built "
                    "for C API ") +
            std::to_string(DP_C_API_VERSION) +
            " but the loaded DeePMD-kit library reports " +
            std::to_string(loaded_api) +
            ". The argument lists differ, so a call reads the wrong arguments "
            "(an old canonical call hands its edge count to the stream "
            "argument). Put the library of the tree this plugin was built from "
            "on LD_LIBRARY_PATH, or rebuild the plugin against that tree.");
  }

  lammpsplugin_t plugin;
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc)regfunc;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "deepmd";
  plugin.info = "deepmd pair style " STR_GIT_SUMM;
  plugin.author = "Han Wang";
  plugin.creator.v1 = (lammpsplugin_factory1*)&pairdeepmd;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);

#ifdef LMP_KOKKOS
  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "deepmd/kk";
  plugin.info = "deepmd Kokkos pair style " STR_GIT_SUMM;
  plugin.author = "Tiancheng Li";
  plugin.creator.v1 = (lammpsplugin_factory1*)&pairdeepmdkokkosdevice;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);

  plugin.name = "deepmd/kk/device";
  (*register_plugin)(&plugin, lmp);

  plugin.name = "deepmd/kk/host";
  plugin.creator.v1 = (lammpsplugin_factory1*)&pairdeepmdkokkoshost;
  (*register_plugin)(&plugin, lmp);

  plugin.name = "dpa4spin/kk";
  plugin.info = "dpa4spin Kokkos pair style " STR_GIT_SUMM;
  plugin.author = "Tiancheng Li";
  plugin.creator.v1 = (lammpsplugin_factory1*)&pairdpa4spinkokkosdevice;
  (*register_plugin)(&plugin, lmp);

  plugin.name = "dpa4spin/kk/device";
  (*register_plugin)(&plugin, lmp);

  plugin.name = "dpa4spin/kk/host";
  plugin.creator.v1 = (lammpsplugin_factory1*)&pairdpa4spinkokkoshost;
  (*register_plugin)(&plugin, lmp);
#endif

  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "deepspin";
  plugin.info = "deepspin pair style " STR_GIT_SUMM;
  plugin.author = "Duo Zhang";
  plugin.creator.v1 = (lammpsplugin_factory1*)&pairdeepspin;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);

  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "dpa4spin";
  plugin.info = "dpa4spin pair style " STR_GIT_SUMM;
  plugin.author = "Tiancheng Li";
  plugin.creator.v1 = (lammpsplugin_factory1*)&pairdpa4spin;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);

  plugin.style = "compute";
  plugin.name = "deeptensor/atom";
  plugin.info = "compute deeptensor/atom " STR_GIT_SUMM;
  plugin.author = "Han Wang";
  plugin.creator.v2 = (lammpsplugin_factory2*)&computedeepmdtensoratom;
  (*register_plugin)(&plugin, lmp);

  plugin.style = "compute";
  plugin.name = "deepmd/fparam/dedn";
  plugin.info = "compute deepmd/fparam/dedn " STR_GIT_SUMM;
  plugin.author = "Li Fu";
  plugin.creator.v2 = (lammpsplugin_factory2*)&computedeepmdfparamdedn;
  (*register_plugin)(&plugin, lmp);

  plugin.style = "fix";
  plugin.name = "dplr";
  plugin.info = "fix dplr " STR_GIT_SUMM;
  plugin.author = "Han Wang";
  plugin.creator.v2 = (lammpsplugin_factory2*)&fixdplr;
  (*register_plugin)(&plugin, lmp);

#if LAMMPS_VERSION_NUMBER >= 20220328
  // lammps/lammps#
  plugin.style = "kspace";
  plugin.name = "pppm/dplr";
  plugin.info = "kspace pppm/dplr " STR_GIT_SUMM;
  plugin.author = "Han Wang";
  plugin.creator.v1 = (lammpsplugin_factory1*)&pppmdplr;
  (*register_plugin)(&plugin, lmp);
#endif
}
