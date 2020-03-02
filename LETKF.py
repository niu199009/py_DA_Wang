from basic_import import *
from localization import *



def LETKF(ens, obs, H, set_DA):

    #ens,obs,H,set_DA = foreCastEnsemble, obs_for_DA, H, set_DA

    # mp = False # True False
    # _map = multiproc_map if mp else map

    prm = container(set_DA)
    N1 = prm.N - 1
    N = prm.N
    Nx = prm.J
    jj = prm.obs_inds
    loc_rad = prm.local_scale
    loc_dic = prm.local_dic
    loc_fuc = prm.local_func
    R = prm.obs_cov.C

    E = ens.T
    # Decompose ensmeble
    mu = np.mean(E, 0)
    A = E - mu

    # Obs space variables
    y = obs

    HE = (H(E,set_DA)).T
    xo = np.mean(HE, axis=1, keepdims=True)
    Y = (HE - xo).transpose()
    # Y, xo = X ,x

    # Transform obs space
    Y = Y @ R.sym_sqrt_inv.T
    dy = (y - xo.ravel()) @ R.sym_sqrt_inv.T

    local = loc_setup((Nx,), (2,), jj, periodic=True)
    state_batches, obs_taperer = local(loc_rad, 'x2y', 0.05, loc_fuc)

    # for ii in state_batches:
    def local_analysis(ii):

        # Locate local obs
        jj, tapering = obs_taperer(ii)
        if len(jj) == 0:
            return E[:, ii], N1  # no update
        Y_jj = Y[:, jj]
        dy_jj = dy[jj]

        # Adaptive inflation
        za = N1  # effective_N(Y_jj, dy_jj, xN, g) if infl == '-N' else N1

        # Taper
        Y_jj *= np.sqrt(tapering)
        dy_jj *= np.sqrt(tapering)

        # Compute ETKF update
        if len(jj) < N:
            # SVD version
            V, sd, _ = svd0(Y_jj)
            d = pad0(sd ** 2, N) + za
            Pw = (V * d ** (-1.0)) @ V.T
            T = (V * d ** (-0.5)) @ V.T * np.sqrt(za)
        else:
            # EVD version
            d, V = eigh(Y_jj @ Y_jj.T + za * np.eye(N))
            T = V @ np.diag(d ** (-0.5)) @ V.T * np.sqrt(za)
            Pw = V @ np.diag(d ** (-1.0)) @ V.T
        AT = T @ A[:, ii]
        dmu = dy_jj @ Y_jj.T @ Pw @ A[:, ii]
        Eii = mu[ii] + dmu + AT
        return Eii, za

    # Run local analyses
    zz = []
    # result = _map(local_analysis, state_batches)
    # for ii, (Eii, za) in zip(state_batches, result):
    #     zz += [za]
    #     E[:, ii] = Eii

    # result = _map(local_analysis, state_batches)
    # for ii, (Eii, za) in zip(state_batches, result):
    #     zz += [za]
    #     E[:, ii] = Eii

    for ii in state_batches:
        Eii, za = local_analysis(ii)
        E[:, ii] = Eii
        zz += [za]

    return E.T

    # Eii, za = local_analysis(ii= state_batches[0])
