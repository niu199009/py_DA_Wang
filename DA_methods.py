from LETKF import LETKF
# from LPF_GT import LPF_GT
# from all_settings import generate_settings,generare_metrics
from basic_import import *
import BMILorenz


def main_func(set_DA):

    def func(name):

        if name == 'LETKF':
            return LETKF
        elif name == 'LPF_GT':
            return LPF_GT
        else:
            print('error!!! check the name')

    da_func = func(set_DA['da_methods'])

    # to prepare settings of model, DA and metrics

    # to load DA settings
    set_DA = set_DA
    # to load metrics settings
    set_metrics = generare_metrics(set_DA)# load_data('set_metrics', set_DA)
    # to load model settings
    settings = generate_settings(set_DA)# yaml_load('settings', set_DA)

    indx_ob = set_DA['obs_inds']

    yaml_dump(
        settings,set_DA,
        name='settings')

    ##########################################################################
    # to prepare data including observations, truth and initial ensemble  (or
    # to load data directly)

    # to load data of truth
    if set_DA['truth_new']:
        truth_all = get_new_truth(set_DA)
    else:
        truth_all = load_data('truth', set_DA=set_DA)

    # to load data of observations
    if set_DA['obs_type'] != 'non':
        if set_DA['obs_new_sigma']:
            obs_all = change_std(truth_all, set_DA)
            set_DA['obs_sigma'] = set_DA['obs_new_sigma_value']
        else:
            if set_DA['truth_new']:
                set_DA['obs_sigma'], set_DA['obs_new_sigma_value'] = 1, 1
                obs_all = change_std(truth_all, set_DA)
            else:
                obs_all = load_data('observations', set_DA=set_DA)
    else:
        obs_all = load_data('observations', set_DA=set_DA)



    # to load data of initial ensemble
    ens_ini = load_data('ens_ini', set_DA=set_DA)

    if set_DA['obs_type'] == 'non':
        obs_all = obs_all.T

    # only for EnKF
    if set_DA['obs_type'] != 'non':
        observation_real_all = H_operator(truth_all, set_DA)
        obs_for_DA = H_operator(obs_all, set_DA)

    ##########################################################################
    # to initial ensemble of the model

    ensemble = []

    for n in range(set_DA['N']):
        # add an ensemble methods
        ensemble.append(BMILorenz.BMILorenz())
        ensemble[n].initialize('settings.yaml')
        ensemble[n].set_value('state', ens_ini[:, n])

    ##########################################################################
    # to run DA

    # to prepare forecast ensemble
    foreCastEnsemble = np.zeros((set_DA['J'], set_DA['N']))

    # to initial metrics
    stats_res = evaluation(set_DA, set_metrics)

    # to initial model time step
    truthModel = BMILorenz.BMILorenz()
    truthModel.initialize('settings.yaml')  # settings for creating truth

    if set_DA['pbar']:
        bar = tqdm(total=set_DA['T'])


    i = 0
    while truthModel.get_current_time() <= truthModel.get_end_time():

        if set_DA['pbar']:
            bar.update()

        # to get the current truth
        truth_now = truth_all[i, :]

        # to move to next time step
        truthModel.update()

        # to move forward model to next time step and get forecast from the ensemble of the model
        for n in range(set_DA['N']):
            ensemble[n].update()
            foreCastEnsemble[:, n] = ensemble[n].get_value('state')

        # to check conditions to run DA methods
        # the conditions include checking spinup time, update interval
        if i > set_DA['spinup'] and (
                i -
                set_DA['spinup']) > 0 and (
                (i -
                 set_DA['spinup']) %
                set_DA['updateInterval'] == 0):

            # print(i,np.where(stats_res.update_time_inds==i))

            if set_DA['da_methods'] == 'LETKF':
                # (1)to get current observations and (2)to inital observation covariance
                #    (1) observations without noise. This is for generating observation ensemble.
                if set_DA['obs_type'] != 'non':
                    obs_real_now = observation_real_all[i, :]
                else:
                    obs_real_now = obs_all[i, indx_ob]
                #    (2) to initial observation covariance and put it into the DA settings
                obs_cov = GaussRV(mu=obs_real_now, C=set_DA['obs_sigma'])
                set_DA['obs_cov'] = obs_cov

            if set_DA['da_methods'] == 'LPF_GPR':
                set_DA['time'] = i

                def get_final_list(set_DA, i):
                    GP_list = load_GP(set_DA, i // 1000)
                    return get_GP_list(i, set_DA['obs_num'], GP_list)

                set_DA['GPR_list'] = get_final_list(set_DA, i)

            # prior assessment
            if 'f' in set_DA['fau']:
                stats_res.ens_valuate(
                    i,
                    truth=truth_now,
                    type='forecast',
                    E=foreCastEnsemble.T)

                # a test here
                # print('i',i)
                # t = np.where(set_DA['update_time_inds'] == i)
                # print('mu ',stats_res.mu.f[t,0])
                # print('truth: ',truth_now[0])
                # print('x1_mean: ',(stats_res.x1.f[t,:]).mean() )
                # print('error: ',stats_res.err.f[t,0])
                # print('mu-truth: ',stats_res.mu.f[t,0]-truth_now[0])


            # to run DA
            analysesEnsemble = da_func(
                foreCastEnsemble, obs_for_DA[i, :], H_operator, set_DA)

            # to inflate enssemble
            if set_DA['infl_not']:
                pass
            else:
                HE = H_operator(analysesEnsemble.T,set_DA)
                #foreCastEnsemble_var = foreCastEnsemble.var(axis=1)
                analysesEnsemble = infl_ens(set_DA,analysesEnsemble, obs_for_DA[i, :], HE)
                # print(analysesEnsemble.var(axis=1)/foreCastEnsemble_var)

            # posterior assessment
            if 'a' in set_DA['fau']:
                stats_res.ens_valuate(
                    i,
                    truth=truth_now,
                    type='analysis',
                    E=analysesEnsemble.T)


            # to check the outliers of the ensemble
            np.clip(analysesEnsemble, -10, 20, out=analysesEnsemble)

            for jj in range(set_DA['N']):
                ensemble[jj].set_value('state', analysesEnsemble[:, jj])

            if set_DA['save_ens_or_not']:
                data_saved(
                    analysesEnsemble, set_DA,
                    data_name='ens' + str(i)
                )

            if 'u' in set_DA['fau']:
                stats_res.ens_valuate(
                    i,
                    truth=truth_now,
                    type='all',
                    E=analysesEnsemble.T)

            # truth_now = truth_all[i, :]

        else:
            if i > set_DA['spinup']:
                if set_DA['save_ens_or_not']:
                    data_saved(
                        analysesEnsemble, set_DA,
                        data_name='ens' + str(i)
                    )
                if 'u' in set_DA['fau']:
                    stats_res.ens_valuate(
                        i, truth=truth_now, type='all', E=foreCastEnsemble.T)


        i = i + 1

    if set_DA['pbar']:
        bar.close()

    data_saved(
        stats_res, set_DA,
        data_name='stats_res'
    )

    # time.sleep(30)


if __name__ == '__main__':
    pass

