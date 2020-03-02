
from DA_methods import *
from basic_import import *


if __name__ == "__main__":
    # path = os.path.join(os.getcwd(), 'data_DA')
    path = os.path.abspath('E:\data_DA')

    set_DA = generate_set_DA(local_scale=1, T=1200, obs_type='linear', da_method='LETKF', a2=0.65, a1=0.5,
                             updateInterval=1,
                             data_saved_file='LETKF' +'_non')
    set_DA['data_saved_path'] = path
    set_DA['obs_sigma'] =0.1
    # set_DA['obs_new_sigma'] = True  # True False
    # set_DA['obs_new_sigma_value'] = 0.1
    set_DA['std_infl'] = 2
    set_DA['pbar'] = True
    set_DA['obs_type'] = 'non'
    set_DA['truth_new'] = False
    set_DA['change_seed2get_obs'] = False

    main_func(set_DA)


    # with (Path(path) / tmp['data_saved_file_name'] / 'stats_res').open('rb') as dill_file:
    #     kaka = dill.load(dill_file)
    #
    # print(kaka.rmse.f[0:(tmp['update_times']-1 )].mean())
    # print(kaka.rmv.f[0:(tmp['update_times'] - 1)].mean())






