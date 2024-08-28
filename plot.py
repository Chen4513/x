from utils.figure_util import ResultReaderFromFile, curve_average, curve_average2
import matplotlib.pylab as plt
import matplotlib
import glob
from copy import deepcopy
import numpy as np

font = {'family': 'Times New Roman'}
matplotlib.rc('font', **font)
matplotlib.rc('axes', titlesize=18)  # Font size for axes titles
matplotlib.rc('axes', labelsize=15)  # Font size for x and y labels
matplotlib.rc('xtick', labelsize=15)  # Font size for x tick labels
matplotlib.rc('ytick', labelsize=15)  # Font size for y tick labels
matplotlib.rc('legend', fontsize=15)

learning_curves={'x1':{}, 'x5': {}}
smooth_weight = 0.4

alpha = 0.2

envs = {'Environment 1':'x1', 'Environment 2':'x5'}
types = ['sd', 'ds']
run_times = 3


# environment 1 data
base_dir = '/home/sunnylin/torchrl-TS-LLM/Experiments/'

learning_curves['x1'] = {types[0]:{}, types[1]: {}}

for i, t in enumerate(types):
    learning_curves['x1'][t] = {}
    curve_data = [None] * run_times
    for n in range(run_times):
        dir = base_dir + '06err-' + t + '-1query_trial' + str(n+1) + '/AdvisedTrainer*'

        # Using glob to find files matching the pattern
        print(dir)
        files = glob.glob(dir)
        print(files)
        curve_data[n] = ResultReaderFromFile(files[0])
        curve_data[n].read_results(smooth_weight = 0)
    # print(len(curve_data))
    # print(curve_data[0].episode_reward_mean)
    common_timesteps, average_performance, performance_variance, max_performance, min_performance = curve_average2(deepcopy(curve_data))
    learning_curves['x1'][t]['common_ts'] = common_timesteps
    learning_curves['x1'][t]['avg_performance'] = average_performance
    learning_curves['x1'][t]['performance_var'] = performance_variance
    learning_curves['x1'][t]['max_performance'] = max_performance
    learning_curves['x1'][t]['min_performance'] = min_performance

# environment 2 data
learning_curves['x5'] = {types[0]:{}, types[1]: {}}
for i, t in enumerate(types):
    learning_curves['x5'][t] = {}
    curve_data = [None] * run_times
    for n in range(run_times):
        dir = base_dir + '06err-' + t + '-5query_trial' + str(n+1) + '/AdvisedTrainer*'

        # Using glob to find files matching the pattern
        files = glob.glob(dir)
        curve_data[n] = (ResultReaderFromFile(files[0]))
        curve_data[n].read_results(smooth_weight = 0)

    common_timesteps, average_performance, performance_variance, max_performance, min_performance = curve_average2(deepcopy(curve_data))
    learning_curves['x5'][t]['common_ts'] = common_timesteps
    learning_curves['x5'][t]['avg_performance'] = average_performance
    learning_curves['x5'][t]['performance_var'] = performance_variance
    learning_curves['x5'][t]['max_performance'] = max_performance
    learning_curves['x5'][t]['min_performance'] = min_performance


learning_curves['x0'] = {}
curve_data = [None] * run_times
for n in range(run_times):
    dir = base_dir + 'hopper_default_trial' + str(n+1) + '/AdvisedTrainer*'

    # Using glob to find files matching the pattern
    files = glob.glob(dir)
    curve_data[n] = (ResultReaderFromFile(files[0]))
    curve_data[n].read_results(smooth_weight = 0)

common_timesteps, average_performance, performance_variance, max_performance, min_performance = curve_average2(deepcopy(curve_data))
learning_curves['x0']['common_ts'] = common_timesteps
learning_curves['x0']['avg_performance'] = average_performance
learning_curves['x0']['performance_var'] = performance_variance
learning_curves['x0']['max_performance'] = max_performance
learning_curves['x0']['min_performance'] = min_performance


print(learning_curves.keys())
plt.figure(figsize=(18,9))

legends = ['5-query potential-difference', '1-query potential-difference', '5-query direct-score', '1-query direct-score', 'Default Rewards']

#for n, (env, env_brief) in enumerate(envs.items()):
    #plt.subplot(1,2,n+1)
    
for type_ in types:
    for env_brief in ['x5', 'x1']:
        line, = plt.plot(learning_curves[env_brief][type_]['common_ts'],  learning_curves[env_brief][type_]['avg_performance'])

plt.plot(learning_curves['x0']['common_ts'],  learning_curves['x0']['avg_performance'])

plt.title('Flipped 60% Rankings')
plt.ylabel("Rewards", labelpad=0)

plt.xlabel("Env steps", labelpad=0)
plt.xlim((0,500000))
plt.ylim((-0.1, 3000))
plt.xticks([0, 500000], ['0', '5e5'])
plt.yticks([0, 3000], ['0', '3000'])

# for n, (env, env_brief) in enumerate(envs.items()):
#     for type_ in types:
for type_ in types:
    for env_brief in ['x5', 'x1']:
        plt.fill_between(learning_curves[env_brief][type_]['common_ts'], learning_curves[env_brief][type_]['max_performance'],
                learning_curves[env_brief][type_]['min_performance'], alpha=alpha)

plt.fill_between(learning_curves['x0']['common_ts'], learning_curves['x0']['max_performance'],
        learning_curves['x0']['min_performance'], alpha=alpha)
        
# plt.figtext(0.5, 0.92, 'Environment 1', ha='center', va='center', fontsize=18, fontweight='bold', transform=plt.gcf().transFigure)
# plt.figtext(0.5, 0.65, 'Environment 2', ha='center', va='center', fontsize=18, fontweight='bold', transform=plt.gcf().transFigure)
# plt.figtext(0.5, 0.33, 'Environment 3', ha='center', va='center', fontsize=18, fontweight='bold', transform=plt.gcf().transFigure)

plt.figlegend(legends, loc='upper center', bbox_to_anchor=(0.5, 1), ncol = len(legends))
plt.subplots_adjust(top=0.88, bottom=0.1, left=0.05, right=0.95)  # Adjust these values as needed
# plt.subplots_adjust()
# plt.tight_layout(pad=3)
plt.show()