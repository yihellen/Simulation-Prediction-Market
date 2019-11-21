import imageio
import glob
import numpy as np
images = []
for true_prob in np.arange(0, 1, 0.01):
    filename = 'sim_res_budget_0.3_iter_500_prob_{}_*/profit_init_belief*.png'\
        .format(true_prob)
    file_list = glob.glob(filename)
    for file in file_list:
        images.append(imageio.imread(file))
imageio.mimsave('belief.gif', images)
