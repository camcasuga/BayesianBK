import numpy as np

# append theta file below with old theta file
theta_file_100d = "trainingdata4p100d_theta.dat"
theta_file_121d = "mve/orthLHS/121d/theta.dat"
new_theta_file = "mve/hybrid/theta.dat"

# complete old 100d (10.0 GeV, 50.0] training set with [2.0, 10.0] GeV training set (append to the right of the old training set)
#train_file_100d_1 = "trainingdata4p100d_train.dat"
#train_file_100d_2 = "trainagainpt1.dat"
#new_train_file_100d = "100d_train.dat"
# bad idea! ordering is different? how to better append this huhu
# okay, I just generated the whole experimental set na lang

# append 100d complete training set to existing 121d training set
train_file_121d = "mve/orthLHS/121d/train.dat"
train_file_100d = "trainingdata4p100_all.dat"
new_train_file = "mve/hybrid/train.dat"

theta_array_100d = np.loadtxt(theta_file_100d)
theta_array_121d = np.loadtxt(theta_file_121d)
new_theta_array = np.concatenate((theta_array_121d, theta_array_100d), axis = 0)

np.savetxt(new_theta_file, new_theta_array)

train_array_100d = np.loadtxt(train_file_100d)
train_array_121d = np.loadtxt(train_file_121d)
new_train_array = np.concatenate((train_array_121d, train_array_100d), axis = 0)

np.savetxt(new_train_file, new_train_array)