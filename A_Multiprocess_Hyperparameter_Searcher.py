import gymenv_v2
from gymenv_v2 import make_multiple_env
from gymenv_v2 import timelimit_wrapper, GurobiOriginalEnv
import numpy as np
import pandas as pd
import tensorflow as tf
import NeuralNetworkModels
import importlib
import matplotlib
from matplotlib import pyplot as plt
import wandb

import multiprocessing
import time


## Helper function for testing
def make_gurobi_env(load_dir, idx, timelimit):
	print('loading training instances, dir {} idx {}'.format(load_dir, idx))
	A = np.load('{}/A_{}.npy'.format(load_dir, idx))
	b = np.load('{}/b_{}.npy'.format(load_dir, idx))
	c = np.load('{}/c_{}.npy'.format(load_dir, idx))
	env = timelimit_wrapper(GurobiOriginalEnv(A, b, c, solution=None, reward_type='obj'), timelimit)
	return env

## Training data setup
# Setup: You may generate your own instances on which you train the cutting agent.
custom_config = {
    "load_dir"        : 'instances/randomip_n60_m60',   # this is the location of the randomly generated instances (you may specify a different directory)
    "idx_list"        : list(range(20)),                # take the first 20 instances from the directory
    "timelimit"       : 50,                             # the maximum horizon length is 50
    "reward_type"     : 'obj'                           # DO NOT CHANGE reward_type
}

# Easy Setup: Use the following environment settings. We will evaluate your agent with the same easy config below:
easy_config = {
    "load_dir"        : 'instances/train_10_n60_m60',
    "idx_list"        : list(range(10)),
    "timelimit"       : 50,
    "reward_type"     : 'obj'
}

# Hard Setup: Use the following environment settings. We will evaluate your agent with the same hard config below:
hard_config = {
    "load_dir"        : 'instances/train_100_n60_m60',
    "idx_list"        : list(range(99)),
    "timelimit"       : 50,
    "reward_type"     : 'obj'
}


def run_training_loop(argument):

    ### Choose easy or hard setup
    dif = "easy" # easy OR hard
    # create env
    env = make_multiple_env(**easy_config)

    ### TRAINING
    run = wandb.init(project="finalproject", entity="ezequiel-piedras",
                     tags=["training-{}".format(dif)],
                     name="{dif}-{a:.3f}".format(dif=dif,a=argument))

    # Initialize parameters
    actor_lr = 1e-2  # learning rate for actor
    numtrajs = 10  # num of trajectories from the current policy to collect in each iteration
    iterations = 50  # total num of iterations
    gamma = .1  # discount
    sigma = 10.
    state_dim = 60

    # Instantiate actor model
    actor = NeuralNetworkModels.Policy(state_dim, actor_lr, argument)

    # Training reward history
    r_history = []

    for ite in range(iterations):

        # For recording trajectories played in this iteration
        OBSERVED_STATES = []
        ACTIONS = []
        MC_VALUES = []

        # Record avg. iteration reward
        ravgit = np.zeros((numtrajs))

        for traj in range(numtrajs):

            observed_states = []
            actions = []
            rewards = []

            s = env.reset()  # samples a random instance every time env.reset() is called
            done = False

            t = 0
            repisode = 0

            while not done:
                # Run policy
                action_space_size = s[-1].size  # size of action space changes every time
                prob = actor.compute_prob(s)
                prob = prob.flatten()
                prob /= np.sum(prob)
                action = np.random.choice(action_space_size, p=prob, size=1)
                new_s, r, done, _ = env.step(action)

                # Save the move
                observed_states.append(s)
                actions.append(action[0])
                rewards.append(r)

                # Update trackers
                s = new_s
                t += 1
                repisode += r

            # Record trajectory
            V_hats = NeuralNetworkModels.discounted_rewards(np.array(rewards), gamma)
            OBSERVED_STATES.extend(observed_states)
            ACTIONS.extend(actions)
            MC_VALUES.extend(V_hats)

            # Log training reward
            ravgit[traj] = repisode

        # Update policy model
        ES = MC_VALUES + np.random.normal(0, 1, len(MC_VALUES)) / sigma
        progr = actor.train(OBSERVED_STATES, ACTIONS, ES)
        # Print training progress
        ravgit = np.mean(ravgit)
        r_history.append(ravgit)
        wandb.log({"Training reward ({} config)".format(dif): ravgit})
        print("Argu: {argu:2.3f}    It: {ite:2.0f}   Loss: {progr: .3f}   R: {repisode:.3f}" \
              .format(argu=argument, ite=ite, progr=progr, repisode=ravgit))

    r_history = pd.DataFrame(r_history)
    r_history['MAvg'] = r_history.rolling(5, min_periods=5).mean()
    #r_history.plot(figsize=(10, 5))


    ### Also test the policy function in the random-generated ILPs
    r_test = np.zeros((10))
    for env_i in range(10):
        env = make_gurobi_env('instances/randomip_n60_m60', env_i, 100)
        s = env.reset()
        d = False
        t = 0
        rtestit = 0.
        while not d:
            # Run policy
            action_space_size = s[-1].size  # size of action space changes every time
            prob = actor.compute_prob(s)
            prob = prob.flatten()
            prob /= np.sum(prob)
            action = np.random.choice(action_space_size, p=prob, size=1)
            s, r, d, _ = env.step(action)
            #print('step', t, 'reward', r, 'action space size', s[-1].size)
            rtestit += r
            t += 1
        r_test[env_i] = rtestit
        wandb.log({"Test reward ({} config)".format(dif): rtestit})

    print("Argu: {argu:2.3f}. Average reward on test set: {avg:.3f}".\
          format(argu=argument,avg=np.mean(r_test)))
    wandb.finish()

    return r_history["MAvg"]


def test(l):

    return pd.Series([l*i for i in range(10)])


# Only to be executed by master process and not by slave processes
if __name__ == '__main__':

    start_time = time.time()

    pool = multiprocessing.Pool(processes=4)

    # List of hyperparameter values to try
    hyperparameter_list = [8,16,32,64]

    training_res = pool.map(run_training_loop, hyperparameter_list)

    training_results = pd.DataFrame(training_res).T
    training_results.columns = map(str,hyperparameter_list)

    print(training_results)

    training_results.plot(figsize = (10,5))
    plt.title("Two f_0. LSTM(8),Dense(h,tanh),Dense(h). Dif h")
    plt.show()

    end_time = time.time()
    secs = end_time - start_time
    print("Finished multiprocessing in {secs:.0f} secs.".format(secs=secs))

