import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import shape
from copy import deepcopy
import time
start_time = time.time()
# np.random.seed(72)
# np.random.seed(32)

np.random.seed(9)
# np.random.seed(10)
# np.random.seed(16)
# np.random.seed(28)
# np.random.seed(12)
energy_state = np.arange(0,24.2,0.2)
# energy_state = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]
SCALAR = 1.0
# energy_state = [0, 0.2, 0.4, 0.6, 0.8, 1]
energy_state = [round(x * SCALAR,2) for x in energy_state]
action_list = [0.2, 0.4, 0.6, 0.8, 1, 0, -0.2, -0.4, -0.6, -0.8, -1]
action_list = [x * SCALAR for x in action_list]
phi = [10, 10, 10, 10, 10]  # degradation cost
# phi = [5, 10, 15, 20, 25]  # degradation cost
E = len(energy_state)  # energy space
A = len(action_list)  # action space
eta = 0.9  ### Nik's comment: haven't add efficiency 

################### Load price data ###############
pdata_all = []
with open("WEST.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]

for sublist in rows:
    for item in sublist:
        pdata_all.append(float(item))

############## Discretized price #################
### Nik's comment: change price upper and lower bound according to locations
# breakpoint = [-800, 0, 20, 40, 60, 80, 100, 600]
breakpoint = np.arange(0, 200, 2)
breakpoint = np.insert(breakpoint, 0, -1700)
breakpoint = np.insert(breakpoint, len(breakpoint), 4000)
# breakpoint = [10, 15, 17.5, 20, 22.5, 25, 27.5, 30, 32.5, 35, 37.5, 40, 50, 60, 70, 80, 90, 100, 150]


def discretize_price(price, breakpoint):
    T = len(price)
    N = len(breakpoint)  # number of discretized class
    signal = -1 * np.ones((T, 1))  # discretized signal
    for i in range(T):
        for j in range(N - 1):
            if breakpoint[j] <= price[i] and price[i] < breakpoint[j + 1]:
                signal[i] = j
        if signal[i] == -1:
            signal[i] = N - 1
    return signal


################ Initilize Q tabular ###############
P = len(breakpoint)  # discretized price space
# agent = DQNAgent(E, P, A)
Q = np.zeros([E, P, A], dtype=float)
GAMMA = 0.99  # discount

############ piecewise linear degradation cost
### Nik's comment modified to degradation cost only in discharge
def degradation(action):
    cost = 0
    # power = abs(action)
    power = abs(min(0, action))
    interval = [0.2, 0.4, 0.6, 0.8, 1]
    interval = [x * SCALAR for x in interval]
    if power <= interval[0]:
        cost += power * phi[0]
    elif power <= interval[1]:
        cost = (interval[0]) * phi[0] + (power - interval[0]) * phi[1]
    elif power <= interval[2]:
        cost = (
            (interval[0]) * phi[0]
            + (interval[1] - interval[0]) * phi[1]
            + (power - interval[1]) * phi[2]
        )
    elif power <= interval[3]:
        cost = (
            interval[0] * phi[0]
            + (interval[1] - interval[0]) * phi[1]
            + (interval[2] - interval[1]) * phi[2]
            + (power - interval[2]) * phi[3]
        )
    else:
        cost = (
            interval[0] * phi[0]
            + (interval[1] - interval[0]) * phi[1]
            + (interval[2] - interval[1]) * phi[2]
            + (interval[3] - interval[2]) * phi[3]
            + (power - interval[3]) * phi[4]
        )
    return cost


################## Instant reward
def reward(action, state, p_signal):
    if action < 5:  # charge
        reward = (-p_signal) * action_list[action] - degradation(action_list[action])
    elif action == 5:  # hold
        reward = 0
    else:  # discharge
        reward = (p_signal) * (-action_list[action]) - degradation(action_list[action])
    return reward



########## availabel actions at each energy level (for E_max = 3)
"""def available_actions(state):
    # action: (charge) 0, 1, 2, 3, 4, (hold) 5, (discharge) 6, 7, 8, 9, 10
    if state==0: #0
        action= [0,1,2,3,4,5] 
    elif state == 1: #0.2
        action = [0,1,2,3,4,5,6] 
    elif state == 2: #0.4
        action = [0,1,2,3,4,5,6,7]
    elif state == 3: #0.6
        action = [0,1,2,3,4,5,6,7,8]
    elif state == 4: #0.8
        action = [0,1,2,3,4,5,6,7,8,9]
    elif state == 11: #2.2
        action = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10]
    elif state == 12: #2.4
        action = [0, 1, 2, 5, 6, 7, 8, 9, 10]
    elif state == 13: #2.6
        action = [0, 1, 5, 6, 7, 8, 9, 10]
    elif state == 14: #2.8
        action = [0, 5, 6, 7, 8, 9, 10]
    elif state == 15: #3
        action = [5, 6, 7, 8, 9, 10]
    else: #1
        action = [0,1,2,3,4,5,6,7,8,9,10]

    return action"""

######### availabel actions at each energy level (for E_max = 24)
def available_actions(state):
    # action: (charge) 0, 1, 2, 3, 4, (hold) 5, (discharge) 6, 7, 8, 9, 10
    if state == 0:  # 0
        action = [0, 1, 2, 3, 4, 5]
    elif state == 1:  # 0.2
        action = [0, 1, 2, 3, 4, 5, 6]
    elif state == 2:  # 0.4
        action = [0, 1, 2, 3, 4, 5, 6, 7]
    elif state == 3:  # 0.6
        action = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    elif state == 4:  # 0.8
        action = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    elif state == 116:  # 1.2
        action = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10]
    elif state == 117:  # 1.4
        action = [0, 1, 2, 5, 6, 7, 8, 9, 10]
    elif state == 118:  # 1.6
        action = [0, 1, 5, 6, 7, 8, 9, 10]
    elif state == 119:  # 1.8
        action = [0, 5, 6, 7, 8, 9, 10]
    elif state == 120:  # 2
        action = [5, 6, 7, 8, 9, 10]
    else:  # 1
        action = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    return action

########## availabel actions at each energy level (for E_max = 2)
# def available_actions(state):
#     # action: (charge) 0, 1, 2, 3, 4, (hold) 5, (discharge) 6, 7, 8, 9, 10
#     if state == 0:  # 0
#         action = [0, 1, 2, 3, 4, 5]
#     elif state == 1:  # 0.2
#         action = [0, 1, 2, 3, 4, 5, 6]
#     elif state == 2:  # 0.4
#         action = [0, 1, 2, 3, 4, 5, 6, 7]
#     elif state == 3:  # 0.6
#         action = [0, 1, 2, 3, 4, 5, 6, 7, 8]
#     elif state == 4:  # 0.8
#         action = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#     elif state == 6:  # 1.2
#         action = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10]
#     elif state == 7:  # 1.4
#         action = [0, 1, 2, 5, 6, 7, 8, 9, 10]
#     elif state == 8:  # 1.6
#         action = [0, 1, 5, 6, 7, 8, 9, 10]
#     elif state == 9:  # 1.8
#         action = [0, 5, 6, 7, 8, 9, 10]
#     elif state == 10:  # 2
#         action = [5, 6, 7, 8, 9, 10]
#     else:  # 1
#         action = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

#     return action


########## availabel actions at each energy level (for E_max = 1)
# def available_actions(state):
#     # action: (charge) 0, 1, 2, 3, 4, (hold) 5, (discharge) 6, 7, 8, 9, 10
#     # [0.2, 0.4, 0.6, 0.8, 1, 0, -0.2, -0.4, -0.6, -0.8, -1]
#     if state == 0:  # 0
#         action = [0, 1, 2, 3, 4, 5]
#     elif state == 1:  # 0.2
#         action = [0, 1, 2, 3, 5, 6]
#     elif state == 2:  # 0.4
#         action = [0, 1, 2, 5, 6, 7]
#     elif state == 3:  # 0.6
#         action = [0, 1, 5, 6, 7, 8]
#     elif state == 4:  # 0.8
#         action = [0, 5, 6, 7, 8, 9]
#     elif state == 5:  # 1
#         action = [5, 6, 7, 8, 9, 10]
#     else:  # 1
#         action = [5]  # hold

#     return action


########### epsilon-greedy
def sample_next_action(available_act, current_state, prob, current_price):
    # use epsilon greedy to sample the next action
    if prob <= GAMMA:
        m = len(available_act)
        q_val = []
        for i in range(m):
            q_val.append(Q[current_state, current_price, available_act[i]])
        max_val = np.argmax(q_val)
        next_action = available_act[max_val]
    else:
        next_action = int(np.random.choice(available_act, 1))
    return next_action


############### update energy level
### Nik's comment: add efficiency, haven't done yet
def update_state(current_state, action):
    # update battery state
    state_change = int(action_list[action] / (0.2 * SCALAR))
    return current_state + state_change


############### Key function of Q learning: receive rewards, update states,
############### update Q table
def update(t, price, current_state, current_price, action, next_price, stepsize):
    # update Q value matrix
    reward_val = reward(action, current_state, price[t])
    # print("current state", current_state, "action", action, "price", price[t], "reward:", reward_val)
    # next energy level
    next_state = update_state(current_state, action)
    # possible action
    next_pos_act = available_actions(next_state)

    Q[current_state, current_price, action] = (1 - stepsize) * Q[
        current_state, current_price, action
    ] + stepsize * (
        reward_val + GAMMA * np.max(Q[next_state, next_price, next_pos_act])
    )

    return next_state


def replay(t, price, current_state, current_price, action, next_price, stepsize):
    # replay function
    reward_val = reward(action, current_state, price[t])

    next_state = update_state(current_state, action)

    next_pos_act = available_actions(next_state)

    Q[current_state, current_price, action] = (1 - stepsize) * Q[
        current_state, current_price, action
    ] + stepsize * (
        reward_val + GAMMA * np.max(Q[next_state, next_price, next_pos_act])
    )

    return


############# Main function ############
for i in range(1):
    # simulation length
    ### Nik's comment: changed to 5-min
    pdata = pdata_all[0:210528]
    n_episodes = len(pdata)
    # n_episodes = 200

    action_history = []
    energy_history = []

    current_state = 0
    print("current state: %d" % current_state)

    # ||Q_t - Q_{t-1}||_2
    scores = []

    # random number for epsilon-greedy
    rand = np.random.uniform(0, 1, (n_episodes, 1))

    ##################### Price #######################
    signal = discretize_price(pdata, breakpoint)
    ###################################################

    # Qlearning
    for episode in range(n_episodes - 1):
        if episode % 1000 == 0:
            print("Finish", episode, "/", n_episodes)
            print("--- %s seconds ---" % (time.time() - start_time))

        # available action
        available_act = available_actions(current_state)

        # epsilon greedy
        action = sample_next_action(
            available_act, current_state, rand[episode], int(signal[episode])
        )
        action_history.append(action)
        energy_history.append(energy_state[current_state])

        # take the action and update Q func
        stepsize = 0.5 / (0.001 * episode + 1)
        preQ = deepcopy(Q)
        new_state = update(
            episode,
            pdata,
            current_state,
            int(signal[episode]),
            action,
            int(signal[episode + 1]),
            stepsize,
        )

        ################# replay and update other value in Q table ############
        for act in available_act:
            if act != action:  # not the implemented one
                replay(
                    episode,
                    pdata,
                    current_state,
                    int(signal[episode]),
                    act,
                    int(signal[episode + 1]),
                    stepsize,
                )
        for replay_state in np.arange(0, E):
            if replay_state != current_state:
                replay_act = available_actions(replay_state)
                for act in replay_act:
                    replay(
                        episode,
                        pdata,
                        replay_state,
                        int(signal[episode]),
                        act,
                        int(signal[episode + 1]),
                        stepsize,
                    )
        #######################################################################

        current_state = deepcopy(new_state)

        # check for convergence condition
        scores.append(np.linalg.norm(preQ - Q))
        if episode > 50000 and scores[-1] < 10 ** (-5):
            break
    # print ('Score:', str(score))
print(episode)
print(Q)


################### BEGIN: if we seperate training and testing #################
### Nik's comment: changed to 5-min
testprice = pdata_all[210528:]
testlen = len(testprice)
profit = np.zeros((testlen, 1))
total_profit = np.zeros((testlen, 1))
power_history = np.zeros((testlen, 1))
energy_history = np.zeros((testlen, 1))
current_state = 3
rand_test = np.random.uniform(0, 1, (testlen, 1))
signal_test = discretize_price(testprice, breakpoint)
for i in range(testlen):
    # available action
    available_act = available_actions(current_state)

    # epsilon greedy
    action = sample_next_action(
        available_act, current_state, rand_test[i], int(signal_test[i])
    )

    # state transition
    next_state = update_state(current_state, action)

    current_state = deepcopy(next_state)

    power_history[i] = action_list[action]

    energy_history[i] = energy_state[current_state]

    if action < 5:  # charge
        profit[i] = -testprice[i] * action_list[action] - degradation(
            action_list[action]
        )
    elif action == 5:  # hold
        profit[i] = 0
    else:  # discharge
        profit[i] = -testprice[i] * action_list[action] - degradation(
            action_list[action]
        )

    total_profit[i] = np.sum(profit[0:i])
################### END: if we seperate training and testing ####################

############# plot accumulated reward
# profit = np.zeros((n_episodes - 1, 1))
# total_profit = np.zeros((n_episodes - 1, 1))
# power_history = np.zeros((n_episodes - 1, 1))
# for i in range(n_episodes - 1):
#     power_history[i] = action_list[action_history[i]]
#     if action_history[i] < 5:  # charge
#         profit[i] = -pdata[i] * action_list[action_history[i]] - degradation(
#             action_list[action_history[i]]
#         )
#     elif action_history[i] == 5:  # hold
#         profit[i] = 0
#     else:  # discharge
#         profit[i] = -pdata[i] * action_list[action_history[i]] - degradation(
#             action_list[action_history[i]]
#         )

#     total_profit[i] = np.sum(profit[0:i])

# plt.figure("""Total profit""")
# plt.grid()
# plt.plot(
#     total_profit, "r", label="Average reward"
# )  # set reward2; 'instant reward', set reward1
# # plt.plot(total_profit_average,'b',label='Average reward')
# plt.legend()
# plt.xlabel("Time")
# plt.ylabel("Total reward")

# ###############################################################################
# t = np.arange(0, 100)
# fig, ax1 = plt.subplots()
# color = "tab:red"
# ax1.set_xlabel("time (s)")
# ax1.set_ylabel("price", color=color)
# ax1.plot(t, testprice[3100:3200], "-^", color=color)
# ax1.tick_params(axis="y", labelcolor=color)
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# color = "tab:blue"
# ax2.set_ylabel("response", color=color)  # we already handled the x-label with ax1
# ax2.plot(t, power_history[3100:3200], "-^", color=color)
# ax2.tick_params(axis="y", labelcolor=color)
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()

# ########### paste to Excel
A1 = np.reshape(power_history,len(power_history)).tolist()
A2 = np.reshape(energy_history,len(energy_history)).tolist()
A3 = np.reshape(profit,len(profit)).tolist()
A4 = np.reshape(total_profit,len(total_profit)).tolist()
np.save('result/Q_WEST.npy', Q)
 
df = pd.DataFrame({'Power': A1, 'Energy': A2, 'Profit': A3, 'Total Profit': A4})
df.to_csv('result/WEST.csv')
print("Final profit:", total_profit[-1])
