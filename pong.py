# Learning to play Pong with Reinforcement Learning (using a fully connected neural network)
# Dominic Weber

# The chosen method
# -is model free
# -is policy based (policy gradient method)
# -is an on-policy method
# -uses a stochastic policy (as opposed to a deterministic one)

# IMPORTS
import numpy as np
import gym
import matplotlib.pyplot as plt

# 3 LAYER NEURAL NETWORK STRUCTURE <-- SET VALUES FOR LAYER_2_NEURONS, OUTPUTS
inputs = 6400  # 80 * 80 pixels (refer to 'preprocess' function for details)
layer_2_neurons = 250
outputs = 3 # Note: Can only choose between 1, 2, and 3 outputs, as this part of the code is not written in a fully generalized way.

# WEIGHT UPDATE (OPTIMIZER) AND WEIGHT INITIALIZATION DEFINITIONS <-- SET PARAMETERS
batch_size = 10
optimizer = 'ADAM' # momentum, RMSprop, ADAM
alpha = 1e-2 # Learning rate (or step size): 1e-2 (for momentum and ADAM), 1e-3 (for RMSprop). Note: As gradients were summed over batches and then divided by the batch size (i.e. gradients were averaged over the batches), the required learning rate is effectively dependent on the batch size.
decay_rate = 0.99
decay_rate_1 = 0.9 # as per ADAM paper
decay_rate_2 = 0.999 # as per ADAM paper
epsilon = 1e-8 # To avoid division by zero
momentum = 0.5 # Typical values according to Goodfellow-Bengio-Courville are 0.5, 0.9, and 0.99
initialization = 4 # determines which initialization method to use (Note: Only methods 2 to 4 have been implemented, not method 1. Only method 4 has been tested.)

# REINFORCEMENT LEARNING DEFINITIONS
actions_three = [2,5,0] # Used for the scenario where there are 3 outputs in the network
actions_two = [2,5] # Used for the scenario where there are 2 outputs in the network
discount_factor = 0.99 # For discounting future rewards

# INITIALIZATIONS FOR NEURAL NETWORK
# Input node values, hidden node values, outputs node values, and log policy gradient wrt logits o_3 (dE/do_3, where E = log(pi), which is also equal to the negative of the cross entropy loss function i.e. sum_over_classes(ylog(y_hat)) which is equal to log(y_hat) when using one-hot encoded vectors)
frame_previous = None
x_1_list = [] # input node values
x_2_list = [] # hidden node values
y_array = np.eye(outputs,outputs) # output node values
dEdo_3_array = np.zeros((outputs,1)) # gradient wrt o_3

# Weights
#if initialization == 1:
    # Option 1: Weights initilized using a truncated normal distribution. According to Deep Learning IU lecture notes this is OK for shallow networks with activations that are non-zero for positive and negative arguments. The latter condition is not met here, however. Not implemented in this code.
if initialization == 2:
    # Option 2: Initialization with the 'commonly used heuristic' used in the paper by Glorot and Bengio.
    lower_bound1 = -1 / (np.sqrt(inputs))
    upper_bound1 = 1 / (np.sqrt(inputs))
    lower_bound2 = -1 / (np.sqrt(layer_2_neurons))
    upper_bound2 = 1 / (np.sqrt(layer_2_neurons))
    w_1_2 = np.random.uniform(lower_bound1, upper_bound1, (layer_2_neurons, inputs+1))
    w_1_2[:, inputs] = np.zeros(layer_2_neurons)
    w_2_3 = np.random.uniform(lower_bound2, upper_bound2, (outputs, layer_2_neurons+1))
    w_2_3[:, layer_2_neurons] = np.zeros(outputs)
if initialization == 3:
    # Option 3: Normalized initialization as per Glorot and Bengio (shown to be effective at reducing gradient variance for deep networks with tanh() activtion functions)
    lower_bound1 = -np.sqrt(6) / (np.sqrt(inputs + layer_2_neurons))
    upper_bound1 = np.sqrt(6) / (np.sqrt(inputs + layer_2_neurons))
    lower_bound2 = -np.sqrt(6) / (np.sqrt(layer_2_neurons + outputs))
    upper_bound2 = np.sqrt(6) / (np.sqrt(layer_2_neurons + outputs))
    w_1_2 = np.random.uniform(lower_bound1, upper_bound1, (layer_2_neurons, inputs+1))
    w_1_2[:, inputs] = np.zeros(layer_2_neurons)
    w_2_3 = np.random.uniform(lower_bound2, upper_bound2, (outputs, layer_2_neurons+1))
    w_2_3[:, layer_2_neurons] = np.zeros(outputs)
if initialization == 4:
    # Option 4: Initialization as per He et al. This approach was specifically developed for ReLU activations according to IU Deep Learning lecture notes. The paper by He et al shows this can have faster convergance than option 3 on deep networks.
    w_1_2 = np.random.normal(0, np.sqrt(2/inputs), (layer_2_neurons,inputs+1))
    # print('w_1_2: ', type(w_1_2), w_1_2.shape, w_1_2.dtype)
    w_1_2[:,inputs] = np.zeros(layer_2_neurons)
    print('w_1_2: ', type(w_1_2), w_1_2.shape, w_1_2.dtype, w_1_2)
    w_2_3 = np.random.normal(0, np.sqrt(2/layer_2_neurons), (outputs, layer_2_neurons+1))
    w_2_3[:, layer_2_neurons] = np.zeros(outputs)
    print('w_2_3: ', type(w_2_3), w_2_3.shape, w_2_3.dtype, w_2_3)
w = {}
w['w_1_2'] = w_1_2 # weights between layers 1 and 2
w['w_2_3'] = w_2_3 # weights between layers 2 and 3

# Episode counters
episode_number = 0
episode_counter = 0 # this is used to detect the completion of a batch

# INITIALIZATIONS FOR WEIGHTS UPDATE (OPTIMIZER)
z = {}
m = {}
v = {}
RdEdw_dict_batch = {}
for i, j in iter(w.items()):
    z[i] = np.zeros_like(j)
    m[i] = np.zeros_like(j)
    v[i] = np.zeros_like(j)
    RdEdw_dict_batch[i] = np.zeros_like(j)

# INITIALIZATIONS FOR REINFORCEMENT LEARNING
rewards_list = []

# FUNCTION DEFINITIONS
# Activation functions
def ReLU(x):
    x[x<=0] = 0
    return x

def logistic_function(x):
    logistic = 1 / (1 + np.exp(-x))
    return logistic

def softmax_function(x):
    softmax = np.exp(x) / np.sum(np.exp(x))
    return softmax

# Optimizers
def momentum_optimization(z, i, momentum, RdEdw_dict_batch, w, alpha): # This function is used for basic gradient ascent, gradient ascent with momentum, and can be used for Nesterov Accelerated Gradient
    # z[i] = momentum*z[i] + RdEdw_dict_batch[i] # IU lecture notes version
    # w[i] += alpha * z[i] # IU lecture notes version
    z[i] = momentum * z[i] + alpha*RdEdw_dict_batch[i] # Goodfellow-Bengio-Courville version
    w[i] += z[i] # Goodfellow-Bengio-Courville version
    return w[i]

def rms_prop_optimization(v, i, decay_rate, RdEdw_dict_batch, epsilon, w, alpha):
    v[i] = decay_rate * v[i] + (1 - decay_rate) * RdEdw_dict_batch[i] ** 2 # Note: ** does element by element powers just like np.power()
    w[i] += alpha * RdEdw_dict_batch[i] / (np.sqrt(v[i]) + epsilon) # Note: / does element-by-element division just like np.divide()
    return w[i]

def adam_optimization(m, i, decay_rate_1, RdEdw_dict_batch, decay_rate_2, v, episode_number, batch_size, w, alpha, epsilon):
    m[i] = decay_rate_1 * m[i] + (1 - decay_rate_1) * RdEdw_dict_batch[i]
    v[i] = decay_rate_2 * v[i] + (1 - decay_rate_2) * (RdEdw_dict_batch[i] ** 2)
    m_hat = m[i] / (1 - decay_rate_1 ** (episode_number/batch_size))
    v_hat = v[i] / (1 - decay_rate_2 ** (episode_number/batch_size))
    w[i] += (alpha * m_hat) / (np.sqrt(v_hat) + epsilon)
    return w[i]

# Data processing. Purpose: To reduce dimensionality for the purpose of reducing computational effort and reducing risk of overfitting.
def preprocess(input_data):
    # Convert the input (210 x 160 pixels, each with 3 colours (uint8 (8 bit, i.e. 0 to 255)) to a float array, removing unnecessary data
    # print('Input Data: ', type(input_data), input_data.shape, input_data.dtype, input_data)
    input_data = input_data[35:195:2,::2,0]  # crop irrelevant parts of image, only use every second pixel (every 4th or higher would be insufficient though to show the ball), leaving only a 80 x 80 pixel image, and use only 2 bits of the 8 bits of colour. Note: Indeces for cropping were taken from Karpathy's implementation but can be confirmed by printing the entire input_data array and checking where background values (see comments below) start to dominate).
    # with np.printoptions(threshold=np.inf):
    #     print(input_data) # this shows that the vast majority of pixels have a value of either 109 or 144, so this must be the background.
    input_data[input_data == 109] = 0  # set background to 0.
    input_data[input_data == 144] = 0  # set background to 0.
    input_data[input_data != 0] = 1 # set rest of pixels (i.e. the rackets and the ball) to 1
    # with np.printoptions(threshold=np.inf):
    #     print(input_data)
    input_data = input_data.astype(np.float) # shape: (80,80)
    frame_current = input_data.reshape(inputs) # shape: (6400,)
    return frame_current

# Backpropagation
def forward_pass(x, w):
    x_1w_12 = np.dot(w['w_1_2'],x) # shapes: (200, 6401) . (6401,1) = (200,1)
    x_2 = ReLU(x_1w_12) # Note: This function operates element-by-element
    x_2 = np.append(x_2, 1)
    x_2 = x_2[:, np.newaxis] # (201,1)
    o_3 = np.dot(w['w_2_3'],x_2) # shapes: (3,201) . (201,1) = (3,1)
    if outputs == 1:
        y_hat = logistic_function(o_3)
    else:
        y_hat = softmax_function(o_3) # shape: (3,1)
    return y_hat, x_2  # y_hat is the probability of taking the actions

def get_discounted_total_rewards(rewards_array, discount_factor):  # This discounts the further-in-the-future rewards more heavily than the closer ones, i.e. rewards received as an immediate result of the action taken have more weight
    discounted_total_rewards = np.zeros_like(rewards_array)
    for i in reversed(range(rewards_array.size)):
        if rewards_array[i,:] != 0:
            discounted_total_rewards[i,:] = rewards_array[i,:]
            non_zero_index = i
            rewards_nz_index = discounted_total_rewards[non_zero_index]
        else:
            rewards_nz_index = discount_factor * rewards_nz_index
            discounted_total_rewards[i,:] = rewards_nz_index
    return discounted_total_rewards
    # for example: rewards_array = [1,0,0,1,0,0,0,1], so we should get discounted_total_rewards = [1*1,!GAME RESET! 0*1 + 0*0.9 + 1*0.9^2, 0*1 + 1*0.9, 1*1, !GAME RESET!, 0*1 +....]

def backward_pass(x_1_array, x_2_array, RdEdo_3_array, w):
    RdEdw_2_3 = np.dot(RdEdo_3_array,x_2_array) # shapes: (3,states).(states,201) = (3,201).
    w_2_3 = w['w_2_3']
    w_2_3_ = w_2_3[:,0:-1]
    RdEdx_2 = np.dot(w_2_3_.T, RdEdo_3_array) # for positive inputs to the ReLU activation. shapes: (200,3).(3,states) = (200,states)
    RdEdx_2[x_2_array[:,0:-1].T <= 0] = 0  # for non-positive inputs to the ReLU activation
    RdEdw_1_2 = np.dot(RdEdx_2,x_1_array) # shapes: (200,states).(states,inputs+1) = (200,inputs+1).
    RdEdw_dict = {'w_1_2': RdEdw_1_2, 'w_2_3': RdEdw_2_3}
    return RdEdw_dict

# CREATE PONG ENVIRONMENT AND GET FIRST STATE (OBSERVATION)
environment = gym.make("Pong-v0")
input_data = environment.reset() # resets environment to initial state and returns initial observation

# INITIALIZATIONS FOR DATA WHICH IS FOR INFORMATION ONLY
total_reward = 0
total_reward_list = []
episode_list = []
nzerosones_ratio = []
E_array = []

fig, axis = plt.subplots(3)
# fig2, axis2 = plt.subplots(1)

# MAIN LOOP OVER STATES
while episode_number < 2501:
    environment.render()

    # FORWARD PASS
    # Preprocess observation to get input, i.e. state
    frame_current = preprocess(input_data)

    # # Visualizing the preprocessed data
    # # backtosquare = frame_current.reshape(160, 160)
    # backtosquare = frame_current.reshape(80, 80)
    # # backtosquare = frame_current.reshape(40, 40)
    # axis2.imshow(backtosquare)
    # axis2.set_title('Preprocessed image')
    # plt.draw()
    # plt.pause(0.05)

    # Get state vector
    if frame_previous is None:
        x_1 = np.zeros(inputs+1) # shape: (inputs,)
        x_1[-1] = 1 # for the bias term
    else:
        x_1 = frame_current - frame_previous # shape: (inputs,)
        x_1 = np.append(x_1, 1)
    x_1_list.append(x_1) # to be used in backward pass
    x_1 = x_1[:, np.newaxis] # shape: (inputs,1)
    # print('x_1', x_1, type(x_1), x_1.shape)

    # Predict action
    y_hat, x_2 = forward_pass(x_1, w) # shapes: (3,1), (200,1)
    # print('x_2', x_2, type(x_2), x_2.shape)

    # Take action in accordance with current policy
    if outputs == 3:
        y_hat_1D = np.reshape(y_hat, outputs) # shape: (3,1) --> (3,)
        action = np.random.choice(actions_three, size=1, p=y_hat_1D)
        if action == 2: # Up
            y = y_array[0,:,np.newaxis] # map selected action onto logistic range between 0 and 1 (i.e. a probability) so it can acts as the new true one-hot vector (i.e. label)
        if action == 5: # Down
            y = y_array[1,:,np.newaxis]
        if action == 0: # No move
            y = y_array[2,:,np.newaxis]
    if outputs == 2:
        y_hat_1D = np.reshape(y_hat, outputs) # shape: (2,1) --> (2,)
        action = np.random.choice(actions_two, size=1, p=y_hat_1D)
        if action == 2: # Up
            y = y_array[0, :, np.newaxis]  # map selected action onto logistic range between 0 and 1 (i.e. a probability) so it can acts as the new true one-hot vector (i.e. label)
        if action == 5:  # Down
            y = y_array[1, :, np.newaxis]
    if outputs == 1:
        y_hat_1D = np.reshape(y_hat, outputs)  # shape: (1,1) --> (1,)
        y_hat_1D__ = np.array([1-y_hat_1D[0]])
        y_hat_1D_ = np.append(y_hat_1D,y_hat_1D__)
        action = np.random.choice(actions_two, size=1, p=y_hat_1D_)
        if action == 2:  # map selected action onto logistic range between 0 and 1 (i.e. a probability) so it can acts as the new true one-hot vector (i.e. label)
            y = np.array([[1]])
        else:
            y = np.array([[0]])

    # Log policy = cross entropy cost with opposite sign:
    # E = sum_over_classes(y*ln(y_hat)) = ln(y_hat) because y is one-hot encoded.

    # GET REWARD
    # Get reward, and the new state for the next forward pass
    input_data, reward, end_of_episode, lives = environment.step(action)

    # PREPARE DATA FOR BACKWARD PASS, NEXT FORWARD PASS, AND FOR PLOTTING
    # Add up rewards for the episode (for plotting only)
    total_reward += reward

    # Reset values after each state, i.e. after each forward pass
    frame_previous = frame_current

    # Determine gradient wrt outputs o_3 (for backward pass)
    dEdo_3 = y - y_hat # for cross entropy cost and softmax output activation this is the analytical solution (and the solution is the same for logistic function for binary classification using one output node).
    dEdo_3_array = np.append(dEdo_3_array, dEdo_3, axis=1)  # shape: (3,states)
    x_2_ = np.squeeze(x_2, axis=1)  # shape: (200,1) --> (200,)
    x_2_list.append(x_2_) # Alternatively can use np.append(), but appending lists is probably faster
    rewards_list.append(reward)  # Alternatively can use np.append(), but appending lists is probably faster

    # EPISODE IS COMPLETE
    if end_of_episode == True:

        # Update counters
        episode_number += 1
        episode_counter += 1

        # Update list of episodes, total rewards for the episodes, ratio of non-negative rewards, and compute cost (for plotting only)
        episode_list.append(episode_number)
        print('Episodes: ', episode_list)
        total_reward_list.append(total_reward)
        print('Total rewards for the episodes: ', total_reward_list)
        print('Rewards for current episode: ', rewards_list)
        nzeros = sum(i==0 for i in rewards_list)
        nones = sum(i==1 for i in rewards_list)
        nzr = (nzeros + nones)/len(rewards_list)
        nzerosones_ratio.append(nzr)
        print('Ratio of 0 and 1 rewards out of total rewards: ', nzerosones_ratio)
        episode_array = np.array(episode_list)
        total_reward_array = np.array(total_reward_list)
        E = y * np.log(y_hat) # this is always negative
        E = np.sum(E)
        E_array.append(E) # stores negative cross entropy cost (or log policy) of last state-action pair in the episode
        print('E: ', E_array)

        # BACKWARD PASS
        # Apply discounted total reward to gradient to promote policies with high rewards
        rewards_array_ = np.array(rewards_list) # shape: (states,)
        rewards_array = rewards_array_[:, np.newaxis] # shape: (states,1)
        discounted_total_rewards = get_discounted_total_rewards(rewards_array, discount_factor)
        print ('Discounted total rewards for current episode: ', discounted_total_rewards)
        discounted_total_rewards_with_baseline = discounted_total_rewards - np.mean(discounted_total_rewards)
        # Note on variance reduction: Sutton suggests subtracting the mean and Somnuk implements it like this too, greensmith states this is suboptimal though. Karpathy normalizes the reward to standard normal distributon instead.
        # Levine uses the average rewards over all trajectories, not just over the current trajectory, (but also shows the variance minimizing version of th baseline is the more optimal choice). Weaver also describes the former as the 'long term', average reward.
        # Note: For implementation purposes it is however easier to implement the baseline as only the average reward for the current trajectory, as opposed to either a reward averaged over all trajectories or a variance minimizing baseline.
        RdEdo_3_array = np.dot(dEdo_3_array[:,1:], np.diag(np.squeeze(discounted_total_rewards_with_baseline, axis=1))) # shapes: (3,states)*(states,states) = (3,states)

        # Get gradients wrt weights
        x_1_array = np.array(x_1_list) # shape: (states, inputs)
        x_2_array = np.array(x_2_list)  # shape: (states, 200)
        RdEdw_dict = backward_pass(x_1_array, x_2_array, RdEdo_3_array, w)

        # PREPARE GRADIENTS FOR BATCH WEIGHTS UPDATE. Note: This is where the summation over the trajectories happens.
        for i in w:
            RdEdw_dict_batch[i] += RdEdw_dict[i]

        # BATCH OF EPISODES IS COMPLETE
        if episode_counter == batch_size:
            # Determine average gradients over the batch
            for i in w:
                RdEdw_dict_batch[i] /= batch_size

            # WEIGHTS UPDATE (OPTIMIZER)
            for i, j in iter(w.items()):

                if optimizer == 'momentum':
                    # Weights update with gradient ascent or gradient ascent with momentum (momentum term creates an exponentially decaying running average of past gradients)
                    w[i] = momentum_optimization(z, i, momentum, RdEdw_dict_batch, w, alpha)

                # if optimizer == 'NAG':
                    # Weights Update with Nesterov Accelerated Gradient (possible future work)
                    # w[i] = momentum_optimization(z, i, momentum, RdEdw_dict_batch_prediction, w_prediction, alpha)

                if optimizer == 'RMSprop':
                    # Weights update with RMSprop:
                    w[i] = rms_prop_optimization(v, i, decay_rate, RdEdw_dict_batch, epsilon, w, alpha)

                if optimizer == 'ADAM':
                    # Weights update with ADAM (combination of momentum and RMSprop)
                    w[i] = adam_optimization(m, i, decay_rate_1, RdEdw_dict_batch, decay_rate_2, v, episode_number, batch_size, w, alpha, epsilon)

                # if optimizer == 'AMSgrad':
                    # Weights update with AMSGrad optimizer (possible future work)

                # Reset gradients to zero
                RdEdw_dict_batch[i] = np.zeros_like(j)

            print('Weights updated by a batch of samples.\n')

            # Reset batch episode counter to zero
            episode_counter = 0

        # RESET EPISODE-SPECIFIC VALUES AND ENVIRONMENT
        frame_previous = None
        x_1_list = []
        x_2_list = []
        dEdo_3_array = np.zeros((outputs,1))
        rewards_list = []
        total_reward = 0 # for display of results only
        input_data = environment.reset() # resets environment to initial state and returns initial observation

        # PLOT AND PRINT RESULTS
        if episode_number % 10 == 0:

            axis[0].plot(episode_array,total_reward_array, 'b') # Purpose: To check if objective function really is being minimized
            axis[0].set_xlabel('Episodes')
            axis[0].set_ylabel('Total reward per episode')

            axis[1].plot(episode_array, nzerosones_ratio, c='k') # Purpose: To check if objective function really is being minimized. Here, this is easier to see than in the previous graph.
            axis[1].set_xlabel('Episodes')
            axis[1].set_ylabel('Ratio of non-negative rewards')

            axis[2].plot(episode_array, E_array, c='g') # Purpose: To check if the neural network is working correctly.
            axis[2].set_xlabel('Episodes')
            axis[2].set_ylabel('Most recently computed cost')
            plt.draw()
            plt.pause(0.05)

            w_2_3_plot = w['w_2_3']
            w_1_2_plot = w['w_1_2']
            print('Weights: ', w_1_2_plot.shape, w_2_3_plot.shape, w)