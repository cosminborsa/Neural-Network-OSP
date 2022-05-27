# Implementation of a sequence of neural networks in LSM
# The training was sped up by wieght and bias initialization

# Implementation of a single neural network in LSM
# The neural network exhibits catastrophic forgetting

from payoffs import payoff

## Libraries
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import TruncatedNormal

def NN_seq_train(stock, model, theta = 'average', node_num = 16, epoch_num = 50, 
                 batch_num = 64, actfct = 'elu', 
                 initializer = TruncatedNormal(mean = 0.0, stddev = 0.05),
                 optim = 'adam', lossfct = 'mean_squared_error', display_time = False):
    '''
    Implementation of the Longstaff Schwartz Algorithm using a sequence of neural 
    network objects. Training is done along a fixed set of paths
    
    Parameters
    ----------
    stock : Array of shape M x N x model['dim'] of simulated stock paths
            M is the number of steps
            N is the number of simulations
            model['dim'] is the number of dimensions
    model : Dictionary containing all the parameters of the stock and contract
    node_num : Number of nodes. The default is 16.
    epoch_num : Number of epochs. The default is 50.
    batch_num : Number of batches. The default is 64.
    actfct : Activation function. The default is 'elu'.
    initializer : Keras initializer. The default is set to 
                  TruncatedNormal(mean = 0.0, stddev = 0.05).
    optim : Keras optimizer. The default is 'adam'.
    lossfct : Loss function. The default is 'mean_squared_error'.
    theta : Initialization policy of the weights and the biases of the neural 
            network objects. The first noural netowrk is initialized with the 
            keras 'initializer'. 
            If theta = 'average', then the weights and biases from subsequent 
            networks are initialized with the averges of the weights and biases 
            from the previous steps. 
            If theta = 'previous', then the weights and biases from subsequent 
            networks are initialized with the the weights and biases from the 
            previous step. 
            If theta = 'random', then the weights and biases of all the neural
            networks are initialized with the keras 'initializer'
            The default is 'average'.
    display_time : Display time spent per step. The default is True.

    Returns
    -------
    v : Backward price of the option contract
    stopT : List of size N of stopping times
    NN : List of neural network objects (size M-1)
    convert_in : List of input scaling objects (size M-1)
    convert_out : List of output scaling objects (size M-1) 
    '''
    nn_dim = model['dim']                   # Dimension of the neural network
    nSims = len(stock[0])                   # Number of Simulations
    nSteps = int(model['T']/model['dt'])    # Number of Steps
    stopT = np.repeat(nSteps*model['dt'], nSims) # List of stopping times
    q = payoff(stock[-1], model)      # List of continuation values
    convert_in = []     # List of input scaling objects -- MinMaxScaler() 
    convert_out = []    # List of output scaling objects -- MinMaxScaler()
    NN = []             # List of neural network objects 
    
    # Backward Loop
    for i in reversed(range(0,nSteps-1)):
        if display_time:
            start_time = time.time()
        
        # Selecting In-the-Money paths for training
        itm = [] 
        for k in range(nSims):
            if payoff(stock[i][k], model) > 0:
                itm.append([stock[i][k], q[k]])
        x = np.stack(np.array(itm, dtype=object).transpose()[0])
        y = np.array(itm, dtype=object).transpose()[1]
        
        # Scaling neural network inputs        
        input_train_scaled = []
        input_scaler_dim = []
        for j in range(model['dim']):
             input_train = x.transpose()[j].reshape((-1, 1))
             input_scaler = MinMaxScaler(feature_range = (0,1))
             input_scaler.fit(input_train)
             input_train_scaled.append(input_scaler.transform(input_train))
             input_scaler_dim.append(input_scaler)
        convert_in.append(input_scaler_dim)
        x_ = np.stack(input_train_scaled).transpose()[0]
        
        # Scaling neural network outputs
        valuefun_train = y
        output_scaler_valuefun = MinMaxScaler(feature_range = (0,1))
        output_scaler_valuefun.fit(valuefun_train.reshape(-1, 1))
        valuefun_train_scaled = output_scaler_valuefun.transform(valuefun_train.reshape(-1,1))
        convert_out.append(output_scaler_valuefun)
        
        # Defining and training the neural network
        if  i == nSteps-2:
            NNet_seq = Sequential()    
            NNet_seq.add(Dense(node_num, input_shape = (nn_dim,), activation = actfct,
                         kernel_initializer = initializer, bias_initializer = initializer))            
            NNet_seq.add(Dense(node_num, activation = actfct,
                         kernel_initializer = initializer, bias_initializer = initializer))
            NNet_seq.add(Dense(node_num, activation = actfct, 
                         kernel_initializer = initializer, bias_initializer = initializer))
            NNet_seq.add(Dense(1, activation = None, 
                         kernel_initializer = initializer, bias_initializer = initializer))
            NNet_seq.compile(optimizer = optim, loss = lossfct)
            NNet_seq.fit(x_, valuefun_train_scaled, epochs = epoch_num, \
                         batch_size = batch_num, verbose = 0)
        else:
            if theta == 'average':
                # Average weights and biases
                w_mean = []
                b_mean = []
                w = np.empty(shape = (len(NN),4), dtype = object)
                b = np.empty(shape = (len(NN),4), dtype = object)
                for n in range(len(NN)):
                    for m in range(len(NN[n].layers)): # Number of layers
                        w[n][m] = NN[n].layers[m].get_weights()[0]
                        b[n][m]= NN[n].layers[m].get_weights()[1]
                for m in range(4):
                    w_mean.append(w.transpose()[m].mean())
                    b_mean.append(b.transpose()[m].mean())
            elif theta == 'previous':
                # Previous weights and biases
                w_mean  = []
                b_mean = []
                for lay in NN[-1].layers:
                    w_mean.append(lay.get_weights()[0])
                    b_mean.append(lay.get_weights()[1])
            elif theta == 'random':
                # Random weights and biases
                w_mean = initializer
                b_mean = initializer
                  
            NNet_seq = Sequential()
            NNet_seq.add(Dense(node_num, input_shape = (nn_dim,), activation = actfct,
                            kernel_initializer = tf.keras.initializers.Constant(w_mean[0]), 
                            bias_initializer = tf.keras.initializers.Constant(b_mean[0])))            
            NNet_seq.add(Dense(node_num, activation = actfct,
                            kernel_initializer = tf.keras.initializers.Constant(w_mean[1]),
                            bias_initializer = tf.keras.initializers.Constant(b_mean[1])))
            NNet_seq.add(Dense(node_num, activation = actfct, 
                            kernel_initializer = tf.keras.initializers.Constant(w_mean[2]), 
                            bias_initializer = tf.keras.initializers.Constant(b_mean[2])))
            NNet_seq.add(Dense(1, activation = None, 
                            kernel_initializer = tf.keras.initializers.Constant(w_mean[3]), 
                            bias_initializer = tf.keras.initializers.Constant(b_mean[3])))
            NNet_seq.compile(optimizer = optim, loss = lossfct)
            NNet_seq.fit(x_, valuefun_train_scaled, epochs = epoch_num, \
                                  batch_size = batch_num, verbose = 0)
        # Predicting continuation values using the neural network
        aux = []
        for j in range(model['dim']):
            aux.append(input_scaler_dim[j].transform(stock[i].transpose()[j].reshape((-1, 1))))
        input_train_scaled_all = np.stack(aux).transpose()[0]
        
        pred = NNet_seq.predict(input_train_scaled_all)
        prediction = np.ravel(output_scaler_valuefun.inverse_transform(pred))
        NN.append(NNet_seq)
        
        # Computing continuation values
        qhat = np.array([prediction, np.zeros(nSims)]).max(axis = 0)
        qhat = np.exp(-model['r']*model['dt'])*qhat
        imm_pay = payoff(stock[i], model)
        
        # Updating the continuation values and stopping times
        for k in range(nSims):
            if (imm_pay[k] > 0) and (qhat[k] <= imm_pay[k]):
                stopT[k] = (i+1)*model['dt']
                q[k] = imm_pay[k]
            else:
                q[k] = np.exp(-model['r']*model['dt'])*q[k]
        
        # Displaying Time per Step
        if display_time:
            print('Step i =',i+1,' Time =', np.round(time.time()-start_time,2), 'sec')
        
    # Computing the backward price
    imm_pay = float(payoff(model['x0'], model))
    v = max(imm_pay, np.mean(q))  
    
    # Reversing lists 
    NN.reverse()
    convert_in.reverse()
    convert_out.reverse()
    return (v, stopT, NN, convert_in, convert_out)

def NN_one_train(stock, model, ratio = None, node_num = 16, epoch_num = 50, 
                 batch_num = 64, actfct = 'elu', 
                 initializer = TruncatedNormal(mean = 0.0, stddev = 0.05), 
                 optim = 'adam', lossfct = 'mean_squared_error', display_time = False):
    '''
    Implementation of the Longstaff Schwartz Algorithm using a single neural 
    network objects. Training is done along a fixed set of paths and the
    neural network exhibits catastrophic forgetting
        
    Parameters
    ----------
    stock : Array of shape M x N x model['dim'] of simulated stock paths
            M is the number of steps
            N is the number of simulations
            model['dim'] is the number of dimensions
    model : Dictionary containing all the parameters of the stock and contract
    ratio : Ratio of old data to new data used in training the neural network. 
            E.g. ratio = 2 means that the ratio of old data to new data used in
            training the neural network is 2:1. 
            If the ratio is note provided, then old data is not reused in training
            the network. The default is None.
    node_num : Number of nodes. The default is 16.
    epoch_num : Number of epochs. The default is 50.
    batch_num : Number of batches. The default is 64.
    actfct : Activation function. The default is 'elu'.
    initializer : Keras initializer. The default is set to 
                  TruncatedNormal(mean = 0.0, stddev = 0.05).
    optim : Keras optimizer. The default is 'adam'.
    lossfct : Loss function. The default is 'mean_squared_error'.
    display_time : Display time spent per step. The default is False.

    Returns
    -------
    v : Backward price of the option contract
    stopT : List of size N of stopping times
    NN : List of neural network objects (size M-1)
    convert_in : List of input scaling objects (size M-1)
    convert_out : List of output scaling objects (size M-1) 
    '''
    nn_dim = model['dim'] + 1               # Dimension of the neural network
    nSims = len(stock[0])                   # Number of simulations
    nSteps = int(model['T']/model['dt'])    # Number of steps
    stopT = np.repeat(nSteps*model['dt'], nSims) # List of stopping times
    q = payoff(stock[-1], model)      # List of continuation values
    convert_in = []     # List of input scaling objects -- MinMaxScaler() 
    convert_out = []    # List of output scaling objects -- MinMaxScaler()
    if (ratio != None) and (ratio > 0):
        accum = np.array([]) # Array of in-the-money paths used in training
        len_accum = 0        # Tracks the length of accum 
    
    dim_scaler = MinMaxScaler(feature_range = (0,1))
    dim_train = np.array(list(reversed(range(0,nSteps-1)))).reshape((-1, 1))
    dim_scaler.fit(dim_train)
    
    # Backward Loop
    for i in reversed(range(0,nSteps-1)):
        if display_time:
            start_time = time.time()
        
        # Selecting In-the-Money paths for training
        itm = [] 
        for k in range(nSims):
            if payoff(stock[i][k], model) > 0:
                itm.append([stock[i][k], q[k]])
        x = np.stack(np.array(itm, dtype=object).transpose()[0])
        y = np.array(itm, dtype=object).transpose()[1]
        
        # Scaling neural network inputs        
        input_train_scaled = [dim_scaler.transform(np.repeat(i,len(itm)).reshape(-1, 1))]
        input_scaler_dim = [dim_scaler]
        for j in range(model['dim']):
             input_train = x.transpose()[j].reshape((-1, 1))
             input_scaler = MinMaxScaler(feature_range = (0,1))
             input_scaler.fit(input_train)
             input_train_scaled.append(input_scaler.transform(input_train))
             input_scaler_dim.append(input_scaler)
        convert_in.append(input_scaler_dim)
        x_ = np.stack(input_train_scaled).transpose()[0]
        
        # Scaling neural network outputs
        valuefun_train = y
        output_scaler_valuefun = MinMaxScaler(feature_range = (0,1))
        output_scaler_valuefun.fit(valuefun_train.reshape(-1, 1))
        valuefun_train_scaled = output_scaler_valuefun.transform(valuefun_train.reshape(-1,1))
        convert_out.append(output_scaler_valuefun)
        
        # Defining and training the neural network
        if  i == nSteps-2:
            NNet = Sequential()    
            NNet.add(Dense(node_num, input_shape = (nn_dim,), activation = actfct, \
                           kernel_initializer = initializer, bias_initializer = initializer))            
            NNet.add(Dense(node_num, activation = actfct, kernel_initializer = initializer, \
                           bias_initializer = initializer))
            NNet.add(Dense(node_num, activation = actfct, kernel_initializer = initializer, \
                           bias_initializer = initializer))
            NNet.add(Dense(1, activation = None, kernel_initializer = initializer, \
                           bias_initializer = initializer))
            NNet.compile(optimizer = optim, loss = lossfct)
            NNet.fit(x_, valuefun_train_scaled, epochs = epoch_num, \
                     batch_size = batch_num, verbose = 0)
            
            if (ratio != None) and (ratio > 0):
                len_accum += len(itm)
                accum = np.append(accum, np.append(x_.transpose(), \
                            valuefun_train_scaled.transpose()).reshape((model['dim'] + 2, \
                            len(itm))).transpose()).reshape((model['dim'] + 2, len_accum), \
                            order = 'F').transpose()
        else:
            if (ratio != None) and (ratio > 0):
                if ratio*len(itm) > len_accum:
                    # Include all the previous data
                    x_val = np.concatenate([accum.transpose()[:-1].transpose(), x_], \
                                           axis = 0)
                    y_val = np.concatenate([accum.transpose()[-1].reshape(-1, 1), \
                                            valuefun_train_scaled], axis = 0)
                else:
                    # Shuffle and use the old to new data ratio 
                    np.random.shuffle(accum)
                    x_val = np.concatenate([accum.transpose()[:-1].transpose()[:2*len(itm)], x_], \
                                       axis = 0)
                    y_val = np.concatenate([accum.transpose()[-1].reshape(-1, 1)[:2*len(itm)], \
                                       valuefun_train_scaled], axis = 0)
            else:
                x_val = x_
                y_val = valuefun_train_scaled
            NNet.fit(x_val, y_val, epochs = epoch_num, batch_size = batch_num, verbose = 0)
            
            if (ratio != None) and (ratio > 0):
                len_accum += len(itm)
                accum = np.append(accum, np.append(x_.transpose(), \
                            valuefun_train_scaled.transpose()).reshape((model['dim'] + 2, \
                            len(itm))).transpose()).reshape((model['dim'] + 2, len_accum), \
                            order = 'F').transpose()
        # Predicting continuation values using the neural network
        aux = [input_scaler_dim[0].transform(np.repeat(i,nSims).reshape(-1, 1))]
        for j in range(model['dim']):
            aux.append(input_scaler_dim[j+1].transform(stock[i].transpose()[j].reshape((-1, 1))))
        input_train_scaled_all = np.stack(aux).transpose()[0]
        
        pred = NNet.predict(input_train_scaled_all)
        prediction = np.ravel(output_scaler_valuefun.inverse_transform(pred))
        
        # Computing continuation values
        qhat = np.array([prediction, np.zeros(nSims)]).max(axis = 0)
        qhat = np.exp(-model['r']*model['dt'])*qhat
        imm_pay = payoff(stock[i], model)
        
        # Updating the continuation values and stopping times
        for k in range(nSims):
            if (imm_pay[k] > 0) and (qhat[k] <= imm_pay[k]):
                stopT[k] = (i+1)*model['dt']
                q[k] = imm_pay[k]
            else:
                q[k] = np.exp(-model['r']*model['dt'])*q[k]
        
        # Displaying Time per Step
        if display_time:
            print('Step i =',i+1,' Time =', np.round(time.time()-start_time,2), 'sec')
        
    # Computing the backward price
    imm_pay = float(payoff(model['x0'], model))
    v = max(imm_pay, np.mean(q))
    
    # Reversing lists 
    convert_in.reverse()
    convert_out.reverse()
    return (v, stopT, NNet, convert_in, convert_out)

def NN_train(stock, model, net, ratio = None, theta = 'average', node_num = 16, epoch_num = 50, 
                 batch_num = 64, actfct = 'elu', 
                 initializer = TruncatedNormal(mean = 0.0, stddev = 0.05),
                 optim = 'adam', lossfct = 'mean_squared_error', display_time = False):
    '''
    Aggregartes the implementation of the Longstaff Schwartz Algorithm using 
    either a single neural network or a sequence of network objects. 
        
    Parameters
    ----------
    net : Selects the training option of LSM
    Other parameters are either from NN_seq_train or NN_one_train
    
    Raises
    ------
    TypeError : Neural network type has not been properly selected.
    
    Returns
    -------
    Return of either NN_seq_train or NN_one_train
    '''
    # Runs the training of sequence of neural networks
    if net == 'seq':
        return NN_seq_train(stock, model, theta, node_num, epoch_num, batch_num, 
                            actfct, initializer, optim, lossfct, display_time)
    # Runs the training of single of neural network
    elif net == 'single':
        return NN_one_train(stock, model, ratio, node_num, epoch_num, batch_num, 
                            actfct, initializer, optim, lossfct, display_time)
    else:
        raise TypeError('Neural network type has not been properly selected.')
