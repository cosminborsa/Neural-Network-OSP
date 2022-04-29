'''
LONGSTAFF SCHWARTZ ALGORITHM with NEURAL NETWORKS       
MODULARITY IMPLEMENTED    
'''

#### Libraries
# Standard library
import time

# Third-party libraries
import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.models import load_model 

def sim_gbm(x0, model):
    '''
    Simulate paths of Geometric Brownian Motion with constant parameters
    Simulate from \eqn{p(X_t|X_{t-1})}
    
    Parameters
    ----------
    x0 : Starting values (matrix of size N x model['dim'])
    model : Dictionary containing all the parameters, including volatility,
            interest rate, and continuous dividend yield

    Returns
    -------
    A matrix of same dimensions as x0
    '''
    length = len(x0)
    newX = []
    dt = model['dt']
    for j in range(model['dim']): # indep coordinates
       newX.append(x0[:,j]*np.exp(np.random.normal(loc = 0, scale = 1, size = length)*
                                 model['sigma'][j]*np.sqrt(dt) +
            (model['r'] - model['div']- model['sigma'][j]**2/2)*dt))
    return np.reshape(np.ravel(np.array(newX)), (length, model['dim']), order='F')

def stock_sim(nSims, model, start = None):
    '''
    Simulates stock paths using the Geometric Brownian Motion
    
    Parameters
    ----------
    nSims : Number of simulations -- N
    model : Dictionary containing all the parameters, including volatility, 
            interest rate, and continuous dividend yield
    start : Optional start value of the stock. The default is N x model['x0'].

    Returns
    -------
    An arrayof shape M x N x model['dim']
    where M is the number of steps model['T']/model['dt']
    '''
    if start is None:
        start = np.reshape(np.repeat(model['x0'], nSims), (nSims, model['dim']))
    if start.shape != (nSims, model['dim']):
        start = np.reshape(start, (nSims, model['dim']))
    nSteps = int(model['T']/model['dt'])
    test_j = []
    test_j.append(sim_gbm(start, model))
    for i in range(1,nSteps):
        test_j.append(sim_gbm(test_j[i-1], model))
    return np.reshape(np.ravel(test_j), (nSteps, nSims, model['dim']))

def payoff(stock_v, model):
    '''
    Generates option payoffs
    
    Parameters
    ----------
    stock_v : matrix of size N x model['dim'] or vector of size model['dim'] 
    model : Dictionary containing all the parameters, including strike, 
            and the payoff function

    Returns
    -------
    A vector of size N with option payoffs if stock_v is a matrix 
    An option payoff if stock_v is a vector of size model['dim']
    '''
    try:
        (nSims, dim) = stock_v.shape
    except ValueError:
        dim = None
    
    
    ## Arithmetic basket Put on average asset price
    ## Put payoff \eqn{(K-mean(x))_+}
    if model['payoff.func'] == 'put.payoff':
        if dim:
            return np.array([model['K']-np.ndarray.mean(stock_v, axis = 1), \
                             np.zeros(nSims)], dtype=object).max(axis = 0)
        else:
            return np.array([model['K'] - np.ndarray.mean(stock_v, axis = 0), \
                             np.zeros(1)], dtype=object).max(axis = 0)
    
    ## Multivariate Min Put
    ## Min Put payoff \eqn{(K-min(x))_+}
    elif model['payoff.func'] == 'mini.put.payoff':
        if dim: 
            return np.array([model['K']-np.ndarray.min(stock_v, axis = 1), \
                     np.zeros(nSims)], dtype=object).max(axis = 0)
        else:
            return np.array([model['K']-np.ndarray.min(stock_v, axis = 0), \
                     np.zeros(1)], dtype=object).max(axis = 0)
        
    ## Arithmetic basket Call on average asset price
    ## Call payoff \eqn{(mean(x)-K)_+}
    elif model['payoff.func'] == 'call.payoff': 
        if dim: 
            return np.array([np.ndarray.mean(stock_v, axis = 1) - model['K'], \
                     np.zeros(nSims)], dtype=object).max(axis = 0)
        else:
            return np.array([np.ndarray.mean(stock_v, axis = 0) - model['K'], \
                     np.zeros(1)], dtype=object).max(axis = 0)
    
    ## Multivariate Max Call
    ## Max Call payoff \eqn{(max(x)-K)_+}
    elif model['payoff.func'] == 'maxi.call.payoff':
        if dim: 
            return np.array([np.ndarray.max(stock_v, axis = 1) - model['K'], \
                     np.zeros(nSims)], dtype=object).max(axis = 0)
        else:
            return np.array([np.ndarray.max(stock_v, axis = 0) - model['K'], \
                     np.zeros(1)], dtype=object).max(axis = 0)

def NN_back(stock, model, node_num = 16, epoch_num = 50, batch_num = 64, 
            actfct = 'elu', initializer = TruncatedNormal(mean = 0.0, stddev = 0.05),
            optim = 'adam', lossfct = 'mean_squared_error', display_time = True):
    '''
    Longstaff Schwartz Algorithm using neural networks
    Trains the network along a fixed set of paths
    
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
    display_time : Display time spent per step. The default is True.

    Returns
    -------
    v : Backward price of the option contract
    stopT : List of size N of stopping times
    NN : List of Neural Network objects (size M-1)
    convert_in : List of input scaling objects (size M-1)
    convert_out : List of output scaling objects (size M-1) 
    '''
    nn_dim = model['dim'] + 1               # Dimension of the Neural Network
    nSims = len(stock[0])                   # Number of Simulations
    nSteps = int(model['T']/model['dt'])    # Number of Steps
    stopT = np.repeat(nSteps*model['dt'], nSims) # List of stopping times
    q = payoff(stock[-1], model)      # List of continuation values
    convert_in = []     # List of input scaling objects -- MinMaxScaler() 
    convert_out = []    # List of output scaling objects -- MinMaxScaler()
    accumulation = np.array([])
    len_accum = 0
    
    dim_scaler = MinMaxScaler(feature_range = (0,1))
    dim_train = np.array(list(reversed(range(0,nSteps-1)))).reshape((-1, 1))
    dim_scaler.fit(dim_train)
    
    # Backward Loop
    for i in reversed(range(0,nSteps-1)):
        if display_time:
            start_time = time.time()
        
        # Selecting In-the-Money Paths for training
        itm = [] 
        for k in range(nSims):
            if payoff(stock[i][k], model) > 0:
                itm.append([stock[i][k], q[k]])
        x = np.stack(np.array(itm, dtype=object).transpose()[0])
        y = np.array(itm, dtype=object).transpose()[1]
        
        # Scaling Neural Network inputs        
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
        
        # Scaling Neural Network outputs
        valuefun_train = y
        output_scaler_valuefun = MinMaxScaler(feature_range = (0,1))
        output_scaler_valuefun.fit(valuefun_train.reshape(-1, 1))
        valuefun_train_scaled = output_scaler_valuefun.transform(valuefun_train.reshape(-1,1))
        convert_out.append(output_scaler_valuefun)
        
        # Training the Neural Network
        if  i == nSteps-2:
            nnsolver_valuefun = Sequential()    
            nnsolver_valuefun.add(Dense(node_num, input_shape = (nn_dim,), activation = actfct,
                            kernel_initializer = initializer, bias_initializer = initializer))            
            nnsolver_valuefun.add(Dense(node_num, activation = actfct,
                            kernel_initializer = initializer, bias_initializer = initializer))
            nnsolver_valuefun.add(Dense(node_num, activation = actfct, 
                            kernel_initializer = initializer, bias_initializer = initializer))
            nnsolver_valuefun.add(Dense(1, activation = None, 
                            kernel_initializer = initializer, bias_initializer = initializer))
            nnsolver_valuefun.compile(optimizer = optim, loss = lossfct)
            nnsolver_valuefun.fit(x_, valuefun_train_scaled, \
                                  epochs = epoch_num, batch_size = batch_num, verbose = 0)
            nnsolver_valuefun.save('NN'+str(i)+'.h5')
            
            len_accum += len(itm)
            accumulation = np.append(accumulation, np.append(x_.transpose(), \
                        valuefun_train_scaled.transpose()).reshape((model['dim'] + 2, \
                        len(itm))).transpose()).reshape((model['dim'] + 2, len_accum), \
                        order = 'F').transpose()
        else:
            if 2*len(itm) > len_accum:
                # Include all the previous data
                x_val = np.concatenate([accumulation.transpose()[:-1].transpose(), x_], \
                                       axis = 0)
                y_val = np.concatenate([accumulation.transpose()[-1].reshape(-1, 1), \
                                        valuefun_train_scaled], axis = 0)
            else:
                # Shuffle and use a new to old data ratio of 2:1 
                np.random.shuffle(accumulation)
                x_val = np.concatenate([accumulation.transpose()[:-1].transpose()[:2*len(itm)], x_], \
                                       axis = 0)
                y_val = np.concatenate([accumulation.transpose()[-1].reshape(-1, 1)[:2*len(itm)], \
                                        valuefun_train_scaled], axis = 0)
            nnsolver_valuefun = load_model('NN'+str(i+1)+'.h5')
            # nnsolver_valuefun.compile(optimizer = optim, loss = lossfct)
            nnsolver_valuefun.fit(x_val, y_val, \
                                  epochs = epoch_num, batch_size = batch_num, verbose = 0)
            nnsolver_valuefun.save('NN'+str(i)+'.h5')
            
            len_accum += len(itm)
            accumulation = np.append(accumulation, np.append(x_.transpose(), \
                        valuefun_train_scaled.transpose()).reshape((model['dim'] + 2, \
                        len(itm))).transpose()).reshape((model['dim'] + 2, len_accum), \
                        order = 'F').transpose()
        # Predicting continuation values using the Neural Network
        aux = [input_scaler_dim[0].transform(np.repeat(i,nSims).reshape(-1, 1))]
        for j in range(model['dim']):
            aux.append(input_scaler_dim[j+1].transform(stock[i].transpose()[j].reshape((-1, 1))))
        input_train_scaled_all = np.stack(aux).transpose()[0]
        
        pred = nnsolver_valuefun.predict(input_train_scaled_all)
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
    return (v, stopT, nnsolver_valuefun, convert_in, convert_out)

def NN_fwd(stock, NN, convert_in, convert_out, model, display_time = True):
    '''
    Longstaff Schwartz Algorithm using neural networks
    Computes the option price using a trained neural network

    Parameters
    ----------
    stock : Array of shape M x N x model['dim'] of simulated stock paths
            M is the number of steps
            N is the number of simulations
            model['dim'] is the number of dimensions
    NN : List of Neural Network objects (size M-1)
    convert_in : List of input scaling objects (size M-1)
    convert_out : List of output scaling objects (size M-1)
    model : Dictionary containing all the parameters of the stock and contract
    display_time : Display time spent per step. The default is True.

    Returns
    -------
    v_ : Option price
    '''
    nSims = len(stock[0])                   # Number of Simulations
    nSteps = int(model['T']/model['dt'])    # Number of Steps
    # List of initial stock prices
    x0 = np.reshape(np.repeat(model['x0'], nSims), (nSims, model['dim']), order = 'F')
    imm_pay = payoff(x0, model) # Vector storing immedaite payoffs
    v_q = payoff(x0, model) # Array storing continuation values
    
    # List of arrays with stopping decisions: 
    # True means continue, False means stop 
    cont = []
    
    # Forward Loop
    for i in range(0, nSteps-1):
        if display_time:
            start_time = time.time()
        
        # Scaling Neural Network inputs
        aux_ = [convert_in[i][0].transform(np.repeat(i,nSims).reshape(-1, 1))]
        for j in range(model['dim']):
            aux_.append(convert_in[i][j+1].transform(stock[i].transpose()[j].reshape((-1, 1))))
        input_scaled = np.stack(aux_).transpose()[0]
        
        # Predicting continuation values 
        # Scaling Neural Network outputs
        pred = NN.predict(input_scaled)
        prediction = np.ravel(convert_out[i].inverse_transform(pred))
        q_hat = np.array([prediction, np.zeros(nSims)]).max(axis = 0)
        q_hat = np.exp(-model['r']*model['dt'])*q_hat
        imm_pay = payoff(stock[i], model)
        
        # Updating the stopping decision
        if i == 0:
            logic = np.logical_or(q_hat > imm_pay, imm_pay == 0)
        else:
            logic = np.logical_and(np.logical_or(q_hat > imm_pay, imm_pay == 0), cont[-1])
        cont.append(logic)
        
        # Perform stopping
        v_q = np.exp(model['r']*model['dt'])*v_q
        for k in range(nSims):
            if i == 0:
                if cont[-1][k] == False:
                    v_q[k] = imm_pay[k]
            else:
                if (cont[-1][k] == False) and (cont[-2][k] == True):
                    v_q[k] = imm_pay[k]
        
        # Displaying Time per Step
        if display_time:
            print('Step i =',i+1,'Time =', np.round(time.time()-start_time,2), 'sec')
    
    # Computing the terminal payoff
    imm_pay = payoff(stock[-1], model)
    v_q = np.exp(model['r']*model['dt'])*v_q
    for k in range(nSims):
        if cont[-1][k] == True:
            v_q[k] = imm_pay[k]
    
    # Discounting and computing the option price
    v_ = np.mean(np.exp(-model['r']*model['T'])*v_q)
    return v_

def NN_contour(step, NN, convert_in, convert_out, model, display_time = True, 
               down = 0.6, up = 1.4, inc = 0.05):
    '''
    Returns contour parameters

    Parameters
    ----------
    step : Time step of the contour
    NN : List of Neural Network objects (size M-1)
    convert_in : List of input scaling objects (size M-1)
    convert_out : List of output scaling objects (size M-1)
    model : Dictionary containing all the parameters of the stock and contract
    display_time : Display time spent per step. The default is True.
    down : Scaling parameter for the lower bound of the map. 
           The default is 0.6 or 60% of the initial price.
    up : Scaling parameter for the upper bound of the map. 
         The default is 1.4 or 140% of the initial price.
    inc : Increment for the stock price grid. The default is 0.05.

    Raises
    ------
    ValueError : Time step is out of range.
    TypeError : Only works for 2-D contracts.

    Returns
    -------
    Tuple consisting of contour parameters

    '''
    if display_time:
        start_time = time.time()
    i = step
    nSteps = int(model['T']/model['dt'])
    if i >= nSteps:
        raise ValueError('Time step is out of range.')
    if model['dim'] != 2:
        raise TypeError('Only works for 2-D options.')
    prices = []
    for x0 in model['x0']:
        prices.append(np.arange(x0*down, x0*up, inc))
    n = len(prices[0])
    x, y = np.meshgrid(prices[0], prices[1])
    aux = []
    for a, b in zip(x, y):
        aux.append(np.reshape(np.ravel([a,b]), (n,2), order = 'F'))
    prices_ = np.reshape(np.ravel(aux), (n**2, 2), order='C')
    
    # prices = np.array(prices).transpose()
    
    imm_pay = payoff(prices_, model)
    
    aux_ = [convert_in[i][0].transform(np.repeat(i,n**2).reshape(-1, 1))]
    for j in range(model['dim']):
        aux_.append(convert_in[i][j+1].transform(prices_.transpose()[j].reshape((-1, 1))))
    input_scaled = np.stack(aux_).transpose()[0]
    
    # Predicting continuation values 
    # Scaling Neural Network outputs
    pred = NN.predict(input_scaled)
    prediction = np.ravel(convert_out[i].inverse_transform(pred))
    q_hat = np.array([prediction, np.zeros(len(prediction))]).max(axis = 0)
    q_hat = np.exp(-model['r']*model['dt'])*q_hat
    q = q_hat - imm_pay 
    q = np.reshape(q, (n, n))
    if display_time:
        print('Time =', np.round(time.time()-start_time,2), 'sec')
    return (prices[0], prices[1], q)

def NN_bound(NN, convert_in, convert_out, model, down = 0.75, up = 1.01, 
             inc = 0.05, display_time = True):
    if display_time:
        start_time = time.time()
    
    nSteps = int(model['T']/model['dt'])
    if model['dim'] != 1:
        raise TypeError('Only works for 1-D options.')
    
    prices = np.arange(model['x0']*down, model['x0']*up, inc)
    n = len(prices)
    steps = np.arange(model['dt'], model['T'], model['dt']) 
    
    imm_pay = payoff(np.reshape(prices, (n,1)), model)
    
    # Predicting continuation values 
    # Scaling Neural Network outputs
    q = []
    for i in range(nSteps-1):
        aux_ = [convert_in[i][0].transform(np.repeat(i,n).reshape(-1, 1))]
        aux_.append(convert_in[i][1].transform(prices.reshape((-1, 1))))
        input_scaled = np.stack(aux_).transpose()[0]
        
        pred = NN.predict(input_scaled)
        prediction = np.ravel(convert_out[i].inverse_transform(pred))
        q_hat = np.array([prediction, np.zeros(len(prediction))]).max(axis = 0)
        q_hat = np.exp(-model['r']*model['dt'])*q_hat
        q.append(list(q_hat - imm_pay))
    if display_time:
        print('Time =', np.round(time.time()-start_time,2), 'sec')
    return (steps, prices, np.array(q).transpose())

#################################################################
### R Script -- testModel[[5]]                                ###
### European Price = 2.213118                                 ###
### c(nnSolve$p[1], mean(oos.nn$payoff), nnSolve$timeElapsed) ###
###    2.379464         2.410582              207.737845      ###
#################################################################
'''
One Neural Network

dt = 0.1
Trial 1 -- Total Time: 103.19 sec

nSims = 10000 epoch_num = 25    
Price Backward 2.424804748833269
nSims = 100000
Price Forward 2.3599486171656987

Trial 2 -- Total Time: 745.27 sec

nSims = 30000 epoch_num = 25    
Price Backward 2.4196599996314685
nSims = 100000
Price Forward 2.3951956211449894

Trial 3 -- Total Time: 2236.09 sec

nSims = 300000 epoch_num = 25    
Price Backward 2.411587602968658
nSims = 100000
Price Forward 2.407359624077011

--------------------------------

dt = 0.05
Trial 1 -- Total Time: 960.22 sec

nSims = 30000 epoch_num = 25    
Price Backward 2.4325111309394942
nSims = 100000
Price Forward 2.37072064505209

Trial 2 -- Total Time: 596.58 sec

nSims = 30000 epoch_num = 15    
Price Backward 2.4241134088731133
nSims = 100000
Price Forward 2.4048027531003635

Trial 3 -- Total Time: 1872.55 sec

nSims = 100000 epoch_num = 15    
Price Backward 2.4216881433126365
nSims = 100000
Price Forward 2.401671509234611
'''

model = {'dim': 1, 'K': 40, 'x0': np.repeat(40, 1), 'sigma': np.repeat(0.2,1), 
         'r': 0.05, 'div': 0, 'T': 1, 'dt': 0.05, 'payoff.func': 'put.payoff'}

start_time = time.time()
np.random.seed(15)
tf.random.set_seed(15)
nSims = 10000
ep_num = 20
stock = stock_sim(nSims, model)
(vq, stop, NNet, c_in, c_out) = NN_back(stock, model, epoch_num = ep_num)
print('Price Backward', vq)

np.random.seed(16)
tf.random.set_seed(16)
nSimsFwd = 100000
stockFwd = stock_sim(nSimsFwd, model)
v_  = NN_fwd(stockFwd, NNet, c_in, c_out, model)
print('Price Forward', v_)
print('Total Time =', np.round(time.time()-start_time, 2),'sec')
 
sns.histplot(data=stop, palette="tab10", linewidth=0.5)
plt.tight_layout()


normalize = matplotlib.colors.Normalize(vmin=-5, vmax=5)
(x,y,z) = NN_bound(NNet, c_in, c_out, model, display_time = False)

contour = plt.contour(x, y, z, [0], colors='black')
plt.clabel(contour, inline=True, fontsize=5)
plt.imshow(np.array(z, dtype = float), extent=[0, 1, 30, 41], aspect='auto', \
           origin='lower', cmap='Spectral', norm = normalize, alpha=1)
plt.colorbar()
plt.savefig('One-NN-Put-Bound-dt-'+str(model['dt']) + '-nSims-' + str(nSims) \
            + '-epoch_num-' + str(ep_num) +'.png', dpi=1000)
plt.clf()

#################################################################
### R Script -- testModel[[4]]                                ###
### European Price = 17.44518                                 ###
### c(nnSolve$p[1], mean(oos.nn$payoff), nnSolve$timeElapsed) ###
###    17.63140          17.44039             510.09126       ###
#################################################################
'''
One Neural Network

Trial 1 -- Total Time: 375.89 sec

nSims = 30000 epoch_num = 50    
Price Backward 17.257572927557014
nSims = 100000
Price Forward 12.852889629302316 --- catastrophic forgetting

New Approach 2:1 Ratio between old and new data

Trial 1 -- Total Time: 696.2 sec

nSims = 30000 epoch_num = 25    
Price Backward 17.05108447921177
nSims = 100000
Price Forward 16.472940374580553
'''

model = {'dim': 2, 'K': 100, 'x0': np.repeat(100, 2), 'sigma': np.repeat(0.2,2), 
         'r': 0.05, 'div': 0, 'T': 1, 'dt': 0.05, 'payoff.func': 'maxi.call.payoff'}


start_time = time.time()
np.random.seed(15)
tf.random.set_seed(15)
nSims = 30000
stock = stock_sim(nSims, model)
(vq, stop, NNet, c_in, c_out) = NN_back(stock, model, epoch_num = 25)
print('Price Backward', vq)

np.random.seed(16)
tf.random.set_seed(16)
nSimsFwd = 100000
stockFwd = stock_sim(nSimsFwd, model)
v_  = NN_fwd(stockFwd, NNet, c_in, c_out, model)
print('Price Forward', v_)
print('Total Time =', np.round(time.time()-start_time, 2),'sec')

# sns.histplot(data=stop, palette="tab10", linewidth=0.5)
# plt.tight_layout()

normalize = matplotlib.colors.Normalize(vmin=-20, vmax=20)
nSteps = int(model['T']/model['dt'])
contours = []
for i in range(nSteps-1):
    start_time = time.time()
    (x,y,z) = NN_contour(i, NNet, c_in, c_out, model, display_time = False)
    
    contours.append(plt.contour(x, y, z, [0], colors='black'))
    plt.clabel(contours[i], inline=True, fontsize=10)
    plt.imshow(np.array(z, dtype = float), extent=[60, 140, 60, 140], \
               origin='lower', cmap='Spectral', norm = normalize, alpha=1)
    plt.colorbar()
    plt.savefig('Maxi-Call-Map'+str(i)+'.png', dpi=1000)
    plt.clf()
    print('Step i =',i+1,'Time =', np.round(time.time()-start_time,2), 'sec')

#################################################################
### R Script -- testModel[[1]]                                ###
### European Price = 1.229925                                 ###
### c(nnSolve$p[1], mean(oos.nn$payoff), nnSolve$timeElapsed) ###
###    1.470019          1.461580            611.567168       ###
#################################################################
'''
Trial 1 -- Total Time: 308.53 sec

nSims = 30000 epoch_num = 50    
Price Backward 1.4665750409531728
nSims = 100000
Price Forward 1.4618117864139293

Trial 2 -- Total Time: 885.41 sec

nSims = 100000 epoch_num = 50    
Price Backward 1.4534664076812505
nSims = 300000
Price Forward 1.449784746205664
'''

model = {'dim': 2, 'K': 40, 'x0': np.repeat(40, 2), 'sigma': np.repeat(0.2,2), 
         'r': 0.06, 'div': 0, 'T': 1, 'dt': 0.04, 'payoff.func': 'put.payoff'}

start_time = time.time()
np.random.seed(15)
tf.random.set_seed(15)
nSims = 100000
stock = stock_sim(nSims, model)
(vq, stop, NNet, c_in, c_out) = NN_back(stock, model, epoch_num = 50)
print('Price Backward', vq)

np.random.seed(16)
tf.random.set_seed(16)
nSimsFwd = 300000
stockFwd = stock_sim(nSimsFwd, model)
v_  = NN_fwd(stockFwd, NNet, c_in, c_out, model)
print('Price Forward', v_)
print('Total Time =', np.round(time.time()-start_time, 2),'sec')

# sns.histplot(data=stop, palette="tab10", linewidth=0.5)
# plt.tight_layout()

nSteps = int(model['T']/model['dt'])
contours = []
for i in range(nSteps-1):
    (x,y,z) = NN_contour(i, NNet, c_in, c_out, model)
    
    contours.append(plt.contour(x, y, z, [0], colors='black'))
    plt.clabel(contours[i], inline=True, fontsize=10)
    plt.imshow(np.array(z, dtype = float), extent=[30, 50, 30, 50], \
               origin='lower', cmap='Spectral', norm = normalize, alpha=1)
    plt.colorbar()
    plt.savefig('PMap'+str(i)+'.png', dpi=1000)
    plt.clf()

#################################################################
### R Script -- testModel[[2]]                                ###
### European Price = 9.636318                                 ###
### c(nnSolve$p[1], mean(oos.nn$payoff), nnSolve$timeElapsed) ###
###    11.30885        11.17898               252.59384       ###
#################################################################
'''
Trial 1 -- Total Time: 101.3 sec

nSims = 30000 epoch_num = 50    
Price Backward 11.193451012635686
nSims = 100000
Price Forward 10.902269528042986

Trial 2 -- Total Time: 825.78 sec

nSims = 300000 epoch_num = 50    
Price Backward 11.352815854116255
nSims = 300000
Price Forward 11.183488239419653
'''

model = {'dim': 3, 'K': 100, 'x0': np.repeat(90, 3), 'sigma': np.repeat(0.2,3), 
         'r': 0.05, 'div': 0.1, 'T': 3, 'dt': 1/3, 'payoff.func': 'maxi.call.payoff'}

start_time = time.time()
np.random.seed(15)
tf.random.set_seed(15)
nSims = 300000
stock = stock_sim(nSims, model)
(vq, stop, NNet, c_in, c_out) = NN_back(stock, model, epoch_num = 50)
print('Price Backward', vq)

np.random.seed(16)
tf.random.set_seed(16)
nSimsFwd = 600000
stockFwd = stock_sim(nSimsFwd, model)
v_  = NN_fwd(stockFwd, NNet, c_in, c_out, model)
print('Price Forward', v_)
print('Total Time =', np.round(time.time()-start_time, 2),'sec')

# sns.histplot(data=stop, palette="tab10", linewidth=0.5)
# plt.tight_layout()

#################################################################
### R Script -- testModel[[3]]                                ###
### European Price = 11.13888                                 ###
### c(nnSolve$p[1], mean(oos.nn$payoff), nnSolve$timeElapsed) ###
###    11.78463         11.57840             195.64841        ###
#################################################################
'''
Trial 1 -- Total Time: 75.89 sec

nSims = 30000 epoch_num = 50    
Price Backward 11.656426664118996
nSims = 100000
Price Forward 11.327571453813313

Trial 2 -- Total Time: 497.61 sec

nSims = 300000 epoch_num = 50    
Price Backward 11.652054666872267
nSims = 300000
Price Forward 11.473564338786465

Trial 3 -- Total Time: 870.0 sec

nSims = 500000 epoch_num = 50    
Price Backward 11.839813495661492
nSims = 1000000
Price Forward 11.666085599090705
'''

model = {'dim': 5, 'K': 100, 'x0': np.repeat(70, 5), 
         'sigma': np.array([0.08,0.16,0.24,0.32,0.4]), 
         'r': 0.05, 'div': 0.1, 'T': 3, 'dt': 1/3, 'payoff.func': 'maxi.call.payoff'}

start_time = time.time()
np.random.seed(15)
tf.random.set_seed(15)
nSims = 500000
stock = stock_sim(nSims, model)
(vq, stop, NNet, c_in, c_out) = NN_back(stock, model, epoch_num = 50)
print('Price Backward', vq)

np.random.seed(16)
tf.random.set_seed(16)
nSimsFwd = 1000000
stockFwd = stock_sim(nSimsFwd, model)
v_  = NN_fwd(stockFwd, NNet, c_in, c_out, model)
print('Price Forward', v_)
print('Total Time =', np.round(time.time()-start_time, 2),'sec')

# sns.histplot(data=stop, palette="tab10", linewidth=0.5)
# plt.tight_layout()