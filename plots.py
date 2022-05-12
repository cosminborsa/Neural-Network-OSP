from payoffs import payoff

## Libraries
import time
import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.initializers import TruncatedNormal

def NN_seq_contour(step, NN, convert_in, convert_out, model, 
               down = 0.6, up = 1.4, inc = 0.1, display_time = False):
    '''
    Returns contour parameters for 2-dim options at a particular step 
    from a sequence of neural network objects

    Parameters
    ----------
    step : Time step of the contour
    NN : List of Neural Network objects (size M-1)
    convert_in : List of input scaling objects (size M-1)
    convert_out : List of output scaling objects (size M-1)
    model : Dictionary containing all the parameters of the stock and contract
    down : Scaling parameter for the lower bound of the map. 
           The default is 0.6 or 60% of the initial price.
    up : Scaling parameter for the upper bound of the map. 
         The default is 1.4 or 140% of the initial price.
    inc : Increment for the stock price grid. The default is 0.1.
    display_time : Display time spent per step. The default is False.
    
    Raises
    ------
    ValueError : Time step is out of range.
    TypeError : Only works for 2-dim contracts.

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
        raise TypeError('Only works for 2-dim options.')
    # Designing the stock price grid for the contour
    prices = []
    for x0 in model['x0']:
        prices.append(np.arange(x0*down, x0*up, inc))
    n = len(prices[0])
    x, y = np.meshgrid(prices[0], prices[1])
    aux = []
    for a, b in zip(x, y):
        aux.append(np.reshape(np.ravel([a,b]), (n,2), order = 'F'))
    prices_ = np.reshape(np.ravel(aux), (n**2, 2), order='C')
    # Immediate payoff of the stock price grid
    imm_pay = payoff(prices_, model)
    # Transforming the input prices for the neural network
    aux_ = []
    for j in range(model['dim']):
        aux_.append(convert_in[i][j].transform(prices_.transpose()[j].reshape((-1, 1))))
    input_scaled = np.stack(aux_).transpose()[0]
    
    # Predicting continuation values and scaling neural network outputs
    pred = NN[i].predict(input_scaled)
    prediction = np.ravel(convert_out[i].inverse_transform(pred))
    q_hat = np.array([prediction, np.zeros(len(prediction))]).max(axis = 0)
    q_hat = np.exp(-model['r']*model['dt'])*q_hat
    q = q_hat - imm_pay 
    q = np.reshape(q, (n, n))
    if display_time:
        print('Time =', np.round(time.time()-start_time,2), 'sec')
    return (prices[0], prices[1], q)

def NN_one_contour(step, NN, convert_in, convert_out, model, 
               down = 0.6, up = 1.4, inc = 0.1, display_time = False):
    '''
    Returns contour parameters for 2-dim options at a particular time-step
    from a single neural network object
    
    Parameters
    ----------
    step : Time step of the contour
    NN : List of Neural Network objects (size M-1)
    convert_in : List of input scaling objects (size M-1)
    convert_out : List of output scaling objects (size M-1)
    model : Dictionary containing all the parameters of the stock and contract
    down : Scaling parameter for the lower bound of the map. 
           The default is 0.6 or 60% of the initial price.
    up : Scaling parameter for the upper bound of the map. 
         The default is 1.4 or 140% of the initial price.
    inc : Increment for the stock price grid. The default is 0.1.
    display_time : Display time spent per step. The default is False.

    Raises
    ------
    ValueError : Time step is out of range.
    TypeError : Only works for 2-dim contracts.

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
        raise TypeError('Only works for 2-dim options.')
    prices = []
    for x0 in model['x0']:
        prices.append(np.arange(x0*down, x0*up, inc))
    n = len(prices[0])
    x, y = np.meshgrid(prices[0], prices[1])
    aux = []
    for a, b in zip(x, y):
        aux.append(np.reshape(np.ravel([a,b]), (n,2), order = 'F'))
    prices_ = np.reshape(np.ravel(aux), (n**2, 2), order='C')
    # Immediate payoff of the stock price grid
    imm_pay = payoff(prices_, model)
    # Transforming the input prices for the neural network
    aux_ = [convert_in[i][0].transform(np.repeat(i,n**2).reshape(-1, 1))]
    for j in range(model['dim']):
        aux_.append(convert_in[i][j+1].transform(prices_.transpose()[j].reshape((-1, 1))))
    input_scaled = np.stack(aux_).transpose()[0]
    
    # Predicting continuation values and scaling neural network outputs
    pred = NN.predict(input_scaled)
    prediction = np.ravel(convert_out[i].inverse_transform(pred))
    q_hat = np.array([prediction, np.zeros(len(prediction))]).max(axis = 0)
    q_hat = np.exp(-model['r']*model['dt'])*q_hat
    q = q_hat - imm_pay 
    q = np.reshape(q, (n, n))
    if display_time:
        print('Time =', np.round(time.time()-start_time,2), 'sec')
    return (prices[0], prices[1], q)

def NN_seq_bound(NN, convert_in, convert_out, model, down = 0.75, up = 1.01, 
             inc = 0.05, display_time = False):
    '''
    Returns bound parameters for 1-dim options from a sequence of neural network objects

    Parameters
    ----------
    NN : List of Neural Network objects (size M-1)
    convert_in : List of input scaling objects (size M-1)
    convert_out : List of output scaling objects (size M-1)
    model : Dictionary containing all the parameters of the stock and contract
    down : Scaling parameter for the lower bound of the stock price. 
           The default is 0.75 or 75% of the initial price.
    up : Scaling parameter for the upper bound of the stock price. 
         The default is 1.01 or 101% of the initial price.
    inc : Increment for the stock price grid. The default is 0.05.
    display_time : Display time spent per step. The default is False.
    
    Raises
    ------
    TypeError : Only works for 1-dim contracts.

    Returns
    -------
    Tuple consisting of bound parameters
    '''
    if display_time:
        start_time = time.time()
    
    nSteps = int(model['T']/model['dt'])
    if model['dim'] != 1:
        raise TypeError('Only works for 1-dim options.')
    prices = np.arange(model['x0']*down, model['x0']*up, inc)
    n = len(prices)
    steps = np.arange(model['dt'], model['T'], model['dt']) 
    # Immediate payoff of the stock prices
    imm_pay = payoff(np.reshape(prices, (n,1)), model)
    
    # Predicting continuation values and scaling neural network outputs
    q = []              # List of continuation values
    for i in range(nSteps-1):
        input_scaled = convert_in[i][0].transform(prices.reshape((-1, 1)))
        pred = NN[i].predict(input_scaled)
        prediction = np.ravel(convert_out[i].inverse_transform(pred))
        q_hat = np.array([prediction, np.zeros(len(prediction))]).max(axis = 0)
        q_hat = np.exp(-model['r']*model['dt'])*q_hat
        q.append(list(q_hat - imm_pay))
    if display_time:
        print('Time =', np.round(time.time()-start_time,2), 'sec')
    return (steps, prices, np.array(q).transpose())

def NN_one_bound(NN, convert_in, convert_out, model, down = 0.75, up = 1.01, 
             inc = 0.05, display_time = False):
    '''
    Returns bound parameters for 1-dim options from a single neural network object

    Parameters
    ----------
    NN : List of Neural Network objects (size M-1)
    convert_in : List of input scaling objects (size M-1)
    convert_out : List of output scaling objects (size M-1)
    model : Dictionary containing all the parameters of the stock and contract
    down : Scaling parameter for the lower bound of the stock price. 
           The default is 0.75 or 75% of the initial price.
    up : Scaling parameter for the upper bound of the stock price. 
         The default is 1.01 or 101% of the initial price.
    inc : Increment for the stock price grid. The default is 0.05.
    display_time : Display time spent per step. The default is False.
    
    Raises
    ------
    TypeError : Only works for 1-dim contracts.

    Returns
    -------
    Tuple consisting of bound parameters
    '''
    if display_time:
        start_time = time.time()
    
    nSteps = int(model['T']/model['dt'])
    if model['dim'] != 1:
        raise TypeError('Only works for 1-D options.')
    
    prices = np.arange(model['x0']*down, model['x0']*up, inc)
    n = len(prices)
    steps = np.arange(model['dt'], model['T'], model['dt']) 
    # Immediate payoff of the stock prices
    imm_pay = payoff(np.reshape(prices, (n,1)), model)
    
    # Predicting continuation values and scaling neural network outputs
    q = []                 # List of continuation values
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

def NN_plot(NN, convert_in, convert_out, model, net, step = None, down = None, up = None, 
             inc = None, display_time = False):
    '''
    Aggregates the plot functions

    Parameters
    ----------
    net : Selects the pricing option of LSM
    Other parameters are needed for NN_seq_bound, NN_one_bound, NN_seq_contour, 
    or NN_one_contour
       
    Raises
    ------
    TypeError : Only works for 1-dim or 2-dim contracts.
                Neural network type has not been properly selected.
    AttributeError : The time step has not been assigned.

    Returns
    -------
    Returns bound or contour parameters
    '''
    if model['dim'] == 1:
        # Setting up defaults
        if down == None:
            down = 0.75
        if up == None:
            up = 1.01
        if inc == None:
            inc = 0.05
        if display_time == None:
            display_time = False
            
        if net == 'seq':
            return NN_seq_bound(NN, convert_in, convert_out, model, down, up, inc, 
                                display_time)
        elif net == 'single':
            return NN_one_bound(NN, convert_in, convert_out, model, down, up, inc, 
                                display_time)
        else:
            raise TypeError('Neural network type has not been properly selected.')
            
    elif model['dim'] == 2:
        # Setting up defaults
        if down == None:
            down = 0.6
        if up == None:
            up = 1.4
        if inc == None:
            inc = 0.1
        if display_time == None:
            display_time = False
        if step == None:
            raise AttributeError('The time step has not been assigned.')
        
        if net == 'seq':
            return NN_seq_contour(step, NN, convert_in, convert_out, model, 
                           down, up, inc, display_time)
        elif net == 'single':
            return NN_one_contour(step, NN, convert_in, convert_out, model, 
                           down, up, inc, display_time)
        else:
            raise TypeError('Neural network type has not been properly selected.')
    else:
        raise TypeError('Only works for 1-dim or 2-dim options.')