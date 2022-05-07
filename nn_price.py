# Prices options using the Longstaff Schwartz Algorithm
# Implementation of a sequence of neural networks in LSM 
# Implementation of a single neural network in LSM

from payoffs import payoff

## Libraries
import time
import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.initializers import TruncatedNormal

def NN_seq_price(stock, NN, convert_in, convert_out, model, display_time = False):
    '''
    Prices options using the Longstaff Schwartz Algorithm
    Computes the option price using a trained sequence of neural network objects

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
    display_time : Display time spent per step. The default is False.

    Returns
    -------
    val : Option price
    '''
    nSims = len(stock[0])                   # Number of Simulations
    nSteps = int(model['T']/model['dt'])    # Number of Steps
    # List of initial stock prices
    x0 = np.reshape(np.repeat(model['x0'], nSims), (nSims, model['dim']), order = 'F')
    imm_pay = payoff(x0, model) # Vector storing immedaite payoffs
    v_cont = payoff(x0, model) # Array storing continuation values
    
    # List of arrays with stopping decisions: 
    # True means continue, False means stop 
    cont = []
    
    # Forward Loop
    for i in range(0, nSteps-1):
        if display_time:
            start_time = time.time()
        
        # Scaling Neural Network inputs
        aux_ = []
        for j in range(model['dim']):
            aux_.append(convert_in[i][j].transform(stock[i].transpose()[j].reshape((-1, 1))))
        input_scaled = np.stack(aux_).transpose()[0]
        
        # Predicting continuation values 
        # Scaling Neural Network outputs
        pred = NN[i].predict(input_scaled)
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
        v_cont = np.exp(model['r']*model['dt'])*v_cont
        for k in range(nSims):
            if i == 0:
                if cont[-1][k] == False:
                    v_cont[k] = imm_pay[k]
            else:
                if (cont[-1][k] == False) and (cont[-2][k] == True):
                    v_cont[k] = imm_pay[k]
        
        # Displaying Time per Step
        if display_time:
            print('Step i =',i+1,'Time =', np.round(time.time()-start_time,2), 'sec')
    
    # Computing the terminal payoff
    imm_pay = payoff(stock[-1], model)
    v_cont = np.exp(model['r']*model['dt'])*v_cont
    for k in range(nSims):
        if cont[-1][k] == True:
            v_cont[k] = imm_pay[k]
    
    # Discounting and computing the option price
    val = np.mean(np.exp(-model['r']*model['T'])*v_cont)
    return val

def NN_one_price(stock, NN, convert_in, convert_out, model, display_time = False):
    '''
    Prices options using the Longstaff Schwartz Algorithm 
    Computes the option price using a single trained neural network

    Parameters
    ----------
    stock : Array of shape M x N x model['dim'] of simulated stock paths
            M is the number of steps
            N is the number of simulations
            model['dim'] is the number of dimensions
    NN : Neural network 
    convert_in : List of input scaling objects (size M-1)
    convert_out : List of output scaling objects (size M-1)
    model : Dictionary containing all the parameters of the stock and contract
    display_time : Display time spent per step. The default is False.

    Returns
    -------
    val : Option price
    '''
    nSims = len(stock[0])                   # Number of Simulations
    nSteps = int(model['T']/model['dt'])    # Number of Steps
    # List of initial stock prices
    x0 = np.reshape(np.repeat(model['x0'], nSims), (nSims, model['dim']), order = 'F')
    imm_pay = payoff(x0, model) # Vector storing immedaite payoffs
    v_cont = payoff(x0, model) # Array storing continuation values
    
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
        v_cont = np.exp(model['r']*model['dt'])*v_cont
        for k in range(nSims):
            if i == 0:
                if cont[-1][k] == False:
                    v_cont[k] = imm_pay[k]
            else:
                if (cont[-1][k] == False) and (cont[-2][k] == True):
                    v_cont[k] = imm_pay[k]
        
        # Displaying Time per Step
        if display_time:
            print('Step i =',i+1,'Time =', np.round(time.time()-start_time,2), 'sec')
    
    # Computing the terminal payoff
    imm_pay = payoff(stock[-1], model)
    v_cont = np.exp(model['r']*model['dt'])*v_cont
    for k in range(nSims):
        if cont[-1][k] == True:
            v_cont[k] = imm_pay[k]
    
    # Discounting and computing the option price
    val = np.mean(np.exp(-model['r']*model['T'])*v_cont)
    return val

def NN_price(stock, NN, convert_in, convert_out, model, net, display_time = False):
    '''
    Aggregartes the pricing of options in LSM using either a single neural 
    network or a sequence of network objects. 
        
    Parameters
    ----------
    net : Selects the pricing option of LSM
    Other parameters are either from NN_seq_price or NN_one_price
    
    Raises
    ------
    TypeError : Neural network type has not been properly selected.
    
    Returns
    -------
    Return of either NN_seq_price or NN_one_price
    '''
    if net == 'seq':
        return NN_seq_price(stock, NN, convert_in, convert_out, model, display_time)
    elif net == 'single':
        return NN_one_price(stock, NN, convert_in, convert_out, model, display_time)
    else:
        raise TypeError('Neural network type has not been properly selected.')