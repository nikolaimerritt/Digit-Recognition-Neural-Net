from random import shuffle
from typing import List
import numpy as np
from NeuralNet.Params import Params

def relu(x):
    return x * (x > 0)

def drelu(x):
    return 1.0 * (x > 0)

def squaredNorm(ndarray):
    return (ndarray * ndarray).sum()

def sumOfSquaresOfEntries(ndarrays):
    return sum(squaredNorm(ndarray) for ndarray in ndarrays)

def softmax(x):
    xmax = np.max(x)
    exponentials = np.exp(x - xmax)
    return exponentials / exponentials.sum()


PREVENT_LOG_0: float = 10 ** -9


def calcHiddenLayers(inLayer: np.ndarray, params: Params):
   x = params.weights[0] @ inLayer + params.biases[0]
   y = params.weights[1] @ relu(x) + params.biases[1]
   z = params.weights[2] @ relu(y) + params.biases[2]
   return x, y, z


def calcOutLayer(inLayer: np.ndarray, params: Params):
    _, _, z = calcHiddenLayers(inLayer, params)
    return softmax(z)
    

def singleCost(inLayer: np.ndarray, desOutLayer: np.ndarray, params: Params):
    outLayer = calcOutLayer(inLayer, params)
    return -desOutLayer @ np.log(outLayer + PREVENT_LOG_0)


def paramsGrad(inLayer: np.ndarray, desOutLayer: np.ndarray, params: Params):
    """ computes the gradient of each weight and bias with one single output layer """
    """ THE MATHS: 
    Write: 
        a as the 0th input layer
        x, y, z as the 1st, 2nd, 3rd output layers before they have been relu'ed
        F, S, T as the 1st, 2nd, 3rd weights
        f, s, t as the 1st, 2nd, 3rd biases
        w as the desired output layer
    """
    a = inLayer
    F, S, T = params.weights
    f, s, t = params.biases
    w = desOutLayer
    """
    Then the cost is given by:
        x_l = F_lm a_m + f_l                    (sum over m)
        y_k = S_kl r(x_l) + s_k                 (sum over l)
        z_j = T_jk r(y_k) + t_j                 (sum over k)

        C = -w_j log(softmax(z)_j)                        (sum over j)
          = -w_j z_j + sum(w) log(exp(z_1) + ... + exp(z_n)) (sum over j) 
    """
    x, y, z = calcHiddenLayers(inLayer, params)
    """
    So the derivatives of C wrt x, y, z are:
        dC/dz_j = -w_j + sum(w) softmax(z)_j
        dC/dy_k =  T_jk dC/dz_j r'(y_k)          (sum over j)
        dC/dx_l = dC/dy_k r'(x_l)               (sum over k)
    """
    dC_dz = -w + sum(w) * softmax(z + PREVENT_LOG_0)
    dC_dy = (np.transpose(T) @ dC_dz) * drelu(y)
    dC_dx = drelu(x) * (np.transpose(S) @ dC_dy)
    """
    This means the derivatives wrt the parameters are:
        dC/dt_j = dC/dz_j       dC/dT_jk = dC/dz_j r(y_k)
        dC/ds_k = dC/dy_k       dC/dS_kl = dC/dy_k r(x_l)
        dC/df_l = dC/dx_l       dC/dF_lm = dC/dx_l a_m
    """
    thdBiasGradient = dC_dz
    thdWeightGradient = np.outer(dC_dz, relu(y))

    sndBiasGradient = dC_dy
    sndWeightGradient = np.outer(dC_dy, relu(x))

    fstBiasGradient = dC_dx
    fstWeightGradient = np.outer(dC_dx, a)

    weightsGradient = [fstWeightGradient, sndWeightGradient, thdWeightGradient]
    biasesGradient = [fstBiasGradient, sndBiasGradient, thdBiasGradient]

    return Params(weightsGradient, biasesGradient)


def meanCost(inLayers: List[np.ndarray], desOutLayers: List[np.ndarray], params: Params) -> float:
    costs = [
        singleCost(inLayer, desOutLayer, params) 
        for inLayer, desOutLayer in zip(inLayers, desOutLayers)
    ]
    return np.mean(costs)


def meanParamsGrad(inLayers: List[np.ndarray], desOutLayers: List[np.ndarray], params):
    paramsGrads = [
        paramsGrad(inLayer, desOutLayer, params) 
        for inLayer, desOutLayer in zip(inLayers, desOutLayers)
    ]
    return (1/len(paramsGrads)) * sum(paramsGrads, start=Params.ZERO) 


def recogniseDigit(inLayer, params):
    outLayer = calcOutLayer(inLayer, params)
    return np.argmax(outLayer)


def proportionCorrect(inLayers, desOutLayers, params):
    correct = 0
    for inLayer, desOutLayer in zip(inLayers, desOutLayers):
        guess = recogniseDigit(inLayer, params)
        actual = np.argmax(desOutLayer)
        if guess == actual:
            correct += 1
    
    return correct / len(desOutLayers)


def saveBatchGradDescent(inLayers: List[np.ndarray], desOutLayers: List[np.ndarray], params: Params, testIn, testDesOut, batchSize=256, epochs=50) -> Params:
    learningRate = 10 ** -3
    momentum = 0.9

    indices = list(range(len(inLayers)))
    paramsChange = Params.ZERO
    
    print(" ---------- starting training ----------")
    for _ in range(epochs):
        shuffle(indices)
        
        for batchIdx in range(0, len(inLayers), batchSize):
            batchIndices = indices[batchIdx : batchIdx + batchSize]
            
            inBatch = np.array([inLayers[idx] for idx in batchIndices])
            desOutBatch = np.array([desOutLayers[idx] for idx in batchIndices])
            
            futureGrad = meanParamsGrad(inBatch, desOutBatch, params + momentum * paramsChange)
            paramsChange = momentum * paramsChange + (-learningRate) * futureGrad
            params += paramsChange

        learningScore = proportionCorrect(inLayers, desOutLayers, params)
        testingScore = proportionCorrect(testIn, testDesOut, params)
        print(f"learning score = {100 * learningScore:.3f}% \t testing score = {100 * testingScore:.3f}% \t gsn = {futureGrad.squaredNorm():.6f}")
        
        params.saveToFile()
    return params