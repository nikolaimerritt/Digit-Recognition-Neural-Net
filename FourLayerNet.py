from typing import List, Tuple
import numpy as np
from Params import Params, paramsSum

def relu(x):
    return x * (x > 0)


def drelu(x):
    return 1.0 * (x > 0)

def squaredNorm(ndarray):
    return (ndarray * ndarray).sum()

def sumOfSquaresOfEntries(ndarrays):
    return sum(squaredNorm(ndarray) for ndarray in ndarrays)


def getRandomParams(inputLayerSize: int, fstLayerSize: int, sndLayerSize: int, thdLayerSize: int) -> Params:        
    fstWeights = np.random.rand(fstLayerSize, inputLayerSize)
    fstBiases = np.random.rand(fstLayerSize)
    
    sndWeights = np.random.rand(sndLayerSize, fstLayerSize)
    sndBiases = np.random.rand(sndLayerSize)
    
    thdWeights = np.random.rand(thdLayerSize, sndLayerSize)
    thdBiases = np.random.rand(thdLayerSize)

    return Params([fstWeights, sndWeights, thdWeights], [fstBiases, sndBiases, thdBiases])


def outputLayer(inputLayer: np.ndarray, params: Params):
    layer = inputLayer
    for weight, bias in zip(params.weights, params.biases):
        layer = relu(weight @ layer + bias)
    return layer
    

def paramsGrad(self, inputLayer: np.ndarray, desiredOutputLayer: np.ndarray, params: Params):
    """ computes the gradient of each weight and bias with one single output layer """
    """ THE MATHS: 
    Write: 
        a as the 0th input layer
        x, y, z as the 1st, 2nd, 3rd output layers before they have been relu'ed
        F, S, T as the 1st, 2nd, 3rd weights
        f, s, t as the 1st, 2nd, 3rd biases
        w as the desired output layer
    """
    a = inputLayer
    F, S, T = params.weights
    f, s, t = params.biases
    w = desiredOutputLayer
    """
    Then the cost is given by:
        x_l = F_lm a_m + f_l                    (sum over m)
        y_k = S_kl r(x_l) + s_k                 (sum over l)
        z_j = T_jk r(y_k) + t_j                 (sum over k)

        C = (r(z_j) - w_j)^2                    (sum over j)
    """
    x = F @ a + f
    y = S @ relu(x) + s
    z = T @ relu(y) + t
    """
    So the derivatives of C wrt x, y, z are:
        dC/dz_j = 2 (r(z_j) - w_j) r'(z_j)
        dC/dy_k =  T_jk dC/dz_j r'(y_k)          (sum over j)
        dC/dx_l = dC/dy_k r'(x_l)               (sum over k)
    """
    dC_dz = 2 * (relu(z) - w) * drelu(z)
    dC_dy = (np.transpose(T) @ dC_dz) * drelu(y)
    dC_dx = drelu(x) * np.sum(dC_dy)
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


def meanCosts(self, inputLayers: List[np.ndarray], desiredOutputLayers: List[np.ndarray]) -> float:
    costs = [
        squaredNorm(outputLayer(inputLayer) - desiredOutputLayer)
        for inputLayer, desiredOutputLayer in zip(inputLayers, desiredOutputLayers)
    ]
    return np.mean(costs)


def meanParamsGrad(self, inputLayers: List[np.ndarray], desiredOutputLayers: List[np.ndarray]) -> Params:
    paramsGrads = [
        paramsGrad(inputLayer, desiredOutputLayer) 
        for inputLayer, desiredOutputLayer in zip(inputLayers, desiredOutputLayers)
    ]
    return sum(paramsGrads) / len(paramsGrads)


def gradientDescent(self, inputLayers: List[np.ndarray], desiredOutputLayers: List[np.ndarray], params: Params) -> Params:
    stepSize = 0.1
