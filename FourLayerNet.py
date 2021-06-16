from typing import List
import numpy as np

def relu(x: float) -> float:
    return 0 if x < 0 else x


def drelu_dx(x: float) -> float:
    return 0 if x < 0 else 1


def dataRelu(data: List[float]) -> List[float]:
    return [relu(x) for x in data]


def normalise(data: List[float]) -> List[float]:
    norm = sum(x ** 2 for x in data) ** (1/2)
    return [x / norm for x in data]



class FourLayerNet:
    def __init__(self, inputLayerSize: int, fstLayerSize: int, sndLayerSize: int, thdLayerSize: int) -> None:
        self.inputLayerSize = inputLayerSize
        
        self.fstLayerSize = fstLayerSize
        self.fstWeights = np.random.rand(fstLayerSize, inputLayerSize)
        self.fstBiases = np.random.rand(fstLayerSize)
        
        self.sndLayerSize = sndLayerSize
        self.sndWeights = np.random.rand(sndLayerSize, fstLayerSize)
        self.sndBiases = np.random.rand(sndLayerSize)
        
        self.thdLayerSize = thdLayerSize
        self.thdWeights = np.random.rand(thdLayerSize, sndLayerSize)
        self.thdBiases = np.random.rand(thdLayerSize)
    

    def calcOutputLayer(self, inputLayer):
        layer1 = dataRelu(self.fstWeights @ inputLayer + self.fstBiases)
        layer2 = dataRelu(self.sndWeights @ layer1 + self.sndBiases)
        outputLayer = dataRelu(self.thdWeights @ layer2 + self.thdBiases)
        return outputLayer
    

    def costGradient(self, inputLayer, desiredOutputLayer):
        """ computes the gradient of each weight and bias with one single output layer """
        """ THE MATHS: 
        Put: 
            a as the 0th input layer
            x, y, z as the 1st, 2nd, 3rd output layers before they have been relu'ed
            F, S, T as the 1st, 2nd, 3rd weights
            f, s, t as the 1st, 2nd, 3rd biases
            w as the desired output layer
        """
        a = inputLayer
        F, S, T = self.fstWeights, self.sndWeights, self.thdWeights
        f, s, t = self.fstBiases, self.sndBiases, self.thdBiases
        w = desiredOutputLayer
        """
        Then the cost is given by:
            x_l = F_lm a_m + f_l                    (sum over m)
            y_k = S_kl r(x_l) + s_k                 (sum over l)
            z_j = T_jk r(y_k) + t_j                 (sum over k)

            C = (r(z_j) - w_j)^2                    (sum over j)
        """
        x = F @ a + f
        y = S @ dataRelu(x) + s
        z = T @ dataRelu(y) + t
        """
        So the derivatives of C wrt x, y, z are:
            dC/dz_j = 2 (r(z_j) - w_j) r'(z_j)
            dC/dy_k = dC/dz_j T_jk r'(y_k)          (sum over j)
            dC/dx_l = dC/dy_k r'(x_l)               (sum over k)
        """
        dC_dz = lambda j:   2 * (relu(z[j]) - w[j]) * drelu_dx(z[j])
        dC_dy = lambda k:   sum(dC_dz(j) * T[j][k] * drelu_dx(y[k]) for j in range(self.thdLayerSize))
        dC_dx = lambda l:   sum(dC_dy(k) * drelu_dx(x[l]) for k in range(self.sndLayerSize))
        """
        Which means the derivatives wrt the parameters are:
            dC/dt_j = dC/dz_j       dC/dT_jk = dC/dz_j r(y_k)
            dC/ds_k = dC/dy_k       dC/dS_kl = dC/dy_k r(x_l)
            dC/df_l = dC/dx_l       dC/dF_lm = dC/dx_l a_m
        """
        thdBiasGradient = np.array([
            dC_dz(j) 
            for j in range(self.thdLayerSize)
        ])

        thdWeightGradient = np.array([
            [dC_dz(j) * relu(y[k]) for k in range(self.sndLayerSize)] 
            for j in range(self.thdLayerSize)
        ])

        sndBiasGradient = np.array([
            dC_dy(k)    
            for k in range(self.sndLayerSize)
        ])
        sndWeightGradient = np.array([
            [dC_dy(k) * relu(x[l]) for l in range(self.fstLayerSize)]
            for k in range(self.sndLayerSize)
        ])

        fstBiasGradient = np.array([
            dC_dx(l)
            for l in range(self.fstLayerSize)
        ])
        fstWeightGradient = np.array([
            [dC_dx(l) * a[m] for m in range(self.inputLayerSize)]
            for l in range(self.fstLayerSize)
        ])
        fstWeightGradient.shape = (16, 784)
    
        return fstWeightGradient, fstBiasGradient, sndWeightGradient, sndBiasGradient, thdWeightGradient, thdBiasGradient

    
    def testDimensions(self, inputLayer, desiredOutputLayer):
        fstWeightGradient, fstBiasGradient, sndWeightGradient, sndBiasGradient, thdWeightGradient, thdBiasGradient = self.costGradient(inputLayer, desiredOutputLayer)

        fstWeightGradient - self.fstWeights
        fstBiasGradient - self.fstBiases
        
        sndWeightGradient - self.sndWeights
        sndBiasGradient - self.sndBiases

        thdWeightGradient - self.thdWeights
        thdBiasGradient - self.thdBiases