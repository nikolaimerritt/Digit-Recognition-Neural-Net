from PIL import Image
import numpy as np
import os

def unbiasedVector(size):
    """ vector is uniform, mean 0, entries [-1, 1)"""
    return (2 * np.random.rand(size) - 1) / 100

def unbiasedMatrix(rows, cols):
    """ matrix is uniform, mean 0, entries [-1, 1)"""
    return (2 * np.random.rand(rows, cols) - 1) / 100

def txtsInFolder(folder: str):
    return [f for f in os.listdir(folder) if f.endswith(".txt")]

def displayProbVector(vector):
    return "    ".join([f"{i}: {(100 * vector[i]):.0f}" for i in range(len(vector))])