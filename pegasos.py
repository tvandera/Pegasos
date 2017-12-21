#!/usr/bin/env python3
# Implementation of Pegasos Algorithm
# Described in Paper: http://www.cs.huji.ac.il/~shais/papers/ShalevSiSrCo10.pdf

import os
import math
import pickle

import numpy as np
from sklearn.datasets import load_svmlight_file

# Primary Computation of Gram Matrix can be done in Parallel, as operations
# are independent one another
class gram_matrix(object):
    def __init__(self, TrainingSamples, Kernel, pre_compute):
        self.samples = TrainingSamples
        self.kernel = Kernel
        self.pre_computed = pre_compute
        if pre_compute:
            self.data = np.empty((dim,dim))
            for i in range(dim):
                self.data[j,:] = self.compute_row(i)

    def dim(self):
        return self.samples.shape[0]

    def compute_row(self, i):
        row = np.empty(self.dim())
        for j in range(self.dim()):		
            row[j] = self.kernel(self.samples[i], self.samples[j])
        return row

    def row(self, i):
        if self.pre_computed:
            return self.data[i,:]
        return self.compute_row(i)
		


# These are Essentially Wrappers with Paramaters
# Which will Return the Corresponding Kernel Functions
def radial_basis(gamma):
    assert gamma>0 , "Parameter in Radial Basis function is not a float > 0"
    def function(vector1, vector2):
        from scipy.sparse.linalg import norm
        new_vector = vector1 - vector2
        return math.exp(-1 * gamma * norm(new_vector) ** 2)
    return function

def homogeneous_polynomial(degree):
    assert isinstance(degree, int) and degree>0, "Parameter in Homogeneous Polynomial function is not an integer > 0" 
    def function(vector1, vector2):
        value = vector1.dot(vector2.transpose()).sum()
        return value ** degree
    return function

def inhomogeneous_polynomial(degree):
    assert isinstance(degree, int) and degree>0, "Parameter in Homogeneous Polynomial function is not an integer > 0" 
    def function(vector1, vector2):
        value = vector1.dot(vector2.transpose()) + 1
        return value ** degree
    return function

def linear(): return homogeneous_polynomial(1)

def hyperbolic_tangent(kappa, c):
    assert kappa>0 and c<0, "Parameter in Hyperbolic Tangent Function is not Valid, kappa must be > 0 and c must be < 0"
    def function(vector1, vector2):
        value = vector1.dot(vector2.transpose())
        return math.tanh(kappa*value+c)
    return function


# Entry Into the SVM Solver
# TrainingFilename  -   Name of File Input                  - String
# TestingFilename   -   Name of File Input                  - String
# Kernel            -   Kernel Funtion                      - Function Returing Integer Type
# niter             -   Iterations for Gradient Descent     - Integer
# Eta               -   Learning Rate                       - Float
# SupportVecFile    -   File to Save Support Vectors. Saving,
#                   -   is useful, since it can be a costly operation
#                   -   to constantly find the support vectors
def main(TrainingFilename, TestingFilename, Kernel, niter, compute_kernel_matrix = True, eta = .001, SupportVecFile = ""):
    TrainingSamples, TrainingLabels=load_svmlight_file(TrainingFilename)
    print("Loaded %d Training Samples with %d features each" % (TrainingSamples.shape))

    TestingSamples, TestingLabels=load_svmlight_file(TestingFilename)
    print("Loaded %d Testing Samples" % (TestingSamples.shape[0]))

    # Once we compute the Gram Matrix Once, We may
    # Pickle it so that we don't have to recompute
    # the Data Structure because Computing the Gram
    # Matrix is a generally costly operation
	
    print("Computing Gram Matrix")
    GramMatrix = gram_matrix(TrainingSamples, Kernel, compute_kernel_matrix)

    print("Computed Gram Matrix (%d x %d)" % (GramMatrix.dim(),GramMatrix.dim()))

    # Apply Gradient Descent SVM Solver and Generate
    # Necessary Support Vectors
    print("Computing %d Iteration of Pegasos" % niter)
    Coeffecients, SupportVectors=Pegasos(TrainingSamples, TrainingLabels, eta, niter, GramMatrix)
    print('Completed Pegasos')

    if(SupportVecFile and isinstance(SupportVecFile, str)):
        with open(SupportVecFile, "wb") as fh:
            pickle.dump((Coeffecients, SupportVectors), fh)

    # Run Tests using the Support Vectors For Classification
    error=RunTests(Coeffecients, SupportVectors, Kernel, TestingSamples, TestingLabels)
    print("%d errors out of %d" % (error, TestingSamples.shape[0]))


def Pegasos(TrainingSamples, TrainingLabels, eta, niter, GramMatrix):
    nsamples = TrainingSamples.shape[0]
    a = np.zeros(nsamples)
    time = 1
    for i in range(niter):
        print("Iteration %d Started" % i)
        for t in range(nsamples):
            wx = a.dot(GramMatrix.row(t))
            a *= float(1-1/time)
            # if mispredicted
            if(TrainingLabels[t]*wx < 1):
                a[t] += float(TrainingLabels[t])/float(eta*time)
            time += 1

    SV = TrainingSamples[a != .0]
    a = a[a!=.0]
    print("%d support vectors" % len(a))
    return a, SV


# Assigns a Class Label to a Particular Sample
def Predictor(a, SV, Kernel, Sample):
    accumulator = 0	
    for index in range(len(a)):
        accumulator += a[index]*Kernel(SV[index], Sample)
    if accumulator < 0: return -1
    return 1

def RunTests(a, SV, Kernel, TestingSamples, TestingLabels):
    errors = 0
    nsamples = TestingSamples.shape[0]
    assert nsamples == TestingLabels.shape[0]
    for i in range(nsamples):
        Sample = TestingSamples.getrow(i)
        correctlabel = TestingLabels[i]
        predictedlabel=Predictor(a, SV, Kernel, Sample)
        if(predictedlabel*correctlabel<0): errors += 1
    return errors

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="""Implementation of Pegasos Algorithm
    Described in Paper: http://www.cs.huji.ac.il/~shais/papers/ShalevSiSrCo10.pdf""")

    parser.add_argument('--train',  metavar='FILE', dest='train_file',  help='Train Data SVM', required=True)
    parser.add_argument('--test',   metavar='FILE', dest='test_file',   help='Test Data')
    parser.add_argument('--iter',   metavar='NUM', dest='iter',         help='Number of iterations', default = 5)
    parser.add_argument('--output', metavar='FILE', dest='output_file', help='Output Model')
    args = parser.parse_args()
    print(args)

    main(args.train_file, args.test_file, radial_basis(1.0), args.iter, False, SupportVecFile = args.output_file)
