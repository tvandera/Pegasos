#!/usr/bin/env python3
# Implementation of Pegasos Algorithm
# Described in Paper: http://www.cs.huji.ac.il/~shais/papers/ShalevSiSrCo10.pdf

import os
import math
import pickle

import numpy as np
from scipy.sparse import dok_matrix
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


def get_kernel(name, params):

    # These are Essentially Wrappers with Paramaters
    # Which will Return the Corresponding Kernel Functions
    def radial_basis(gamma = 1.0):
        print("radial base with gamma %f" % (gamma))
        assert gamma>0 , "Parameter in Radial Basis function is not a float > 0"
        def function(vector1, vector2):
            from scipy.sparse.linalg import norm
            new_vector = vector1 - vector2
            return math.exp(-1 * gamma * norm(new_vector) ** 2)
        return function

    def homogeneous_polynomial(degree = 1.):
        print("homogeneous_polynomial with degree %f" % (degree))
        assert isinstance(degree, int) and degree>0, "Parameter in Homogeneous Polynomial function is not an integer > 0" 
        def function(vector1, vector2):
            value = vector1.dot(vector2.transpose()).sum()
            return value ** degree
        return function

    def inhomogeneous_polynomial(degree = 1.):
        print("inhomogeneous_polynomial with degree %f" % (degree))
        assert isinstance(degree, int) and degree>0, "Parameter in Homogeneous Polynomial function is not an integer > 0" 
        def function(vector1, vector2):
            value = vector1.dot(vector2.transpose()) + 1
            return value ** degree
        return function

    def linear(): return homogeneous_polynomial(1)

    def hyperbolic_tangent(kappa = 1., c = -1.):
        print("hyperbolic_tangent with kappa = %f, c = %f" % (kappa, c))
        assert kappa>0 and c<0, "Parameter in Hyperbolic Tangent Function is not Valid, kappa must be > 0 and c must be < 0"
        def function(vector1, vector2):
            value = vector1.dot(vector2.transpose())
            return math.tanh(kappa*value+c)
        return function

    name_map = {
            'radial_basis':             radial_basis,
            'homogeneous_polynomial':   homogeneous_polynomial,
            'inhomogeneous_polynomial': inhomogeneous_polynomial,
            'linear':                   linear
    }

    return name_map[name](*params)

# Entry Into the SVM Solver
# TrainingFilename  -   Name of File Input                  - String
# TestingFilename   -   Name of File Input                  - String
# Kernel            -   Kernel Funtion                      - Function Returing Integer Type
# niter             -   Iterations for Gradient Descent     - Integer
# Eta               -   Learning Rate                       - Float
# SupportVecFile    -   File to Save Support Vectors. Saving,
#                   -   is useful, since it can be a costly operation
#                   -   to constantly find the support vectors
def main(TrainingFilename, TestingFilename, kernel_str, niter, compute_kernel_matrix = True, eta = .001, SupportVecFile = ""):
    TrainingSamples, TrainingLabels=load_svmlight_file(TrainingFilename)
    print("Loaded %d Training Samples with %d features each" % (TrainingSamples.shape))

    TestingSamples, TestingLabels=load_svmlight_file(TestingFilename)
    print("Loaded %d Testing Samples" % (TestingSamples.shape[0]))

    if kernel_str:
        kernel_params = kernel_str.split(",")
        Kernel = get_kernel(kernel_params[0], kernel_params[1:])
        print("Computing Gram Matrix")
        GramMatrix = gram_matrix(TrainingSamples, Kernel, compute_kernel_matrix)
        print("Computed Gram Matrix (%d x %d)" % (GramMatrix.dim(),GramMatrix.dim()))

        print("Computing %d Iteration of KernelPegasos" % niter)
        Coeffecients, SupportVectors=KernelPegasos(TrainingSamples, TrainingLabels, eta, niter, GramMatrix)
        print('Completed Pegasos')
        error=RunTests(Coeffecients, SupportVectors,TestingSamples, TestingLabels, Kernel)
    else:
        print("Computing %d Iteration of LinearPegasos" % niter)
        Coeffecients, SupportVectors=LinearPegasos(TrainingSamples, TrainingLabels, eta, niter)
        print('Completed Pegasos')
        error=RunTests(Coeffecients, SupportVectors, TestingSamples, TestingLabels)

    print("%d errors out of %d" % (error, TestingSamples.shape[0]))


def KernelPegasos(TrainingSamples, TrainingLabels, eta, niter, GramMatrix):
    nsamples = TrainingSamples.shape[0]
    a = dok_matrix((1,nsamples))
    time = 1
    for i in range(niter):
        print("Iteration %d Started" % i)
        for t in range(nsamples):
            label = TrainingLabels[t]
            wx = a.dot(GramMatrix.row(t)).sum()
            a *= float(1-1/time)
            # if mispredicted
            if(wx*label < 1):
                a[0,t] += float(TrainingLabels[t])/float(eta*time)
            time += 1

    nonzeros = [ k[1] for k in a.keys() ]
    SV = TrainingSamples[nonzeros]
    M = np.fromiter(iter(a.values()), dtype=float)
    print("%d support vectors" % a.getnnz())
    return M, SV

def LinearPegasos(TrainingSamples, TrainingLabels, eta, niter):
    nsamples = TrainingSamples.shape[0]
    nfeat = TrainingSamples.shape[1]
    a = dok_matrix((1,nfeat))
    time = 1
    for i in range(niter):
        print("Iteration %d Started" % i)
        for t in range(nsamples):
            sample = TrainingSamples.getrow(t)
            label = TrainingLabels[t]
            wx = sample.dot(a.transpose()).sum()
            a *= float(1-1/time)
            # if mispredicted
            if(wx * label < 1):
                a[0,t] += float(TrainingLabels[t])/float(eta*time)
            time += 1

    nonzeros = [ k[1] for k in a.keys() ]
    SV = TrainingSamples[nonzeros]
    M = np.fromiter(iter(a.values()), dtype=float)
    print("%d support vectors" % a.getnnz())
    return M, SV

# Assigns a Class Label to a Particular Sample
def Predictor(a, SV, Sample, Kernel):
    accumulator = 0	
    for index in range(len(a)):
        distance = Kernel(SV[index], Sample)
        accumulator += a[index]*distance
    if accumulator < 0: return -1
    return 1

def RunTests(a, SV, TestingSamples, TestingLabels, Kernel = lambda a, b: np.dot(a,b.transpose())):
    errors = 0
    nsamples = TestingSamples.shape[0]
    assert nsamples == TestingLabels.shape[0]
    for i in range(nsamples):
        Sample = TestingSamples.getrow(i)
        correctlabel = TestingLabels[i]
        predictedlabel=Predictor(a, SV, Sample, Kernel)
        if(predictedlabel*correctlabel<0): errors += 1
    return errors

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="""Implementation of Pegasos Algorithm
    Described in Paper: http://www.cs.huji.ac.il/~shais/papers/ShalevSiSrCo10.pdf""",
    epilog="""Possible kernel functions:\\n
    radial_basis(gamma = 1.0)\\n
    homogeneous_polynomial(degree = 1.)\\n
    inhomogeneous_polynomial(degree = 1.)\\n
    linear() - same as homogeneous_polynomial(1)\\n
    hyperbolic_tangent(kappa = 1., c = -1.)""")

    parser.add_argument('--train',  metavar='FILE', dest='train_file',  help='Train Data SVM', required=True)
    parser.add_argument('--test',   metavar='FILE', dest='test_file',   help='Test Data')
    parser.add_argument('--iter',   metavar='NUM', dest='iter',         help='Number of iterations', default = 5)
    parser.add_argument('--output', metavar='FILE', dest='output_file', help='Output Model')
    parser.add_argument('--kernel', metavar='NAME,PARAM1,PARAM2', dest='kernel', help='Use Kernel SVM with this kernel')
    args = parser.parse_args()
    print(args)

    main(args.train_file, args.test_file, args.kernel, int(args.iter), False, SupportVecFile = args.output_file)
