# Stephan Gelever
# Math 510
# HW 2
# 
# Copyright (c) 2018, Stephan Gelever

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

import scipy.optimize
import numpy as np
import itertools

def plot_figs(x_hist, p_hist, f_hist, A, method):
    fig = plt.figure()
    if (len(x_hist) > 0):
        ax = fig.add_subplot(121)

        lin_space = np.linspace(-2.0, 2.0, 1000)
        x, y = np.meshgrid(lin_space, lin_space)

        z = (A * (y - (x*x))**2 + (1 - x)**2)

        #levels = np.logspace(0, 3, num=10, base=2.0)
        levels = np.linspace(0, 8, num=10)

        ax.contour(x, y, z, cmap=plt.get_cmap("jet_r"), zorder=0, levels=levels)

        c = np.linspace(0, 1.0, len(x_hist))

        scatter = ax.scatter(x=x_hist[:, 0], y=x_hist[:, 1], c=c, cmap=plt.get_cmap(
            "YlGn_r"), zorder=1, label=method)
        ax.legend()

    ax = fig.add_subplot(222)
    #plot_type = ax.semilogy if len(x_hist) > 30 else ax.plot
    plot_type = ax.semilogy if m != "Newton" else ax.plot
    plot_type(p_hist, "b", label="||grad f||")
    ax.legend()

    ax = fig.add_subplot(224)
    plot_type = ax.semilogy if m != "Newtwon" else ax.plot
    plot_type(f_hist, "r", label="f(x)")
    ax.legend()

    plt.show()

def fvals(x_hist):
    for i in x_hist:
        print("f(x)", scipy.optimize.rosen(i))

def grads(x_hist):
    for i in x_hist:
        #print("g(x)\n", scipy.optimize.rosen_der(i))
        print("||g(x)||\n", np.linalg.norm(scipy.optimize.rosen_der(i)))

def hess(x_hist):
    for i in x_hist:
        print("H(x)\n", scipy.optimize.rosen_hess(i))

def hess_grad(x_hist):
    for i in x_hist:
        g = scipy.optimize.rosen_der(i)
        print("H(g(x))\n", scipy.optimize.rosen_hess_prod(i, g))

if __name__ == "__main__":
    #As = [1, 100]
    #methods = ["Newton", "SteepestDescent"]
    As = [100]
    #As = [1]
    methods = ["Newton"]

    for A, m in itertools.product(As, methods):
        #x_hist = np.loadtxt("x." + str(A) + ".000000." + m + ".txt")
        p_hist = np.loadtxt("p." + str(A) + ".000000." + m + ".txt")
        f_hist = np.loadtxt("f." + str(A) + ".000000." + m + ".txt")

        print(A, m, len(p_hist))

        #fvals(x_hist)
        #grads(x_hist)
        #hess(x_hist)
        #hess_grad(x_hist)
        #plot_figs(x_hist, p_hist, f_hist, A, m)
        plot_figs([], p_hist, f_hist, A, m)
