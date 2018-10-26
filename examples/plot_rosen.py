# Stephan Gelever
# Math 510 HW
# 
# Copyright (c) 2018, Stephan Gelever

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

import scipy.optimize
import numpy as np
import itertools
import sys

def plot_figs(x_hist, p_hist, f_hist, A, method):
    fig = plt.figure()
    if (len(x_hist) > 0):
        ax = fig.add_subplot(121)

        if A > 1.0:
            x_space = np.linspace(-1.5, 1.5, 1000)
            y_space = np.linspace(-0.1, 1.5, 1000)
            levels = np.logspace(0, 8, num=10, base=2.0)
        else:   
            x_space = np.linspace(-2.0, 2.0, 1000)
            y_space = np.linspace(-2.0, 2.0, 1000)
            levels = np.linspace(0, 8, num=10)

        x, y = np.meshgrid(x_space, y_space)
        z = (A * (y - (x*x))**2 + (1 - x)**2)

        ax.contour(x, y, z, cmap=plt.get_cmap("viridis_r"), zorder=0, levels=levels)

        c = np.linspace(0, 1.0, len(x_hist))

        x_hist = np.vstack(([-1.2, 1.0], x_hist))

        initial = ax.plot(-1.2, 1.0, 'r^')
        initial = ax.plot(1.0, 1.0, 'gv')

        scatter = ax.plot(x_hist[:, 0], x_hist[:, 1], 'bo--', markersize=5)

    ax = fig.add_subplot(222)
    ax.semilogy(p_hist, "b", label="||grad f||")
    ax.legend()

    ax = fig.add_subplot(224)
    ax.semilogy(f_hist, "r", label="f(x)")
    ax.legend()

    plt.suptitle(method + " Method, A: " + str(float(A)))

    #plt.tight_layout()
    plt.savefig(method+"."+str(A)+".png", dpi=300)
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
    method = sys.argv[1]
    As = [float(i) for i in sys.argv[2:]]
    print(method, As)
    print(As)

    for A in As:
        x_hist = np.loadtxt("x." + str(A) + "00000.history.txt")
        p_hist = np.loadtxt("p." + str(A) + "00000.history.txt")
        #g_hist = np.loadtxt("g." + str(A) + "000000.history.txt")
        f_hist = np.loadtxt("f." + str(A) + "00000.history.txt")

        print(A, method, len(p_hist))

        #fvals(x_hist)
        #grads(x_hist)
        #hess(x_hist)
        #hess_grad(x_hist)
        #plot_figs(x_hist, p_hist, f_hist, A, m)
        plot_figs(x_hist, p_hist, f_hist, A, method)
        #plot_figs(x_hist, g_hist, f_hist, A, m)
        #plot_figs([], p_hist, f_hist, A, m)
