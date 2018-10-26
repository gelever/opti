/* 
 * Stephan Gelever
 * Math 510
 * HW 2
 *
 * Copyright (c) 2018, Stephan Gelever
 */

#include <iostream>
#include "linalgcpp.hpp"

using DenseMatrix = linalgcpp::DenseMatrix;
using Vector = linalgcpp::Vector<double>;
using VectorView = linalgcpp::VectorView<double>;
using Operator = linalgcpp::Operator;

int counter = 0;

double f_rosen(double A, const VectorView& x)
{
    assert(x.size() == 2);

    double term1 = x[1] - (x[0]*x[0]);
    double term2 = 1 - x[0];

    counter++;

    return (A * term1 * term1) + (term2 * term2);
}

Vector d_f_rosen(double A, const VectorView& x)
{
    assert(x.size() == 2);

    Vector df(2);

    df[0] = -4 * A * (x[1] - (x[0] * x[0])) * x[0] - 2 * (1 - x[0]);
    df[1] = 2 * A * (x[1] - (x[0] * x[0]));

    return df;
}

DenseMatrix d2_f_rosen(double A, const VectorView& x)
{
    DenseMatrix h(2, 2);

    h(0, 0) = 12 * A * x[0] * x[0] - 4 * A * x[1] + 2;
    h(0, 1) = -4 * A * x[0];
    h(1, 0) = h(0, 1);
    h(1, 1) = 2 * A;

    return h;
}

enum class Method : int { Newton, SteepestDescent};

Vector compute_P(Method method, double A, const VectorView& x, const VectorView& gradient)
{
    Vector p;

    switch (method)
    {
    case Method::Newton:
    {
        DenseMatrix h = d2_f_rosen(A, x);
        h.Invert();

        p = h.Mult(gradient);
        p *= -1.0;
        break;
    }
    case Method::SteepestDescent:
    {
        p = gradient;
        p *= -1.0;
        break;
    }
    default: { throw std::runtime_error("Invalid Method!"); }
    }

    return p;
}

double line_backtrack(double A, double alpha_0, double c, double rho, int alpha_max_iter,
                      const VectorView& x, double f, const VectorView& g,
                      const VectorView& p)
{
    double alpha = alpha_0;

    Vector x_alpha_p(x);
    x_alpha_p.Add(alpha, p);

    double c_grad_p = c * (g * p);

    for (int i = 0; i < alpha_max_iter; ++i)
    {
        if (f_rosen(A, x_alpha_p) < f + (alpha * c_grad_p))
        {
            break;
        }

        alpha *= rho;

        x_alpha_p = x;
        x_alpha_p.Add(alpha, p);
    }

    return alpha;
}

std::string method_to_string(Method method)
{
    switch (method)
    {
    case Method::Newton: return "Newton";
    case Method::SteepestDescent: return "SteepestDescent";
    default: { throw std::runtime_error("Invalid Method!"); }
    }
}

void save_history(const std::vector<Vector>& x, const std::string& prefix, double A, Method method)
{
    std::string filename = prefix + "." + std::to_string(A) + "." +
                           method_to_string(method) + ".txt";

    std::ofstream output(filename);

    for (auto&& x_i : x)
    {
        output << x_i[0] << " " << x_i[1] << "\n";
    }
}

void save_history(const std::vector<double>& x, const std::string& prefix, double A, Method method)
{
    std::string filename = prefix + "." + std::to_string(A) + "." +
                           method_to_string(method) + ".txt";
    std::ofstream output(filename);

    for (auto&& x_i : x)
    {
        output << x_i << "\n";
    }
}

int main()
{
    // Line Backtrack Params
    double alpha_0 = 1.0;
    double c = 0.01;
    double rho = 0.5;
    int alpha_max_iter = 20;

    // Iteration Params
    double tol = 1e-3;
    int max_iter = 8000;

    const auto minimize = [&](double A, Method method)
    {
        // Initial x
        Vector x(2);
        x[0] = -1.2;
        x[1] = 1.0;

        // History
        std::vector<Vector> x_history(1, x);
        std::vector<double> p_history;
        std::vector<double> f_history;

        int i = 1;
        for (; i < max_iter; ++i)
        {
            double f = f_rosen(A, x);
            Vector g = d_f_rosen(A, x);
            Vector p = compute_P(method, A, x, g);
            double p_norm = p.L2Norm();

            double alpha = line_backtrack(A, alpha_0, c, rho,
                                          alpha_max_iter,
                                          x, f, g, p);
            x.Add(alpha, p);

            x_history.push_back(x);
            p_history.push_back(p_norm);
            f_history.push_back(f);

            if (p_norm < tol)
            {
                break;
            }
        }

        printf("Iter Count: %d\tFunction count: %d\n", i, counter);
        save_history(x_history, "x", A, method);
        save_history(p_history, "p", A, method);
        save_history(f_history, "f", A, method);
    };

    // Test params
    std::vector<double> As {1.0, 100.0};
    std::vector<Method> methods {Method::Newton, Method::SteepestDescent};

    for (auto&& method : methods)
    {
        for (auto&& A : As)
        {
            counter = 0;

            minimize(A, method);
        }
    }

    return 0;
}
