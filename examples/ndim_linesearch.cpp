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

class Rosenbrock
{
    public:
        Rosenbrock(double A_in, int num_dim_in = 2) : A(A_in), num_dim(num_dim_in), num_evals(0)
        { assert(num_dim >= 2); assert(A >= 1.0);}

        double Eval(const VectorView& x) const;
        double LineBackTrack(const VectorView& x, double f, const VectorView& grad, const VectorView& p,
                double alpha_0, double c, double rho, int max_iter) const;

        double A;
        int num_dim;

        mutable int num_evals;
};

class Hessian : public Operator
{
    public:
        Hessian(const Rosenbrock& rb_in, const VectorView& x_in)
            : Operator(rb_in.num_dim), rb(rb_in), x(x_in)
        {}

        using Operator::Mult;
        void Mult(const VectorView& input, VectorView output) const override
        {
            int dim = rb.num_dim;
            double A = rb.A;

            output[0] = ((12 * A * x[0] * x[0]) - (4 * A * x[1]) + 2) * input[0] - (4 * A * x[0] * input[1]);

            for (int i = 1; i < dim - 1; ++i)
            {
                output[i] = (-4.0 * A * x[i - 1] * input[i - 1]) + ((2 * A) + 2 + (12 * A * x[i] * x[i]) - (4.0 * A * x[i + 1])) * input[i] - (4.0 * A * x[i] * input[i + 1]);
            }

            output[dim - 1] = (-4.0 * A * x[dim - 2] * input[dim - 2]) + (2.0 * A * input[dim - 1]);
        }

        const Rosenbrock& rb;
        const VectorView& x;
};

class Gradient : public Operator
{
    public:
        Gradient(const Rosenbrock& rb_in, const VectorView& x_in)
            : Operator(rb_in.num_dim), rb(rb_in), x(x_in)
        {}

        using Operator::Mult;
        void Mult(const VectorView& x, VectorView df_x) const override
        {
            double A = rb.A;
            int num_dim = rb.num_dim;

            assert(x.size() == num_dim);
            assert(df_x.size() == num_dim);

            df_x[0] = (-4.0 * A * x[0] * (x[1] - std::pow(x[0], 2))) - (2.0 * (1.0 - x[0]));

            for (int i = 1; i < num_dim - 1; ++i)
            {
                df_x[i] = (2.0 * A * (x[i] - std::pow(x[i - 1], 2))) \
                          - (4 * A * x[i] * (x[i + 1] - std::pow(x[i], 2))) \
                          - (2 * (1 - x[i]));
            }

            df_x[num_dim - 1] = 2.0 * A * (x[num_dim - 1] - std::pow(x[num_dim - 2], 2));
        }

        const Rosenbrock& rb;
        const VectorView& x;
};

class HessianDiagInv : public Operator
{
    public:
        HessianDiagInv(const Rosenbrock& rb_in, const VectorView& x_in)
            : Operator(rb_in.num_dim), rb(rb_in), x(x_in)
        {}

        using Operator::Mult;
        void Mult(const VectorView& input, VectorView output) const override
        {
            int dim = rb.num_dim;
            double A = rb.A;

            output[0] = input[0] / ((12 * A * x[0] * x[0]) - (4 * A * x[1]) + 2);

            for (int i = 1; i < dim - 1; ++i)
            {
                output[i] = input[i] /  ((2 * A) + 2 + (12 * A * x[i] * x[i]) - (4.0 * A * x[i + 1]));
            }

            output[dim - 1] = input[dim - 1] /  (2.0 * A);
        }

        const Rosenbrock& rb;
        const VectorView& x;
};

double Rosenbrock::Eval(const VectorView& x) const
{
    double sum = 0.0;

    for (int i = 0; i < num_dim - 1; ++i)
    {
        double p1 = (x[i + 1] - std::pow(x[i], 2));
        double p2 = (1 - x[i]);

        sum += (A * std::pow(p1, 2)) + std::pow(p2, 2);
    }

    num_evals++;

    return sum;
}

double Rosenbrock::LineBackTrack(const VectorView& x, double f, const VectorView& grad, const VectorView& p,
        double alpha_0, double c, double rho, int max_iter) const
{
    double alpha = alpha_0;
    double c_grad_p = c * (grad * p);

    Vector x_alpha_p(x.size());
    linalgcpp::Add(1.0, x, alpha, p, 0.0, x_alpha_p);

    int iter = 1;
    for (; iter < max_iter; ++iter)
    {
        // Check Alpha
        double f_i = Eval(x_alpha_p);

        if (f_i < f + (alpha * c_grad_p))
        {
            break;
        }

        // Update x + alpha * p
        alpha *= rho;
        linalgcpp::Add(1.0, x, alpha, p, 0.0, x_alpha_p);
    }

    if (iter == max_iter)
    {
        throw std::runtime_error("Maximum number of alpha iterations!");
    }

    return alpha;
}

void write_history(const std::vector<Vector>& x, const std::string& prefix, double A)
{
    std::string filename = prefix + "." + std::to_string(A) + "." +
                           "Newton.txt";
    std::ofstream output(filename);
    output.precision(5);
    output << std::scientific;

    for (auto&& x_i : x)
    {
        for (auto&& i : x_i)
        {
            output << i << " ";
        }

        output << "\n";
    }
}

void write_history(const std::vector<double>& x, const std::string& prefix, double A)
{
    std::string filename = prefix + "." + std::to_string(A) + "." +
                           "Newton.txt";
    std::ofstream output(filename);

    for (auto&& x_i : x)
    {
        output << x_i << "\n";
    }
}

int main(int argc, char ** argv)
{
    // Line Backtrack Params
    double alpha_0 = 1.0;
    double c = 0.01;
    double rho = 0.5;
    int alpha_max_iter = 2000;

    // Iteration Params
    double tol = 1e-3;
    int max_iter = 20000;
    bool save_history = false;

    // Problem Params
    double rb_A = 100.0;
    int dim = 100000;
    double variance = 0.05;

    // Linesearch params
    double alpha_0_linesearch = 1.0;
    double c_linesearch = 0.01;
    double rho_linesearch = 0.50;
    int max_iter_linesearch = 20;

    linalgcpp::ArgParser arg_parser(argc, argv);

    arg_parser.Parse(tol, "--tol", "Solve tolerance.");
    arg_parser.Parse(max_iter, "--iter", "Max iterations");
    arg_parser.Parse(save_history, "--hist", "Save iteration history");

    arg_parser.Parse(rb_A, "--A", "A in Rosenbrock Function");
    arg_parser.Parse(dim, "--dim", "Dimensions");
    arg_parser.Parse(variance, "--var", "Inital vector variance");

    arg_parser.Parse(alpha_0_linesearch, "--alpha", "Inital alpha in linesearch");
    arg_parser.Parse(c_linesearch, "--c", "C factor in linesearch");
    arg_parser.Parse(rho_linesearch, "--rho", "Reduction factor in linesearch");
    arg_parser.Parse(max_iter_linesearch, "--alpha-max-iter", "Maximum iterations in linesearch");

    if (!arg_parser.IsGood())
    {
        arg_parser.ShowHelp();
        arg_parser.ShowErrors();

        return EXIT_FAILURE;
    }

    arg_parser.ShowOptions();

    // Initial point
    Vector x(dim);
    double lo = 1.0 - variance;
    double hi = 1.0 + variance;
    linalgcpp::Randomize(x, lo, hi);

    // Alternate -1.2, 1.0, -1.2

    //x[0] = -1.2;
    //x[1] = 1.0;

    // History
    std::vector<Vector> x_history(1, x);
    std::vector<double> p_history;
    std::vector<double> f_history;

    // CG Solver initialize
    Rosenbrock rb(rb_A, dim);
    Hessian rb_hess(rb, x);
    linalgcpp::PCGSolver cg(rb_hess);

    // Workspace
    Vector grad(dim);
    Vector hess(dim);
    Vector h_inv(dim);
    Vector p(dim);
    Vector ones(dim, 1.0);
    Vector error(dim);

    for (int i = 1; i < max_iter; ++i)
    {
        // Compute f(x)
        double f = rb.Eval(x);

        // Compute gradient at x
        Gradient rb_grad(rb, x);
        rb_grad.Mult(x, grad);

        // Solve p = H \ grad
        Hessian rb_hess(rb, x);
        cg.SetOperator(rb_hess);
        p = 0.0;
        cg.Mult(grad, p);

        //HessianDiagInv rb_hess(rb, x);
        //rb_hess.Mult(grad, p);

        //p = grad;

        p *= -1.0;

        // Find alpha using line backtracking
        double alpha = rb.LineBackTrack(x, f, grad, p,
                alpha_0_linesearch,
                c_linesearch,
                rho_linesearch,
                max_iter_linesearch);
        x.Add(alpha, p);

        // Compute error
        linalgcpp::Sub(ones, x, error);

        double e_norm = error.L2Norm();
        double p_norm = p.L2Norm();

        if (save_history)
        {
            x_history.push_back(x);
            p_history.push_back(p_norm);
            f_history.push_back(f);
        }

        printf("%d: f: %.2e p: %.2e e: %.2e alpha: %.2e grad*p: %.2e cg: %d\n",
                i, f, p_norm, e_norm, alpha, grad * p, cg.GetNumIterations());

        if (p_norm < tol)
        {
            break;
        }
    }

    printf("Total Function Evals: %d\n", rb.num_evals);

    if (save_history)
    {
        write_history(x_history, "x", rb_A);
        write_history(p_history, "p", rb_A);
        write_history(f_history, "f", rb_A);
    }

    return 0;
}
