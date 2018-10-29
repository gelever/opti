/* 
 * Stephan Gelever
 * Math 510
 * HW 2
 *
 * Copyright (c) 2018, Stephan Gelever
 */

#include "rosenbrock.hpp"

using namespace rosenbrock;

double LineBackTrack(const Rosenbrock& rb, const VectorView& x,
        double f, const VectorView& grad, const VectorView& p,
        double alpha_0, double c, double rho, int max_iter)
{
    double alpha = alpha_0;
    double c_grad_p = c * (grad * p);

    Vector x_alpha_p(x);
    x_alpha_p.Add(alpha, p);

    int iter = 1;
    for (; iter < max_iter; ++iter)
    {
        // Check Alpha
        double f_i = rb.Eval(x_alpha_p);

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

void update_p(const std::string& method, const Rosenbrock& rb,
              const VectorView& x, const Operator& hess_inv,
              const VectorView& grad, VectorView p)
{
    if (method == "Dynamic")
    {
        // Try solving, but fall back to steepest descent on failure
        try {
            p = 0.0;
            hess_inv.Mult(grad, p);
        }
        catch(const std::runtime_error& e)
        {
            p.Set(grad);
        }
    }
    else if (method == "Newton")
    {
        p = 0.0;
        hess_inv.Mult(grad, p);
    }
    else if (method == "SteepestDescent")
    {
            p.Set(grad);
    }
    else
    {
        throw std::runtime_error("Invalid Method Selected!");
    }

    p *= -1.0;
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
    bool verbose = false;

    // Problem Params
    double rb_A = 100.0;
    int dim = 2;
    double variance = 0.05;
    std::string method = "Dynamic";
    std::string initial_x = "Standard";

    // Linesearch params
    double alpha_0_linesearch = 1.0;
    double c_linesearch = 0.01;
    double rho_linesearch = 0.50;
    int max_iter_linesearch = 20;

    linalgcpp::ArgParser arg_parser(argc, argv);

    arg_parser.Parse(tol, "--tol", "Solve tolerance.");
    arg_parser.Parse(max_iter, "--iter", "Max iterations");
    arg_parser.Parse(save_history, "--hist", "Save iteration history");
    arg_parser.Parse(verbose, "--verbose", "Show iteration information");

    arg_parser.Parse(rb_A, "--A", "A in Rosenbrock Function");
    arg_parser.Parse(dim, "--dim", "Dimensions");
    arg_parser.Parse(method, "--method", "Method to use: [Newton, SteepestDescent, Dynamic]");
    arg_parser.Parse(initial_x, "--initial-x", "Set initial x [Standard, Random]");
    arg_parser.Parse(variance, "--var", "Inital vector uniform random variance about solution");

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
    Vector x = set_x(dim, initial_x, variance);

    // History
    std::vector<Vector> x_history(1, x);
    std::vector<double> p_history;
    std::vector<double> f_history;

    // CG Solver initialize
    Rosenbrock rb(rb_A, dim);
    Hessian rb_hess(rb, x);
    linalgcpp::PCGSolver cg(rb_hess);
    cg.SetRelTol(1e-24);
    cg.SetAbsTol(1e-24);

    // Workspace
    Vector grad(dim);
    Vector p(dim);
    Vector ones(dim, 1.0);
    Vector error(dim);
    double f = std::numeric_limits<double>::max();

    int iter = 1;
    for (; iter < max_iter; ++iter)
    {
        // Compute f(x)
        f = rb.Eval(x);

        // Compute gradient at x
        Gradient rb_grad(rb, x);
        rb_grad.Mult(x, grad);

        // Update hessian
        Hessian rb_hess(rb, x);
        cg.SetOperator(rb_hess);

        // Update p
        update_p(method, rb, x, cg, grad, p);

        // Find alpha using line backtracking
        double alpha = LineBackTrack(rb, x, f, grad, p,
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

        if (verbose)
        {
            printf("%d: f: %.2e p: %.2e e: %.2e alpha: %.2e grad*p: %.2e cg: %d\n",
                    iter, f, p_norm, e_norm, alpha, grad * p, cg.GetNumIterations());
        }

        if (p_norm < tol)
        {
            break;
        }
    }

    printf("f(x): %.2e Iter: %d\tTotal Function Evals: %d\n", f, iter, rb.num_evals);

    if (save_history)
    {
        write_history(x_history, "x", rb_A);
        write_history(p_history, "p", rb_A);
        write_history(f_history, "f", rb_A);
    }

    return 0;
}
