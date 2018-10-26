/* 
 * Stephan Gelever
 * Math 510
 * HW 2
 *
 * Copyright (c) 2018, Stephan Gelever
 */

#include "rosenbrock.hpp"

using namespace rosenbrock;

void DogLeg(const VectorView& grad, const Operator& B, const Operator& B_inv, double delta, VectorView p)
{
    double gBg = B.InnerProduct(grad, grad); // Note this can be passed in as param
    double g_norm = grad.L2Norm();
    double tau = delta / g_norm;

    if (gBg <= 0.0)
    {
        p.Set(-tau, grad);
        return;
    }

    Vector p_c(grad);
    p_c *= -1.0 * std::min(tau, (g_norm * g_norm) / gBg);

    // Close enough to trust region boundary
    if (std::fabs(p_c.L2Norm() - delta) < 2.2204e-15)
    {
        p.Set(p_c);
        return;
    }

    try
    {
        Vector p_b = B_inv.Mult(grad);
        p_b *= -1.0;

        if (p_b.L2Norm() <= delta)
        {
            p.Set(p_b);
            return;
        }

        p_b -= p_c;

        double b = p_c.Mult(p_b);
        double a = p_b.Mult(p_b);
        double c = p_c.Mult(p_c) - (delta * delta);

        double tau_dog = (-b + std::sqrt((b*b) - (a*c))) / a;

        p.Set(p_c);
        p.Add(tau_dog, p_b);
    }
    catch(const std::runtime_error& e)
    {
        p.Set(p_c);
    }
}

int main(int argc, char ** argv)
{
    // Trust Region Params
    double delta = 0.5;
    double delta_max = 1.0;
    double delta_tol = 1e-12;
    double eta = 0.001;

    // Iteration Params
    double tol = 1e-3;
    int max_iter = 20000;
    bool save_history = false;
    bool verbose = false;

    // Problem Params
    double rb_A = 100.0;
    int dim = 100000;
    double variance = 0.05;

    linalgcpp::ArgParser arg_parser(argc, argv);

    arg_parser.Parse(tol, "--tol", "Solve tolerance.");
    arg_parser.Parse(max_iter, "--iter", "Max iterations");
    arg_parser.Parse(save_history, "--hist", "Save iteration history");
    arg_parser.Parse(verbose, "--verbose", "Show iteration information");

    arg_parser.Parse(rb_A, "--A", "A in Rosenbrock Function");
    arg_parser.Parse(dim, "--dim", "Dimensions");
    arg_parser.Parse(variance, "--var", "Inital vector variance");

    arg_parser.Parse(delta, "--delta", "Inital delta in trust region");
    arg_parser.Parse(delta_max, "--delta-max", "Maximum delta in trust region");
    arg_parser.Parse(delta_tol, "--delta-tol", "Tolerance of delta in trust region");
    arg_parser.Parse(eta, "--eta", "Accept tolerance");

    if (!arg_parser.IsGood())
    {
        arg_parser.ShowHelp();
        arg_parser.ShowErrors();

        return EXIT_FAILURE;
    }

    arg_parser.ShowOptions();

    // Initial point
    Vector x(dim);
    Vector x_propose(dim);

    // Random w/ uniform variance
    double lo = 1.0 - variance;
    double hi = 1.0 + variance;
    linalgcpp::Randomize(x, lo, hi);

    // Alternate -1.2, 1.0, -1.2
    alternate_x(x);

    // History
    std::vector<Vector> x_history(1, x);
    std::vector<Vector> grad_history;
    std::vector<double> p_history;
    std::vector<double> f_history;

    // CG Solver initialize
    Rosenbrock rb(rb_A, dim);
    Hessian rb_hess(rb, x);
    linalgcpp::PCGSolver cg(rb_hess);
    //cg.SetRelTol(1e-24);
    //cg.SetAbsTol(1e-24);

    // Workspace
    Vector grad(dim);
    Vector hess(dim);
    Vector h_inv(dim);
    Vector p(dim);
    Vector ones(dim, 1.0);
    Vector error(dim);

    double f_old = std::numeric_limits<double>::max();
    double f = std::numeric_limits<double>::max();

    int num_fallback = 0;
    int iter = 1;
    for (; iter < max_iter; ++iter)
    {
        // Compute f(x)
        f = rb.Eval(x);

        // Compute gradient at x
        Gradient rb_grad(rb, x);
        rb_grad.Mult(x, grad);


        Hessian rb_hess(rb, x);
        cg.SetOperator(rb_hess);

        DogLeg(grad, rb_hess, cg, delta, p);

        linalgcpp::Add(1.0, x, 1.0, p, 0.0, x_propose);

        double f_propose = rb.Eval(x_propose);
        double m_0 = -1.0 * grad.Mult(p);
        double m_p = -0.5 * rb_hess.InnerProduct(p, p);
        double rho = (f - f_propose) / (m_0 - m_p);

        double p_norm = p.L2Norm();
        double grad_norm = grad.L2Norm();

        if (rho < 0.25)
        {
            delta *= 0.25;
        }
        else if (rho > 0.75 && std::fabs(p_norm - delta) < delta_tol)
        {
            delta = std::min(2 * delta, delta_max);
        }

        if (rho > eta)
        {
            std::swap(x, x_propose);
            std::swap(f, f_old);
        }

        // Compute error
        linalgcpp::Sub(ones, x, error);

        double e_norm = error.L2Norm();

        if (save_history)
        {
            x_history.push_back(x);
            grad_history.push_back(grad);
            p_history.push_back(p_norm);
            f_history.push_back(f);
        }

        if (verbose)
        {
            printf("%d: f: %.2e g: %.8f e: %.2e grad*p: %.2e cg: %d\n",
                    iter, f_old, grad_norm, e_norm, grad * p, cg.GetNumIterations());
        }

        if (grad_norm < tol)
        {
            break;
        }
    }

    printf("f(x): %.2e Iter: %d\tTotal Function Evals: %d\n", f, iter, rb.num_evals);

    if (save_history)
    {
        write_history(x_history, "x", rb_A);
        write_history(grad_history, "g", rb_A);
        write_history(p_history, "p", rb_A);
        write_history(f_history, "f", rb_A);
    }

    return 0;
}
