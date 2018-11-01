/* 
 * Stephan Gelever
 * Math 510
 * HW 3
 *
 * Copyright (c) 2018, Stephan Gelever
 */

#include "rosenbrock.hpp"

using namespace rosenbrock;

void DogLeg(const VectorView& grad, const Operator& B, const Operator& B_inv, double delta, VectorView p)
{
    double gBg = B.InnerProduct(grad, grad);
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

    // Try inverting B
    try
    {
        Vector p_b = B_inv.Mult(grad);
        p_b *= -1.0;

        // P_b is in the interior
        if (p_b.L2Norm() <= delta)
        {
            p.Set(p_b);
            return;
        }

        // Otherwise use dogleg
        p_b -= p_c;

        double b = p_c.Mult(p_b);
        double a = p_b.Mult(p_b);
        double c = p_c.Mult(p_c) - (delta * delta);

        double tau_dog = (-b + std::sqrt((b*b) - (a*c))) / a;

        p.Set(p_c);
        p.Add(tau_dog, p_b);
    }
    // Otherwise use Cauchy point
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
    int dim = 2;
    double variance = 0.05;
    std::string initial_x = "Standard";

    linalgcpp::ArgParser arg_parser(argc, argv);

    arg_parser.Parse(tol, "--tol", "Solve tolerance.");
    arg_parser.Parse(max_iter, "--iter", "Max iterations");
    arg_parser.Parse(save_history, "--hist", "Save iteration history");
    arg_parser.Parse(verbose, "--verbose", "Show iteration information");

    arg_parser.Parse(rb_A, "--A", "A in Rosenbrock Function");
    arg_parser.Parse(dim, "--dim", "Dimensions");
    arg_parser.Parse(initial_x, "--initial-x", "Set initial x [Standard, Random]");
    arg_parser.Parse(variance, "--var", "Inital vector uniform random variance about solution");

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
    Vector x = set_x(dim, initial_x, variance);
    Vector x_propose(dim);

    // Workspace
    Vector grad(dim);
    Vector p(dim);
    Vector ones(dim, 1.0);
    Vector error = ones - x;

    // Problem initialize
    Rosenbrock rb(rb_A, dim);
    Gradient rb_grad(rb, x);
    Hessian rb_hess(rb, x);
    linalgcpp::PCGSolver cg(rb_hess);

    rb_grad.Mult(x, grad);

    double f = rb.Eval(x);
    double g_norm = grad.L2Norm();
    double e_norm = error.L2Norm();

    // History
    std::vector<Vector> x_history(1, x);
    std::vector<double> g_history(1, g_norm);
    std::vector<double> f_history(1, f);

    int iter = 1;
    for (; iter < max_iter; ++iter)
    {
        // Compute dogleg
        DogLeg(grad, rb_hess, cg, delta, p);

        // Check if p is acceptable
        linalgcpp::Add(1.0, x, 1.0, p, 0.0, x_propose);

        double f_propose = rb.Eval(x_propose);
        double m_0 = -1.0 * grad.Mult(p);
        double m_p = -0.5 * rb_hess.InnerProduct(p, p);
        double rho = (f - f_propose) / (m_0 - m_p);

        double p_norm = p.L2Norm();

        // Adjust delta
        if (rho < 0.25)
        {
            delta *= 0.25;
        }
        else if (rho > 0.75 && std::fabs(p_norm - delta) < delta_tol)
        {
            delta = std::min(2 * delta, delta_max);
        }

        // Accept P condition
        if (rho > eta)
        {
            std::swap(x, x_propose);
            std::swap(f, f_propose);

            rb_grad.Mult(x, grad);
            g_norm = grad.L2Norm();
            e_norm = error.L2Norm();

            linalgcpp::Sub(ones, x, error);
        }

        if (save_history)
        {
            x_history.push_back(x);
            g_history.push_back(g_norm);
            f_history.push_back(f);
        }

        if (verbose)
        {
            printf("%d: f: %.2e g: %.8f p: %.8f e: %.2e grad*p: %.2e cg: %d delta: %.3f rho: %.3f\n",
                    iter, f, g_norm, p_norm, e_norm, grad * p, cg.GetNumIterations(), delta, rho);
        }

        if (g_norm < tol)
        {
            break;
        }
    }

    printf("\nFinal Stats:\n------------------------\n");
    printf("f(x):\t%.2e\nIter:\t%d\n", f, iter);
    printf("Function Evals:\t%d\nGrad Evals:\t%d\nHessian Apply:\t%d\n",
            rb.num_evals, rb_grad.num_evals, rb_hess.num_evals);

    if (save_history)
    {
        write_history(x_history, "x", rb_A);
        write_history(g_history, "g", rb_A);
        write_history(f_history, "f", rb_A);
    }

    return 0;
}
