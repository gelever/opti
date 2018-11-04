/* 
 * Stephan Gelever
 * Math 510
 * HW 2
 *
 * Copyright (c) 2018, Stephan Gelever
 */

#include "rosenbrock.hpp"

using namespace rosenbrock;

struct LineSearchParams
{
    double alpha_0;
    double c;
    double rho;
    int max_iter;
};

void LineBackTrack(const Rosenbrock& rb, VectorView x,
        double& f, const VectorView& grad, const VectorView& p,
        const LineSearchParams& ls_params)
{
    MPI_Comm comm = rb.comm;

    double alpha = ls_params.alpha_0;
    double f_0 = f;
    double c_grad_p = ls_params.c * (ParMult(comm,grad, p));

    Vector x_0(x);
    linalgcpp::Add(1.0, x_0, alpha, p, 0.0, x);

    int iter = 1;
    for (; iter < ls_params.max_iter; ++iter)
    {
        // Check Alpha
        f = rb.Eval(x);

        if (f < f_0 + (alpha * c_grad_p))
        {
            break;
        }

        // Update x + alpha * p
        alpha *= ls_params.rho;
        linalgcpp::Add(1.0, x_0, alpha, p, 0.0, x);
    }

    if (iter == ls_params.max_iter)
    {
        throw std::runtime_error("Maximum number of alpha iterations!");
    }
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
    // Initialize MPI
    MpiSession mpi_info(argc, argv);
    MPI_Comm comm = mpi_info.comm;
    int myid = mpi_info.myid;
    int num_procs = mpi_info.num_procs;

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
    LineSearchParams ls_params;
    ls_params.alpha_0 = 1.0;
    ls_params.c = 0.01;
    ls_params.rho = 0.50;
    ls_params.max_iter = 20;

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

    arg_parser.Parse(ls_params.alpha_0, "--alpha", "Inital alpha in linesearch");
    arg_parser.Parse(ls_params.c, "--c", "C factor in linesearch");
    arg_parser.Parse(ls_params.rho, "--rho", "Reduction factor in linesearch");
    arg_parser.Parse(ls_params.max_iter, "--alpha-max-iter", "Maximum iterations in linesearch");

    if (!arg_parser.IsGood())
    {
        ParPrint(myid, arg_parser.ShowHelp());
        ParPrint(myid, arg_parser.ShowErrors());

        return EXIT_FAILURE;
    }

    ParPrint(myid, arg_parser.ShowOptions());

    // Problem initialize
    Rosenbrock rb(comm, rb_A, dim);
    Vector x = set_x(rb, initial_x, variance);
    Gradient rb_grad(rb);
    Hessian rb_hess(rb, x);

    // Hessian Solver initialize
    int cg_max_iter = 1000;
    double cg_rel_tol = 1e-6;
    double cg_abs_tol = 1e-8;
    int cg_verbose = false;
    linalgcpp::PCGSolver cg(rb_hess, cg_max_iter, cg_rel_tol, cg_abs_tol, cg_verbose,
                            linalgcpp::ParMult);

    double f = rb.Eval(x);

    // Workspace
    Vector grad(rb.local_dim);
    Vector p(rb.local_dim);
    Vector ones(rb.local_dim, 1.0);
    Vector error(rb.local_dim);

    // History
    std::vector<Vector> x_history;
    std::vector<double> p_history;
    std::vector<double> f_history;

    int iter = 1;
    for (; iter < max_iter; ++iter)
    {
        // Compute gradient at x
        rb_grad.Mult(x, grad);

        // Update p, update x using line backtracking
        update_p(method, rb, x, cg, grad, p);
        LineBackTrack(rb, x, f, grad, p, ls_params);

        // Compute error
        linalgcpp::Sub(ones, x, error);

        double e_norm = ParL2Norm(comm, error);
        double p_norm = ParL2Norm(comm, p);

        if (save_history)
        {
            x_history.push_back(x);
            p_history.push_back(p_norm);
            f_history.push_back(f);
        }

        if (verbose)
        {
            ParPrint(myid, printf("%d: f: %.2e p: %.2e e: %.2e grad*p: %.2e cg: %d\n",
                    iter, f, p_norm, e_norm, grad * p, cg.GetNumIterations()));
        }

        if (p_norm < tol)
        {
            break;
        }
    }

    ParPrint(myid, printf("\n%s Stats:\n------------------------\n", method.c_str()));
    ParPrint(myid, printf("f(x):\t%.2e\nIter:\t%d\n", f, iter));
    ParPrint(myid, printf("Function Evals:\t%d\nGrad Evals:\t%d\nHessian Apply:\t%d\n",
             rb.num_evals, rb_grad.num_evals, rb_hess.num_evals));

    if (save_history)
    {
        //Dont work in parallel yet
        //write_history(x_history, "x", rb_A);
        //write_history(p_history, "p", rb_A);
        //write_history(f_history, "f", rb_A);
    }

    return 0;
}
