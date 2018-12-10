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
    double c_grad_p = ls_params.c * (ParMult(comm, grad, p));

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

double ComputeBeta(MPI_Comm comm, const std::string& method, VectorView grad,
                   const VectorView& grad_next, const VectorView& p)
{
    double beta = 0.0;

    double grad_grad = ParMult(comm, grad_next, grad_next);
    double grad_p = ParMult(comm, grad_next, p);

    if (method == "FR")
    {
        double beta_0 = grad_grad;
        double beta_1 = ParMult(comm, grad, grad);

        beta = beta_0 / beta_1;
    }
    else if (method == "PR")
    {
        double beta_1 = ParMult(comm, grad, grad);

        grad *= -1.0;
        grad += grad_next;

        double beta_0 = ParMult(comm, grad_next, grad);

        beta = beta_0 / beta_1;
    }
    else if (method == "HS")
    {
        grad *= -1.0;
        grad += grad_next;

        double beta_0 = ParMult(comm, grad_next, grad);
        double beta_1 = ParMult(comm, p, grad);

        beta = beta_0 / beta_1;
    }
    else if (method == "DY")
    {
        grad *= -1.0;
        grad += grad_next;

        double beta_0 = grad_grad;
        double beta_1 = ParMult(comm, p, grad);

        beta = beta_0 / beta_1;
    }
    else
    {
        throw std::runtime_error("Invalid Method Selected!");
    }

    return beta;
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
    int restart = 0;
    bool save_history = false;
    bool verbose = false;

    // Problem Params
    double rb_A = 100.0;
    int dim = 2;
    double variance = 0.05;
    std::string method = "FR";
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
    arg_parser.Parse(restart, "--restart", "Force restart every n iterations");
    arg_parser.Parse(save_history, "--hist", "Save iteration history");
    arg_parser.Parse(verbose, "--verbose", "Show iteration information");

    arg_parser.Parse(rb_A, "--A", "A in Rosenbrock Function");
    arg_parser.Parse(dim, "--dim", "Dimensions");
    arg_parser.Parse(method, "--method", "Method to use: [FR, PR, HS, DY]");
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

    double f = rb.Eval(x);

    // Workspace
    Vector grad(rb.local_dim);
    Vector p(rb.local_dim);
    Vector ones(rb.local_dim, 1.0);
    Vector error(rb.local_dim);

    Vector p_next(rb.local_dim);
    Vector grad_next(rb.local_dim);

    // Initial graident and search direction
    rb_grad.Mult(x, grad);
    p.Set(-1.0, grad);

    // History
    std::vector<Vector> x_history;
    std::vector<double> g_history;
    std::vector<double> f_history;

    int num_restarts = 0;
    int last_restart = 0;

    Timer timer(Timer::Start::True);

    int iter = 1;
    for (; iter < max_iter; ++iter)
    {
        LineBackTrack(rb, x, f, grad, p, ls_params);

        rb_grad.Mult(x, grad_next);

        double beta = ComputeBeta(comm, method, grad, grad_next, p);

        linalgcpp::Add(-1.0, grad_next, beta, p, 0.0, p_next);

        double g_p_next = ParMult(comm, grad_next, p_next);

        if (g_p_next >= -1e-12 || (restart > 0 && (iter - last_restart > restart)))
        {
            num_restarts++;
            last_restart = iter;

            p.Set(-1.0, grad_next);
        }
        else
        {
            swap(p, p_next);
        }

        swap(grad, grad_next);

        // Compute error
        linalgcpp::Sub(ones, x, error);

        double e_norm = ParL2Norm(comm, error);
        double g_norm = ParL2Norm(comm, grad);

        if (save_history)
        {
            x_history.push_back(x);
            g_history.push_back(g_norm);
            f_history.push_back(f);
        }

        if (verbose)
        {
            ParPrint(myid, printf("%d: f: %.2e g: %.2e e: %.2e beta: %.2e\n",
                    iter, f, g_norm, e_norm, beta));
        }

        if (g_norm < tol)
        {
            break;
        }
    }

    timer.Click();

    ParPrint(myid, printf("\nNonlinear - %s Method Stats:\n------------------------\n", method.c_str()));
    ParPrint(myid, printf("f(x):\t%.2e\nIter:\t%d\n", f, iter));
    ParPrint(myid, printf("Function Evals:\t%d\nGrad Evals:\t%d\nRestarts:\t%d\n",
             rb.num_evals, rb_grad.num_evals, num_restarts));
    ParPrint(myid, printf("Time (s):\t%.8f\n", timer.TotalTime()));

    if (save_history)
    {
        linalgcpp_verify(num_procs == 1, "Saving not implemented in parallel yet!");

        write_history(x_history, "x", rb_A);
        write_history(g_history, "g", rb_A);
        write_history(f_history, "f", rb_A);
    }

    return 0;
}
