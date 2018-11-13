/* 
 * Stephan Gelever
 * Math 510
 * HW 3
 *
 * Copyright (c) 2018, Stephan Gelever
 */

#include "rosenbrock.hpp"

using namespace rosenbrock;

bool CauchyPoint(MPI_Comm comm, const VectorView& grad, const Operator& B, double delta, VectorView p)
{
    B.Mult(grad, p);
    double gBg = ParMult(comm, grad, p);
    double g_norm = ParL2Norm(comm, grad);
    double tau = delta / g_norm;

    bool negative_gBg = (gBg <= 0.0);

    p.Set(negative_gBg ? -tau : -std::min(tau, (g_norm * g_norm) / gBg), grad);

    return negative_gBg;
}


void DogLeg(MPI_Comm comm, const VectorView& grad, const Operator& B, const Operator& B_inv,
            double delta, VectorView p, bool verbose = false)
{
    bool negative_gBg = CauchyPoint(comm, grad, B, delta, p);

    int myid;
    MPI_Comm_rank(comm, &myid);

    // Try Dogleg only if gBg is positive or P_c is far enough inside trust region 
    if (!negative_gBg && std::fabs(ParL2Norm(comm, p) - delta) > 2.2204e-15)
    {
        // Try inverting B
        try
        {
            Vector p_b(grad.size(), 0.0);
            B_inv.Mult(grad, p_b);
            p_b *= -1.0;

            // P_b is in the interior
            double p_b_norm = ParL2Norm(comm, p_b);

            if (p_b_norm <= delta)
            {
                p.Set(p_b);

                ParPrint(myid, if (verbose) std::cout << "Using interior Newton point\n");
            }
            // Otherwise use dogleg
            else
            {
                p_b -= p;

                double b = ParMult(comm, p, p_b);
                double a = ParMult(comm, p_b, p_b);
                double c = ParMult(comm, p, p) - (delta * delta);

                double tau_dog = (-b + std::sqrt((b*b) - (a*c))) / a;

                p.Add(tau_dog, p_b);

                if (verbose) ParPrint(myid, std::cout << "Using DogLeg\n");
            }

        }
        catch(const std::runtime_error& e)
        {
            if (verbose) ParPrint(myid, std::cout << "B not SPD, keeping Cauchy Point\n");
        }
    }
    else
    {
        if (verbose)
        {
            ParPrint(myid, std::cout << "Using Cauchy Point:\t");

            if (negative_gBg)
            {
                ParPrint(myid, std::cout << "Negative gBg\n");
            }
            else
            {
                ParPrint(myid, std::cout << "Close Cauchy\n");
            }
        }
    }
}


void ComputeP(MPI_Comm comm, const std::string& method, const VectorView& grad, const Operator& B,
              const Operator& B_inv, double delta, VectorView p, bool verbose = false)
{
    if (method == "CauchyPoint")
    {
        CauchyPoint(comm, grad, B, delta, p);
    }
    else if (method == "Dogleg")
    {
        DogLeg(comm, grad, B, B_inv, delta, p, verbose);
    }
    else
    {
        throw std::runtime_error("Invalid Method Selected!");
    }
}

int main(int argc, char ** argv)
{
    // Initialize MPI
    MpiSession mpi_info(argc, argv);
    MPI_Comm comm = mpi_info.comm;
    int myid = mpi_info.myid;
    int num_procs = mpi_info.num_procs;

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
    std::string method = "CauchyPoint";
    std::string initial_x = "Standard";

    linalgcpp::ArgParser arg_parser(argc, argv);

    arg_parser.Parse(tol, "--tol", "Solve tolerance.");
    arg_parser.Parse(max_iter, "--iter", "Max iterations");
    arg_parser.Parse(save_history, "--hist", "Save iteration history");
    arg_parser.Parse(verbose, "--verbose", "Show iteration information");

    arg_parser.Parse(rb_A, "--A", "A in Rosenbrock Function");
    arg_parser.Parse(dim, "--dim", "Dimensions");
    arg_parser.Parse(method, "--method", "Method to use: [CauchyPoint, Dogleg]");
    arg_parser.Parse(initial_x, "--initial-x", "Set initial x [Standard, Random]");
    arg_parser.Parse(variance, "--var", "Inital vector uniform random variance about solution");

    arg_parser.Parse(delta, "--delta", "Inital delta in trust region");
    arg_parser.Parse(delta_max, "--delta-max", "Maximum delta in trust region");
    arg_parser.Parse(delta_tol, "--delta-tol", "Tolerance of delta in trust region");
    arg_parser.Parse(eta, "--eta", "Accept tolerance");

    if (!arg_parser.IsGood())
    {
        ParPrint(myid, arg_parser.ShowHelp());
        ParPrint(myid, arg_parser.ShowErrors());

        return EXIT_FAILURE;
    }

    ParPrint(myid, arg_parser.ShowOptions());

    ParPrint(myid, std::cout << "Processors: " << num_procs << "\n");

    // Problem initialize
    Rosenbrock rb(comm, rb_A, dim);
    Vector x = set_x(rb, initial_x, variance);
    Vector x_propose(rb.local_dim);
    Gradient rb_grad(rb);
    Hessian rb_hess(rb, x);

    // Hessian Solver initialize
    int cg_max_iter = 10000;
    double cg_rel_tol = 1e-6;
    double cg_abs_tol = 1e-8;
    int cg_verbose = false;
    linalgcpp::PCGSolver cg(rb_hess, cg_max_iter, cg_rel_tol, cg_abs_tol, cg_verbose,
                            linalgcpp::ParMult);

    // Workspace
    Vector grad(rb.local_dim);
    Vector p(rb.local_dim);
    Vector Bp(rb.local_dim);
    Vector ones(rb.local_dim, 1.0);
    Vector error = ones - x;

    rb_grad.Mult(x, grad);

    double f = rb.Eval(x);
    double g_norm = ParL2Norm(comm, grad);
    double e_norm = ParL2Norm(comm, error);

    // History
    std::vector<Vector> x_history(1, x);
    std::vector<double> g_history(1, g_norm);
    std::vector<double> f_history(1, f);

    Timer timer(Timer::Start::True);

    int iter = 1;
    for (; iter < max_iter; ++iter)
    {
        // Compute P and check if acceptable
        ComputeP(comm, method, grad, rb_hess, cg, delta, p, verbose);
        linalgcpp::Add(1.0, x, 1.0, p, 0.0, x_propose);

        rb_hess.Mult(p, Bp);
        double f_propose = rb.Eval(x_propose);
        double m_0 = -1.0 * ParMult(comm, grad, p);
        double pBp = ParMult(comm, p, Bp);
        double m_p = -0.5 * pBp;
        double rho = (f - f_propose) / (m_0 - m_p);

        double p_norm = ParL2Norm(comm, p);

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
            g_norm = ParL2Norm(comm, grad);
            e_norm = ParL2Norm(comm, error);

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
            ParPrint(myid, printf("%d: f: %.2e g: %.2e p: %.2e e: %.2e grad*p: %.2e cg: %d delta: %.3f rho: %.3f\n",
                    iter, f, g_norm, p_norm, e_norm, /*notright*/grad * p, cg.GetNumIterations(), delta, rho));
        }

        if (g_norm < tol)
        {
            break;
        }
    }

    timer.Click();

    ParPrint(myid, printf("\n%s Stats:\n------------------------\n", method.c_str()));
    ParPrint(myid, printf("Total Time:\t%.5f\n", timer.TotalTime()));
    ParPrint(myid, printf("f(x):\t%.2e\nIter:\t%d\n", f, iter));
    ParPrint(myid, printf("Function Evals:\t%d\nGrad Evals:\t%d\nHessian Apply:\t%d\n",
            rb.num_evals, rb_grad.num_evals, rb_hess.num_evals));

    if (save_history)
    {
        write_history(x_history, "x", rb_A);
        write_history(g_history, "g", rb_A);
        write_history(f_history, "f", rb_A);
    }

    return 0;
}
