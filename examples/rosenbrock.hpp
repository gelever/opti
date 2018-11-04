/* 
 * Stephan Gelever
 * Math 510
 * HW 2
 *
 * Copyright (c) 2018, Stephan Gelever
 */

#include "utilities.hpp"

using namespace opti;

namespace rosenbrock
{

class Rosenbrock
{
    public:
        Rosenbrock(MPI_Comm comm_in, double A_in, int num_dim = 2)
            : comm(comm_in), A(A_in), global_dim(num_dim), num_evals(0)
        {
            assert(num_dim >= 2); assert(A >= 1.0);

            MPI_Comm_size(comm, &num_procs);
            MPI_Comm_rank(comm, &myid);

            local_dim = global_dim / num_procs;

            if (myid == 0)
            {
                local_dim += global_dim % num_procs;
            }

            linalgcpp_verify(local_dim > 0);

            offsets = linalgcpp::GenerateOffsets(comm, local_dim);
        }

        double Eval(const VectorView& x) const;

        MPI_Comm comm;
        int myid;
        int num_procs;

        double A;

        int local_dim;
        int global_dim;
        std::vector<int> offsets;

        mutable int num_evals;
};

class Hessian : public Operator
{
    public:
        Hessian(const Rosenbrock& rb_in, const VectorView& x_in)
            : Operator(rb_in.local_dim), rb(rb_in), x(x_in), num_evals(0)
        {}

        using Operator::Mult;
        void Mult(const VectorView& input, VectorView output) const override
        {
            double A = rb.A;
            int dim = rb.local_dim;
            int comm = rb.comm;
            int myid = rb.myid;
            int num_procs = rb.num_procs;

            Vector x_bdr(2);
            Vector input_bdr(2);

            RingUpdate(comm, x, x_bdr);
            RingUpdate(comm, input, input_bdr);

            if (myid == 0)
            {
                output[0] = ((12 * A * x[0] * x[0]) - (4 * A * x[1]) + 2) * input[0] - (4 * A * x[0] * input[1]);
            }
            else
            {
                output[0] = (-4.0 * A * x_bdr[0] * input_bdr[0]) + ((2 * A) + 2 + (12 * A * x[0] * x[0]) - (4.0 * A * x[1])) * input[0] - (4.0 * A * x[0] * input[1]);
            }

            for (int i = 1; i < dim - 1; ++i)
            {
                output[i] = (-4.0 * A * x[i - 1] * input[i - 1]) + ((2 * A) + 2 + (12 * A * x[i] * x[i]) - (4.0 * A * x[i + 1])) * input[i] - (4.0 * A * x[i] * input[i + 1]);
            }

            if (myid == num_procs - 1)
            {
                output[dim - 1] = (-4.0 * A * x[dim - 2] * input[dim - 2]) + (2.0 * A * input[dim - 1]);
            }
            else
            {
                output[dim - 1] = (-4.0 * A * x[dim - 2] * input[dim - 2]) + ((2 * A) + 2 + (12 * A * x[dim - 1] * x[dim - 1]) - (4.0 * A * x_bdr[1])) * input[dim - 1] - (4.0 * A * x[dim - 1] * input_bdr[1]);
            }

            num_evals++;
        }

        const Rosenbrock& rb;
        const VectorView& x;

        mutable int num_evals;
};

class Gradient : public Operator
{
    public:
        Gradient(const Rosenbrock& rb_in)
            : Operator(rb_in.local_dim), rb(rb_in), num_evals(0)
        {}

        using Operator::Mult;
        void Mult(const VectorView& x, VectorView df_x) const override
        {
            double A = rb.A;
            int local_dim = rb.local_dim;
            int comm = rb.comm;
            int myid = rb.myid;
            int num_procs = rb.num_procs;

            Vector x_bdr(2);
            RingUpdate(comm, x, x_bdr);

            assert(x.size() == local_dim);
            assert(df_x.size() == local_dim);

            if (myid == 0)
            {
                df_x[0] = (-4.0 * A * x[0] * (x[1] - std::pow(x[0], 2))) - (2.0 * (1.0 - x[0]));
            }
            else
            {
                df_x[0] = (2.0 * A * (x[0] - std::pow(x_bdr[0], 2)))
                          - (4 * A * x[0] * (x[1] - std::pow(x[0], 2)))
                          - (2 * (1 - x[0]));
            }

            for (int i = 1; i < local_dim - 1; ++i)
            {
                df_x[i] = (2.0 * A * (x[i] - std::pow(x[i - 1], 2)))
                          - (4 * A * x[i] * (x[i + 1] - std::pow(x[i], 2)))
                          - (2 * (1 - x[i]));
            }

            if (myid == num_procs - 1)
            {
                df_x[local_dim - 1] = 2.0 * A * (x[local_dim - 1] - std::pow(x[local_dim - 2], 2));
            }
            else
            {
                df_x[local_dim - 1] = (2.0 * A * (x[local_dim - 1] - std::pow(x[local_dim - 2], 2)))
                          - (4 * A * x[local_dim - 1] * (x_bdr[1] - std::pow(x[local_dim - 1], 2)))
                          - (2 * (1 - x[local_dim - 1]));
            }

            num_evals++;
        }

        const Rosenbrock& rb;

        mutable int num_evals;
};

class HessianDiagInv : public Operator
{
    public:
        HessianDiagInv(const Rosenbrock& rb_in, const VectorView& x_in)
            : Operator(rb_in.local_dim), rb(rb_in), x(x_in)
        {}

        using Operator::Mult;
        void Mult(const VectorView& input, VectorView output) const override
        {
            int dim = rb.local_dim;
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
    Vector x_bdr(2);
    RingUpdate(comm, x, x_bdr);

    double sum = 0.0;

    for (int i = 0; i < local_dim - 1; ++i)
    {
        double p1 = (x[i + 1] - std::pow(x[i], 2));
        double p2 = (1 - x[i]);

        sum += (A * std::pow(p1, 2)) + std::pow(p2, 2);
    }

    if (myid != num_procs - 1)
    {
        double p1 = (x_bdr[1] - std::pow(x[local_dim - 1], 2));
        double p2 = (1 - x[local_dim - 1]);

        sum += (A * std::pow(p1, 2)) + std::pow(p2, 2);
    }

    double global_sum;

    MPI_Allreduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);

    num_evals++;

    return global_sum;
}

void alternate_x(VectorView x, bool offset)
{
    int n = x.size();

    for (int i = offset; i < n; i += 2)
    {
        x[i] = -1.2;
    }
    for (int i = !offset; i < n; i += 2)
    {
        x[i] = 1.0;
    }
}

Vector set_x(const Rosenbrock& rb, const std::string& method, double variance)
{
    Vector x(rb.local_dim);

    if (method == "Standard")
    {
        bool offset = (rb.offsets[0] % 2);
        alternate_x(x, offset);
    }
    else if (method == "Random")
    {
        double lo = 1.0 - variance;
        double hi = 1.0 + variance;
        linalgcpp::Randomize(x, lo, hi);
    }
    else
    {
        throw std::runtime_error("Invalid inital x method!");
    }

    return x;
}


} // namespace rosenbrock
