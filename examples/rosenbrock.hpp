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
        Rosenbrock(double A_in, int num_dim_in = 2) : A(A_in), num_dim(num_dim_in), num_evals(0)
        { assert(num_dim >= 2); assert(A >= 1.0);}

        double Eval(const VectorView& x) const;

        double A;
        int num_dim;

        mutable int num_evals;
};

class Hessian : public Operator
{
    public:
        Hessian(const Rosenbrock& rb_in, const VectorView& x_in)
            : Operator(rb_in.num_dim), rb(rb_in), x(x_in), num_evals(0)
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

            num_evals++;
        }

        const Rosenbrock& rb;
        const VectorView& x;

        mutable int num_evals;
};

class Gradient : public Operator
{
    public:
        Gradient(const Rosenbrock& rb_in, const VectorView& x_in)
            : Operator(rb_in.num_dim), rb(rb_in), x(x_in), num_evals(0)
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
                df_x[i] = (2.0 * A * (x[i] - std::pow(x[i - 1], 2)))
                          - (4 * A * x[i] * (x[i + 1] - std::pow(x[i], 2)))
                          - (2 * (1 - x[i]));
            }

            df_x[num_dim - 1] = 2.0 * A * (x[num_dim - 1] - std::pow(x[num_dim - 2], 2));

            num_evals++;
        }

        const Rosenbrock& rb;
        const VectorView& x;

        mutable int num_evals;
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

void alternate_x(VectorView x)
{
    int n = x.size();

    for (int i = 0; i < n; i += 2)
    {
        x[i] = -1.2;
    }
    for (int i = 1; i < n; i += 2)
    {
        x[i] = 1.0;
    }
}

Vector set_x(int dim, const std::string& method, double variance)
{
    Vector x(dim);

    if (method == "Standard")
    {
        alternate_x(x);
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
