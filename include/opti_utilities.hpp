#ifndef OPTI_UTILITIES_HPP
#define OPTI_UTILITIES_HPP

#include <iostream>
#include "parlinalgcpp.hpp"

namespace opti
{

using DenseMatrix = linalgcpp::DenseMatrix;
using Vector = linalgcpp::Vector<double>;
using VectorView = linalgcpp::VectorView<double>;
using Operator = linalgcpp::Operator;
using MpiSession = linalgcpp::MpiSession;
using linalgcpp::Timer;
using linalgcpp::ParL2Norm;
using linalgcpp::linalgcpp_verify;
using linalgcpp::Timer;

void write_history(const std::vector<Vector>& x, const std::string& prefix, double A);

void write_history(const std::vector<double>& x, const std::string& prefix, double A);

void RingUpdate(MPI_Comm comm, const VectorView& x, VectorView x_bdr);

} // namespace opti

#endif // OPTI_UTILITIES_HPP
