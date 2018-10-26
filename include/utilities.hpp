#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <iostream>
#include "linalgcpp.hpp"

namespace opti
{

using DenseMatrix = linalgcpp::DenseMatrix;
using Vector = linalgcpp::Vector<double>;
using VectorView = linalgcpp::VectorView<double>;
using Operator = linalgcpp::Operator;

void write_history(const std::vector<Vector>& x, const std::string& prefix, double A);

void write_history(const std::vector<double>& x, const std::string& prefix, double A);


} // namespace opti

#endif // UTILITIES_HPP
