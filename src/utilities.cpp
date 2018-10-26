
#include "utilities.hpp"

namespace opti
{

void write_history(const std::vector<Vector>& x, const std::string& prefix, double A)
{
    std::string filename = prefix + "." + std::to_string(A) + "." +
                           "history.txt";
    std::ofstream output(filename);
    output.precision(5);
    output << std::scientific;

    for (auto&& x_i : x)
    {
        for (auto&& i : x_i)
        {
            output << i << " ";
        }

        output << "\n";
    }
}

void write_history(const std::vector<double>& x, const std::string& prefix, double A)
{
    std::string filename = prefix + "." + std::to_string(A) + "." +
                           "history.txt";
    std::ofstream output(filename);

    for (auto&& x_i : x)
    {
        output << x_i << "\n";
    }
}
} // namespace opti
