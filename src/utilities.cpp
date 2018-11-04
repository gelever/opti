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

void RingUpdate(MPI_Comm comm, const VectorView& x, VectorView x_bdr)
{
    int myid;
    int num_procs;
    int tag = 0;

    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &num_procs);
    std::vector<MPI_Request> requests(4);

    int proc_right = (myid + 1) % num_procs;
    int proc_left = myid == 0 ? num_procs - 1 : (myid - 1);

    MPI_Irecv(&x_bdr[0], 1, MPI_DOUBLE, proc_left, tag + 1, comm, &requests[0]);
    MPI_Irecv(&x_bdr[1], 1, MPI_DOUBLE, proc_right, tag, comm, &requests[1]);

    MPI_Isend(&x.front(), 1, MPI_DOUBLE, proc_left, tag, comm, &requests[2]);
    MPI_Isend(&x.back(), 1, MPI_DOUBLE, proc_right, tag + 1, comm, &requests[3]);

    std::vector<MPI_Status> status(4);
    MPI_Waitall(4, requests.data(), status.data());
}

} // namespace opti
