#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>

namespace py = pybind11;

py::array_t<double> invert_matrix(py::array_t<double> input) {
    py::buffer_info buf = input.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("Input should be a 2D array");
    }
    size_t rows = buf.shape[0];
    size_t cols = buf.shape[1];

    if (rows != cols) {
        throw std::runtime_error("Input should be a square matrix");
    }

    Eigen::Map<Eigen::MatrixXd> mat(static_cast<double *>(buf.ptr), rows, cols);
    Eigen::MatrixXd inv_mat = mat.inverse();

    py::array_t<double> result({rows, cols});
    py::buffer_info buf_result = result.request();
    double *ptr_result = static_cast<double *>(buf_result.ptr);
    Eigen::Map<Eigen::MatrixXd>(ptr_result, rows, cols) = inv_mat;

    return result;
}

PYBIND11_MODULE(matrix_inversion, m) {
    m.def("invert_matrix", &invert_matrix, "A function that inverts a matrix");
}