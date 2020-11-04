#include "lscm.h"
#include "vector_area_matrix.h"

#include <igl/cotmatrix.h>
#include <igl/repdiag.h>
#include <igl/massmatrix.h>
#include <igl/eigs.h>
#include <Eigen/SVD>
#include <iostream>

void lscm(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & F,
  Eigen::MatrixXd & U)
{
  // Replace with your code
  // U = V.leftCols(2);

  // Angle distortion setup
  Eigen::SparseMatrix<double> lapacian, diag_lapacian, vector_area;
  igl::cotmatrix(V, F, lapacian);
  igl::repdiag(lapacian, 2, diag_lapacian);
  vector_area_matrix(F, vector_area);
  Eigen::SparseMatrix<double> Q = diag_lapacian - vector_area;

  // Free boundary constraint setup
  Eigen::SparseMatrix<double> mass, B;
  igl::massmatrix(V, F, igl::MassMatrixType::MASSMATRIX_TYPE_DEFAULT, mass);
  igl::repdiag(mass, 2, B);

  Eigen::MatrixXd eigen_vectors;
  igl::eigs(Q, B, 3, igl::EigsType::EIGS_TYPE_SM, eigen_vectors, Eigen::VectorXd());
  // std::cout << "eigen vector size: " << eigen_vectors.size() << std::endl;

  int V_size = V.rows();
  U.resize(V_size, 2);
  U.setZero();

  U.col(0) = eigen_vectors.col(2).head(V_size);
  U.col(1) = eigen_vectors.col(2).tail(V_size);

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(U.transpose() * U, Eigen::ComputeThinU | Eigen::ComputeThinV);
  U = U * svd.matrixU();
}
