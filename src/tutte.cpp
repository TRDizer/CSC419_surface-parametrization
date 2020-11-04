#include "tutte.h"
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/cotmatrix.h>
#include <igl/edges.h>
#include <igl/min_quad_with_fixed.h>

void tutte(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & F,
  Eigen::MatrixXd & U)
{
  // Replace with your code
  // U = V.leftCols(2);
  // Borrowed the exact names from libigl tutorial
  Eigen::VectorXi bnd;
  igl::boundary_loop(F, bnd);
  bnd.reverseInPlace(); // flip beetle surface

  Eigen::MatrixXd bnd_uv;
  igl::map_vertices_to_circle(V, bnd, bnd_uv);

  Eigen::SparseMatrix<double> lapacian;
  // igl::cotmatrix(V, F, lapacian); // non-inverse-weighted cotan gives weird looking frames for beetle mesh
  lapacian.resize(V.rows(), V.rows());
  lapacian.setZero();

  Eigen::MatrixXi edges;
  igl::edges(F, edges);
  int vi, vj;
  double w;
  for (int i = 0; i < edges.rows(); i++) {
    vi = edges(i,0);
    vj = edges(i,1);
    w = 1. / (V.row(vi) - V.row(vj)).norm();
    lapacian.insert(vi, vj) = w;
    lapacian.insert(vj, vi) = w;
  }
  for (int i = 0; i < lapacian.rows(); i++) {
    lapacian.insert(i,i) = -lapacian.row(i).sum();
  }

  // min U wrt 1/2 tr(U^T * L * U) and no U^T * B term
  igl::min_quad_with_fixed_data<double> data;
  igl::min_quad_with_fixed_precompute(lapacian, bnd, Eigen::SparseMatrix<double>(), false, data);
  igl::min_quad_with_fixed_solve(data, Eigen::VectorXd::Zero(V.rows()), bnd_uv, Eigen::VectorXd(), U);
}

