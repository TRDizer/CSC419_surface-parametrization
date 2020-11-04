#include "vector_area_matrix.h"
#include <igl/boundary_loop.h>
#include <iostream>

void vector_area_matrix(
  const Eigen::MatrixXi & F,
  Eigen::SparseMatrix<double>& A)
{
  // Replace with your code
  int V_size = F.maxCoeff()+1;
  A.resize(V_size*2,V_size*2);

  std::vector<std::vector<int>> bnds;
  igl::boundary_loop(F, bnds);

  //             | ei_u  ej_u |
  // det(E_ij) = |            | = ei_u * ej_v - ej+u * ei_v;
  //             | ei_v  ej_v |
  std::vector<Eigen::Triplet<double>> entries;
  int v1, v2;
  for (std::vector<int> bnd : bnds) {
    for (int i = 0; i < bnd.size(); i++) {
      v1 = bnd[i];
      v2 = bnd[(i + 1) % bnd.size()];

      // A
      entries.push_back(Eigen::Triplet<double>(v1, v2 + V_size, 1));
      entries.push_back(Eigen::Triplet<double>(v2, v1 + V_size, -1));

      // A^T
      entries.push_back(Eigen::Triplet<double>(v2 + V_size, v1, 1));
      entries.push_back(Eigen::Triplet<double>(v1 + V_size, v2, -1));
    }
  }

  A.setFromTriplets(entries.begin(), entries.end());
  A *= 0.5;
}

