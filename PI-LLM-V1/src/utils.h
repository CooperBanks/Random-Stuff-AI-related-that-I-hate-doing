//honestly forgot what this does

#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
using Eigen::VectorXd;

VectorXd softmax(const VectorXd &x) {
	VectorXd e = x.array().exp();
	return e / e.sum();
}

#endif
