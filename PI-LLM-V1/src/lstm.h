#ifndef LSTM_H
#define LSTM_H

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cassert>

using Eigen::VectorXd;
using Eigen::MatrixXd;

struct LSTM {

    int input_dim;
    int hidden_dim;
    int vocab_size;

    MatrixXd Wf, Wi, Wo, Wc;
    VectorXd bf, bi, bo, bc;

    MatrixXd Wy;
    VectorXd by;

    VectorXd h, c;

    // Store last step activations
    VectorXd last_f, last_i, last_o, last_g, last_logits;

    LSTM(int vocab_size_, int hidden_dim_)
        : input_dim(vocab_size_), hidden_dim(hidden_dim_), vocab_size(vocab_size_)
    {
        Wf = MatrixXd::Random(hidden_dim, input_dim + hidden_dim);
        Wi = MatrixXd::Random(hidden_dim, input_dim + hidden_dim);
        Wo = MatrixXd::Random(hidden_dim, input_dim + hidden_dim);
        Wc = MatrixXd::Random(hidden_dim, input_dim + hidden_dim);

        bf = VectorXd::Zero(hidden_dim);
        bi = VectorXd::Zero(hidden_dim);
        bo = VectorXd::Zero(hidden_dim);
        bc = VectorXd::Zero(hidden_dim);

        Wy = MatrixXd::Random(vocab_size, hidden_dim);
        by = VectorXd::Zero(vocab_size);

        h = VectorXd::Zero(hidden_dim);
        c = VectorXd::Zero(hidden_dim);

        last_f = VectorXd::Zero(hidden_dim);
        last_i = VectorXd::Zero(hidden_dim);
        last_o = VectorXd::Zero(hidden_dim);
        last_g = VectorXd::Zero(hidden_dim);
        last_logits = VectorXd::Zero(vocab_size);
    }

    void reset_state() {
        h.setZero();
        c.setZero();
    }

    VectorXd sigmoid(const VectorXd &x) { return 1.0 / (1.0 + (-x.array()).exp()); }

    VectorXd softmax(const VectorXd &x) {
        VectorXd shifted = x.array() - x.maxCoeff();
        VectorXd exps = shifted.array().exp();
        return exps / exps.sum();
    }

    VectorXd forward(const VectorXd &x) {

        assert(x.size() == input_dim && "Input vector size mismatch!");

        VectorXd combined(input_dim + hidden_dim);
        combined.segment(0, input_dim) = x;
        combined.segment(input_dim, hidden_dim) = h;

        VectorXd f = sigmoid(Wf * combined + bf);
        VectorXd i = sigmoid(Wi * combined + bi);
        VectorXd o = sigmoid(Wo * combined + bo);
        VectorXd g = (Wc * combined + bc).array().tanh();

        last_f = f;
        last_i = i;
        last_o = o;
        last_g = g;

        c = f.array() * c.array() + i.array() * g.array();
        h = o.array() * c.array().tanh();

        VectorXd logits = Wy * h + by;
        last_logits = logits;

        return softmax(logits);
    }
};

#endif
