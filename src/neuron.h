//----------------------------------------------------------------------------------------------------------------------
//
// Copyright (c) 2014 Christer S.
// All Rights Reserved.
//
// Unauthorized copying of this file, via any medium is strictly prohibited.
//
//----------------------------------------------------------------------------------------------------------------------

#ifndef NEURON_H
#define NEURON_H

#include "network.h"

#include <cstddef>
#include <spdlog/spdlog.h>

struct Connection
{
    double m_weight;
    double m_deltaWeight;
};

class Neuron
{
public:
    Neuron(size_t outputs, size_t neuron_id);

    void set_output_value(double value);
    void feed_forward(const Network::Layer& prev_layer);
    void calc_output_gradients(double target_value);
    void calc_hidden_gradients(const Network::Layer& next_layer);
    double get_output_value() const;
    void update_input_weights(Network::Layer& prev_layer);

private:
    double sum_dow(const Network::Layer& next_layer) const;
    static double transfer_function(double x);
    static double transfer_function_derivative(double x);
    static double random_weight(void);

    double eta;// [0.0..1.0] overall net training rate
    double alpha;// [0.0..n] multiplier of last weight change (momentum)
    std::vector<Connection>m_output_weights;
    size_t m_neuron_id;
    double m_output_value;
    double m_gradient;
};

#endif// NEURON_H
