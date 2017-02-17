//----------------------------------------------------------------------------------------------------------------------
//
// Copyright (c) 2014 Christer S.
// All Rights Reserved.
//
// Unauthorized copying of this file, via any medium is strictly prohibited.
//
//----------------------------------------------------------------------------------------------------------------------

#include "neuron.h"

#include <iostream>
#include <cmath>
#include <cstdlib>

Neuron::Neuron(size_t outputs, size_t neuron_id)

{
    for (size_t c = 0; c < outputs; ++c)
    {
        m_output_weights.emplace_back(Connection());
        m_output_weights.back().m_weight = random_weight();
    }

    m_neuron_id = neuron_id;
}

//----------------------------------------------------------------------------------------------------------------------

void Neuron::set_output_value(double value)
{
    m_output_value = value;
}

//----------------------------------------------------------------------------------------------------------------------
// take previous layer and sum all outputs (our inputs)
// include the bias node
void Neuron::feed_forward(const Network::Layer& prev_layer)
{
    double sum = 0.0;

    for (size_t n = 0; n < prev_layer.size(); ++n)
    {
        sum += prev_layer[n].get_output_value()
               * prev_layer[n].m_output_weights[m_neuron_id].m_weight;
    }

    m_output_value = Neuron::transfer_function(sum);
}

//----------------------------------------------------------------------------------------------------------------------

void Neuron::calc_output_gradients(double target_value)
{
    double delta = target_value - m_output_value;

    m_gradient = delta * Neuron::transfer_function_derivative(m_output_value);
}

//----------------------------------------------------------------------------------------------------------------------

void Neuron::calc_hidden_gradients(const Network::Layer& next_layer)
{
    double dow = sum_dow(next_layer);

    m_gradient = dow * Neuron::transfer_function_derivative(m_output_value);
}

//----------------------------------------------------------------------------------------------------------------------

double Neuron::get_output_value() const
{
    return m_output_value;
}

//----------------------------------------------------------------------------------------------------------------------

void Neuron::update_input_weights(Network::Layer& prev_layer)
{
    for (size_t n = 0; n < prev_layer.size(); ++n)
    {
        Neuron& neuron = prev_layer[n];
        double old_delta_weight
            =neuron.m_output_weights[m_neuron_id].m_deltaWeight;

        double new_delta_weight
            =// Individual input, magnified by the gradient and train rate:
              eta
              * neuron.get_output_value()
              * m_gradient
              // Also add momentum = a fraction of the previous delta weight;
              + alpha
              * old_delta_weight;

        neuron.m_output_weights[m_neuron_id].m_deltaWeight = new_delta_weight;
        neuron.m_output_weights[m_neuron_id].m_weight += new_delta_weight;
    }
}

//----------------------------------------------------------------------------------------------------------------------

double Neuron::sum_dow(const Network::Layer& next_layer) const
{
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed.

    for (size_t n = 0; n < next_layer.size() - 1; ++n)
    {
        sum += m_output_weights[n].m_weight * next_layer[n].m_gradient;
    }

    return sum;
}

//----------------------------------------------------------------------------------------------------------------------

double Neuron::transfer_function(double x)
{
    return std::tanh(x);
}

//----------------------------------------------------------------------------------------------------------------------

double Neuron::transfer_function_derivative(double x)
{
    // tanh derivative
    return 1.0 - x * x;
}

//----------------------------------------------------------------------------------------------------------------------

double Neuron::random_weight()
{
    return std::rand() / double(RAND_MAX);
}
