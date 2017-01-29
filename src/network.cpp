//----------------------------------------------------------------------------------------------------------------------
//
// Copyright (c) 2014 Christer S.
// All Rights Reserved.
//
// Unauthorized copying of this file, via any medium is strictly prohibited.
//
//----------------------------------------------------------------------------------------------------------------------


#include "network.h"
#include "neuron.h"

#include <iostream>
#include <cmath>

Network::Network(Topology& topology)

{
    std::cout << "Instantiating network" << std::endl;
    size_t layers = topology.size();

    // store each array of layers
    for (size_t layer_num = 0; layer_num < layers; ++layer_num)
    {
        m_layers.emplace_back(Layer());

        // each layer has an amount of outputs equal to the next layer
        // exept for the last layer which has no outputs
        auto outputs = layer_num == topology.size() - 1 ? 0 : topology[layer_num + 1];

        m_layers.resize(topology[layer_num]);

        // for each new layer, fill it with neurons and add a bias
        for (size_t neuron_id = 0; neuron_id <= topology[layer_num]; ++neuron_id)
        {
            m_layers.back().emplace_back(Neuron(outputs, neuron_id));
        }

        // last entry in all vectors is the bias node, set it to 1.0
        m_layers.back().back().set_output_value(1.0);
    }
}


//----------------------------------------------------------------------------------------------------------------------

void Network::feed_forward(InputValues& input_values)
{
    for (size_t i = 0; i < input_values.size(); ++i)
    {
        m_layers[0][i].set_output_value(input_values[i]);
    }

    // propagate forward
    for (size_t layer_num = 1; layer_num < m_layers.size(); ++layer_num)
    {
        Layer& prev_layer = m_layers[layer_num - 1];

        for (size_t n = 0; n < m_layers[layer_num].size() - 1; ++n)
        {
            m_layers[layer_num][n].feed_forward(prev_layer);
        }
    }
}


//----------------------------------------------------------------------------------------------------------------------

void Network::back_propagation(const TargetValues& target_values)
{
    Layer& output_layer = m_layers.back();

    m_error = 0.0;

    for (size_t n = 0; n < output_layer.size() - 1; ++n)
    {
        double delta = target_values[n] - output_layer[n].get_output_value();

        m_error += delta * delta;
    }

    m_error /= output_layer.size() - 1;// get average squared
    m_error = std::sqrt(m_error);

    // implement a recent average measurement
    m_recent_average_error = (m_recent_average_error * m_recent_average_smoothing_factor + m_error) /
                             (m_recent_average_smoothing_factor + 1.0);

    // calculate output layer gradients
    for (size_t n = 0; n < output_layer.size() - 1; ++n)
    {
        output_layer[n].calc_output_gradients(target_values[n]);
    }

    // calculate hidden layer gradients
    for (size_t layer_num = m_layers.size() - 2; layer_num > 0; --layer_num)
    {
        Layer& hidden_layer = m_layers[layer_num];
        Layer& next_layer = m_layers[layer_num + 1];

        for (size_t n = 0; n < hidden_layer.size(); ++n)
        {
            hidden_layer[n].calc_hidden_gradients(next_layer);
        }
    }

    // for all layers from outputs to first hidden layer, update connection weights
    for (size_t layer_num = m_layers.size() - 1; layer_num > 0; --layer_num)
    {
        Layer& layer = m_layers[layer_num];
        Layer& prev_layer = m_layers[layer_num - 1];

        for (size_t n = 0; n < layer.size() - 1; ++n)
        {
            layer[n].update_input_weights(prev_layer);
        }
    }
}


//----------------------------------------------------------------------------------------------------------------------

void Network::results(Network::ResultValues& result_vals) const
{
    result_vals.clear();

    for (size_t n = 0; n < m_layers.back().size() - 1; ++n)
    {
        result_vals.push_back(m_layers.back()[n].get_output_value());
    }
}


