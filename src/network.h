//----------------------------------------------------------------------------------------------------------------------
//
// Copyright (c) 2014 Christer S.
// All Rights Reserved.
//
// Unauthorized copying of this file, via any medium is strictly prohibited.
//
//----------------------------------------------------------------------------------------------------------------------

#ifndef NETWORK_H
#define NETWORK_H

#include <cstddef>
#include <cstdint>
#include <spdlog/logger.h>
#include <vector>

class Neuron;
class Network
{
public:
    using Topology = std::vector<uint64_t>;
    using Layer = std::vector<Neuron>;
    using InputValues = std::vector<double>;
    using TargetValues = std::vector<double>;
    using ResultValues = std::vector<double>;
    /**
     * The topology describes the layout of the neuron field.
     *
     */
    Network(Topology& topology, spdlog::logger& log);

    void feed_forward(InputValues& input_values);
    void back_propagation(const TargetValues& target_values);
    void results(ResultValues& result_vals) const;

    void add_logger();

private:
    double m_recent_average_smoothing_factor;
    std::vector<Layer> m_layers;
    double m_error;
    double m_recent_average_error;

    spdlog::logger& m_log;
};

#endif // NETWORK_H
