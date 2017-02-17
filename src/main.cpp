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

#include <cstdint>
#include <vector>
#include <iostream>
#include <memory>

#include <spdlog/spdlog.h>

//----------------------------------------------------------------------------------------------------------------------

static const size_t LOGSIZE = 5 * 1048576;

int main(int argc, char* argv[])

{
    // setup logging (to console and file)
    std::vector<spdlog::sink_ptr>sinks;
    spdlog::set_pattern("*** [%H:%M:%S:%MS %z] [thread %t] %v ***");
    sinks.push_back(std::make_shared<spdlog::sinks::stdout_sink_st>());
    sinks.push_back(std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                        "neuralnet", "log", LOGSIZE, 3));
    auto log = std::make_shared<spdlog::logger>("neuralnet", std::begin(
                                                    sinks), std::end(sinks));
    spdlog::register_logger(log);

    log->info("Neural network starting");

    Network::Topology top;

    top.push_back(128);
    top.push_back(256);
    top.push_back(256);
    top.push_back(64);
    top.push_back(1);

    Network n(top, *log);

    return 0;
}
