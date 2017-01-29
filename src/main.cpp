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

//----------------------------------------------------------------------------------------------------------------------

int main(int argc, char* argv[])

{
    Network::Topology top;

    top.push_back(2);
    top.push_back(4);
    top.push_back(4);
    top.push_back(1);

    Network n(top);

    return 0;

}


