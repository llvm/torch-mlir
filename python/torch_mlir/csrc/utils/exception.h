#pragma once

#include <exception>
#include <sstream>
#include <string>

#define UNIMPLEMENTED_ERROR(msg)                \
    {                                           \
        std::ostringstream err;                 \
        err << "Unimplemented Error: " << msg;  \
        throw std::runtime_error(err.str());    \
    }


#define UNSUPPORTED_ERROR(msg)                  \
    {                                           \
        std::ostringstream err;                 \
        err << "Unsupported Error: " << msg;    \
        throw std::runtime_error(err.str());    \
    }
