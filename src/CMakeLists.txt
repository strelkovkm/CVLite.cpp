add_library(cvlite STATIC
    core/shape.cpp
    core/tensor.cpp
    data/dataloader.cpp
    layers/conv3x3.cpp
    layers/fullyconnected.cpp
    layers/relu.cpp
    layers/softmax.cpp
    losses/cross-entropy.cpp
    optimize/adam.cpp
    pipeline/sequential.cpp
)

target_include_directories(cvlite
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(cvlite PUBLIC cxx_std_20)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cvlite PRIVATE -Wall -Wextra -Wpedantic)
endif()
