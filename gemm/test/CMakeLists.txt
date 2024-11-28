
# https://github.com/xgqdut2016/hpc2torch/blob/main/CMakeLists.txt

cmake_minimum_required(VERSION 3.16)

project(MyCUDAProject)

# 查找 Python 库
find_package(Python3 REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})

# 设置 CUDA 编译选项
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")

# 添加选项控制不同的编译方式
option(USE_CUDA "Enable CUDA compilation" OFF)
option(USE_CPU "Enable CPU-only compilation" OFF)

# 查找源文件
# 添加头文件搜索路径  
include_directories(${PROJECT_SOURCE_DIR}/backends/include)
include_directories(${PROJECT_SOURCE_DIR}/utils/include)

#使用 GLOB 命令找到 include/ 下的所有 .cpp 文件
file(GLOB UTILS_CPP_SOURCE_FILES "${PROJECT_SOURCE_DIR}/backends/utils/src/**.cpp")
file(GLOB UTILS_CUDA_SOURCE_FILES "${PROJECT_SOURCE_DIR}/backends/utils/src/**.cu")

# 使用 list(APPEND ...) 命令将 INCLUDE_SOURCE_FILES 添加到 CPP_SOURCE_FILES
file(GLOB CPP_SOURCE_FILES "${PROJECT_SOURCE_DIR}/backends/**/*.cpp")
list(APPEND CPP_SOURCE_FILES ${UTILS_CPP_SOURCE_FILES})

file(GLOB CUDA_SOURCE_FILES "${PROJECT_SOURCE_DIR}/backends/**/*.cu")
list(APPEND CUDA_SOURCE_FILES ${UTILS_CUDA_SOURCE_FILES})

# 根据选项决定编译哪些源文件
if(USE_CUDA)
    message(STATUS "CUDA build enabled.")
    enable_language(CXX)
    enable_language(CUDA)
    list(APPEND ALL_SOURCE_FILES ${CUDA_SOURCE_FILES} ${CPP_SOURCE_FILES})
    add_library(my_linear SHARED ${ALL_SOURCE_FILES})# 创建库或可执行文件
elseif(USE_CPU)
    message(STATUS "CPU-only build enabled.")
    enable_language(CXX)
    list(APPEND ALL_SOURCE_FILES ${CPP_SOURCE_FILES})
    add_library(my_linear SHARED ${ALL_SOURCE_FILES})# 创建库或可执行文件
else()
    message(FATAL_ERROR "No valid compilation mode specified. Please enable USE_CUDA, USE_BANG, or USE_CPU.")
endif()




# 设置编译选项
target_compile_features(my_linear PUBLIC cxx_std_11)

# 链接 Python 库
target_link_libraries(my_linear PRIVATE ${Python3_LIBRARIES})

# 指定输出目录
set_target_properties(my_linear PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)