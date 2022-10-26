find_package(PkgConfig)

PKG_CHECK_MODULES(PC_GR_SPECSENSOR gnuradio-specsensor)

FIND_PATH(
    GR_SPECSENSOR_INCLUDE_DIRS
    NAMES gnuradio/specsensor/api.h
    HINTS $ENV{SPECSENSOR_DIR}/include
        ${PC_SPECSENSOR_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    GR_SPECSENSOR_LIBRARIES
    NAMES gnuradio-specsensor
    HINTS $ENV{SPECSENSOR_DIR}/lib
        ${PC_SPECSENSOR_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/gnuradio-specsensorTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GR_SPECSENSOR DEFAULT_MSG GR_SPECSENSOR_LIBRARIES GR_SPECSENSOR_INCLUDE_DIRS)
MARK_AS_ADVANCED(GR_SPECSENSOR_LIBRARIES GR_SPECSENSOR_INCLUDE_DIRS)
