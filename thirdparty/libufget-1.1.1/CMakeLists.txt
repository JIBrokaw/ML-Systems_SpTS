CMAKE_MINIMUM_REQUIRED(VERSION 3.15.0)
CMAKE_POLICY(SET CMP0048 NEW)
CMAKE_POLICY(SET CMP0060 NEW)

# Assuming the canonical version is listed in a single line
# This would be in several parts if picking up from MAJOR, MINOR, etc.
set(VERSION_REGEX "#define LIBUFGET_VERSION_STRING[ \t]+\"(.+)\"")

# Read in the line containing the version
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/src/libufget.h"
    VERSION_STRING REGEX ${VERSION_REGEX})

# Pick out just the version
string(REGEX REPLACE ${VERSION_REGEX} "\\1" VERSION_STRING "${VERSION_STRING}")
string(STRIP ${VERSION_STRING} VERSION_STRING)
MESSAGE(STATUS "Configure libufget ${VERSION_STRING}")


PROJECT(LIBUFGET VERSION ${VERSION_STRING} LANGUAGES C)

### Require out-of-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()



OPTION(BUILD_SHARED_LIBS "Build shared libraries" ON)
OPTION(EXAMPLES "Build Examples" ON)

INCLUDE(GNUInstallDirs)

IF(DEFINED SYSCONFDIR)
 SET(CMAKE_INSTALL_FULL_SYSCONFDIR ${SYSCONFDIR})
ENDIF()
MESSAGE(STATUS "SYSCONFDIR: ${CMAKE_INSTALL_FULL_SYSCONFDIR}")
MESSAGE(STATUS "LIBDIR:     ${CMAKE_INSTALL_FULL_LIBDIR}")
MESSAGE(STATUS "BINDIR:     ${CMAKE_INSTALL_FULL_BINDIR}")
MESSAGE(STATUS "INCLUDEDIR: ${CMAKE_INSTALL_FULL_INCLUDEDIR}")
MESSAGE(STATUS "MANDIR:     ${CMAKE_INSTALL_FULL_MANDIR}")
MESSAGE(STATUS "Compiler_ID: ${CMAKE_C_COMPILER_ID}")
# MESSAGE(STATUS "FCompiler_ID: ${CMAKE_Fortran_COMPILER_ID}")

set(CMAKE_SKIP_BUILD_RPATH FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
SET(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMakeModules")


set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


find_library(MATH_LIBRARY m)

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")

#
# Find libraries
#
FIND_PACKAGE(CURL REQUIRED)
#FIND_PACKAGE(OpenSSL REQUIRED)
FIND_PACKAGE(MATIO REQUIRED)
FIND_PACKAGE(SQLITE3 REQUIRED)
FIND_PACKAGE(Archive REQUIRED)
FIND_PACKAGE(ZLIB REQUIRED)
FIND_PACKAGE(BZip2 REQUIRED)
FIND_PACKAGE(LibLZMA REQUIRED)


ADD_SUBDIRECTORY(src)
IF (EXAMPLES STREQUAL ON )
    ADD_SUBDIRECTORY(examples)
ENDIF()


### Setup PKG_CONFIG
set(prefix ${CMAKE_INSTALL_PREFIX})
set(libdir ${CMAKE_INSTALL_FULL_LIBDIR})
set(PKG_CONFIG_DIR ${CMAKE_INSTALL_FULL_LIBDIR}/pkgconfig)
set(includedir ${CMAKE_INSTALL_PREFIX}/include/)
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/libufget.pc.in ${CMAKE_CURRENT_BINARY_DIR}/libufget.pc)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libufget.pc  DESTINATION ${PKG_CONFIG_DIR} )

INCLUDE(CMakePackageConfigHelpers)
WRITE_BASIC_PACKAGE_VERSION_FILE(
    libufget-config-version.cmake
    COMPATIBILITY SameMajorVersion
 )
INSTALL(EXPORT libufget-targets
        NAMESPACE libufget::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libufget
)

CONFIGURE_FILE(libufget-config.cmake.in libufget-config.cmake @ONLY)
INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/libufget-config.cmake"
             "${CMAKE_CURRENT_BINARY_DIR}/libufget-config-version.cmake"
              DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libufget
        )


