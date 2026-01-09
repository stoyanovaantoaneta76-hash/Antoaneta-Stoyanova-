# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/botir-khaltaev/repos/adaptive/nordlys/nordlys-core/build-profile/_deps/usearch-src")
  file(MAKE_DIRECTORY "/home/botir-khaltaev/repos/adaptive/nordlys/nordlys-core/build-profile/_deps/usearch-src")
endif()
file(MAKE_DIRECTORY
  "/home/botir-khaltaev/repos/adaptive/nordlys/nordlys-core/build-profile/_deps/usearch-build"
  "/home/botir-khaltaev/repos/adaptive/nordlys/nordlys-core/build-profile/_deps/usearch-subbuild/usearch-populate-prefix"
  "/home/botir-khaltaev/repos/adaptive/nordlys/nordlys-core/build-profile/_deps/usearch-subbuild/usearch-populate-prefix/tmp"
  "/home/botir-khaltaev/repos/adaptive/nordlys/nordlys-core/build-profile/_deps/usearch-subbuild/usearch-populate-prefix/src/usearch-populate-stamp"
  "/home/botir-khaltaev/repos/adaptive/nordlys/nordlys-core/build-profile/_deps/usearch-subbuild/usearch-populate-prefix/src"
  "/home/botir-khaltaev/repos/adaptive/nordlys/nordlys-core/build-profile/_deps/usearch-subbuild/usearch-populate-prefix/src/usearch-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/botir-khaltaev/repos/adaptive/nordlys/nordlys-core/build-profile/_deps/usearch-subbuild/usearch-populate-prefix/src/usearch-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/botir-khaltaev/repos/adaptive/nordlys/nordlys-core/build-profile/_deps/usearch-subbuild/usearch-populate-prefix/src/usearch-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
