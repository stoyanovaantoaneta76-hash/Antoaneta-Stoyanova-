include(FindPackageHandleStandardArgs)
set(${CMAKE_FIND_PACKAGE_NAME}_CONFIG ${CMAKE_CURRENT_LIST_FILE})
find_package_handle_standard_args(usearch CONFIG_MODE)

if(NOT TARGET usearch::usearch)
    include("${CMAKE_CURRENT_LIST_DIR}/usearchTargets.cmake")
    if((NOT TARGET usearch) AND
       (NOT usearch_FIND_VERSION OR
        usearch_FIND_VERSION VERSION_LESS 3.2.0))
        add_library(usearch INTERFACE IMPORTED)
        set_target_properties(usearch PROPERTIES
            INTERFACE_LINK_LIBRARIES usearch::usearch
        )
    endif()
endif()
