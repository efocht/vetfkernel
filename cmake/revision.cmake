execute_process(
        COMMAND git log -1 --format=%h
        WORKING_DIRECTORY ${TOPDIR}
        OUTPUT_VARIABLE VETFKERNEL_REVISION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )

execute_process(
        COMMAND git log -1 --format=%h
        WORKING_DIRECTORY ${TOPDIR}/libs/vednn
        OUTPUT_VARIABLE VEDNN_REVISION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )

message(VETFKERNEL_REVISION=${VETFKERNEL_REVISION})
message(VEDNN_REVISION=${VEDNN_REVISION})

configure_file(
        ${TOPDIR}/src/revision.h.in
        ${CMAKE_BINARY_DIR}/revision.h)

