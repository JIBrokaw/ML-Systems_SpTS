INCLUDE(GenerateExportHeader)

SET(SRC
	conv.c
	uf_collection.c
	uf_matrix.c
	uf_query.c
	uf_query_pre.c
	uf_read.c
	sql.c
	io/io.c
	io/compress_handler.c
	io/compress_io_bzip.c
	io/compress_io_gzip.c
	io/compress_io_xz.c
	io/compress_io_uncompressed.c
    libufget.h
    uf_internal.h
)

ADD_LIBRARY(ufget ${SRC})
TARGET_LINK_LIBRARIES(ufget
    PRIVATE
        ${ARCHIVE_LIBRARIES}
        ${ZLIB_LIBRARIES}
        ${LIBLZMA_LIBRARIES}
        ${BZIP2_LIBRARIES}
        ${CURL_LIBRARIES}
        ${OPENSSL_LIBRARIES}
        ${MATIO_LIBRARIES}
        ${SQLITE3_LIBRARIES}
        ${MATH_LIBRARY}
    )
TARGET_INCLUDE_DIRECTORIES(ufget
    PRIVATE
        ${ARCHIVE_INCLUDE_DIR}
        ${CURL_INCLUDE_DIR}
        ${OPENSSL_INCLUDE_DIR}
        ${MATIO_INCLUDE_DIR}
        ${SQLITE3_INCLUDE_DIR}
        ${BZIP2_INDCUDE_DIR}
        ${ZLIB_INCLUDE_DIR}
        ${LIBLZMA_INCLUDE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_FULL_INCLUDEDIR}>
    )

generate_export_header(ufget
    EXPORT_FILE_NAME libufget_exports.h
    INCLUDE_GUARD_NAME LIBUFGET_EXPORTS_H
    BASE_NAME libufget
    )

set_target_properties(ufget PROPERTIES
    SOVERSION
        "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}"
    VERSION
        "${PROJECT_VERSION_MAJOR}")

set_property(TARGET ufget
    PROPERTY PUBLIC_HEADER
        libufget.h
        ${CMAKE_CURRENT_BINARY_DIR}/libufget_exports.h)

set_target_properties(ufget PROPERTIES C_VISIBILITY_PRESET hidden)

ADD_EXECUTABLE(ufget-update ufget-update.c)
TARGET_LINK_LIBRARIES(ufget-update PRIVATE ufget)

# INSTALL

INSTALL(TARGETS ufget
        EXPORT  libufget-targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_FULL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_FULL_INCLUDEDIR}
        )

INSTALL(TARGETS ufget-update
        RUNTIME DESTINATION ${CMAKE_INSTALL_FULL_BINDIR})

