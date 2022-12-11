ENABLE_LANGUAGE(Fortran)
FIND_PACKAGE(BLAS REQUIRED)


SET (EXAMPLE_SRC
	demo_cg
	demo_download_all
	demo_convert_all
	)

foreach (srcfile ${EXAMPLE_SRC})
    get_filename_component(BASENAME ${srcfile} NAME_WE)
    add_executable (${BASENAME} ${srcfile}.c)
    target_link_libraries (${BASENAME}
        PRIVATE
            ufget
            ${BLAS_LIBRARIES}
            ${MATH_LIBRARY}
    )
endforeach (srcfile)

