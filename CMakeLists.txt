cmake_minimum_required(VERSION 3.16.3)

project(graph)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_BUILD_TYPE Debug)
set(ubsan "ub sanitaser" ON)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(X11 REQUIRED)
include_directories(${GLFW_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)





IF(CMAKE_BUILD_TYPE MATCHES Debug)
  message(" \n ——————————► DEBUG ◄——————————")
  # debugging information in a binary file
  add_compile_options( -g3 -O0 ) 

  #add_compile_options(-Werror) # turnsform  warnings into errors
  
  # Warnings include  
  add_compile_options(
    -pedantic-errors
    -Wall
    -Wextra
    -pedantic
    -Wpedantic
    -std=c++20
    -Wfloat-equal
    -Wpointer-arith
    #-Wstrict-overflow=5 
    #-Wwrite-strings
    #-Waggregate-return
    #-Wswitch-default -Wswitch-enum
    #-Wconversion
    #-Wunreachable-code
    #-fstack-clash-protection -fstack-protector	-fcf-protection
    #-fasynchronous-unwind-tables


    #-Wcast-align
    #-Wcast-qual
    #-Wctor-dtor-privacy
    #-Wduplicated-branches
    #-Wduplicated-cond

    #-Wextra-semi
    #-Wlogical-op
    #-Wnon-virtual-dtor
    #-Wold-style-cast
  
    #-Woverloaded-virtual
    #-Wredundant-decls
    #-Wsign-conversion
    #-Wsign-promo
    #-Weffc++
    #-Woverloaded-virtual
    )



  IF(tsan)
    # https://clang.llvm.org/docs/ThreadSanitizer.html
    message("TREAD sanitazer ON") 
    add_compile_options(  -fsanitize=thread  ) 
  ENDIF()
  IF(asan) 
    # https://clang.llvm.org/docs/AddressSanitizer.html
    message("ADRESS sanitazer ON")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address ")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address  ")
    SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address ")
    add_compile_options( -fsanitize=address  ) 
  ENDIF()

  IF(msan) 
    #https://clang.llvm.org/docs/MemorySanitizer.html
    message("MEMORY sanitazer ON")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory  -fno-omit-frame-pointer ")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=memory  -fno-omit-frame-pointer  ")
   SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=memory   -fno-omit-frame-pointer ")
    add_compile_options( -fsanitize=memory  -fno-omit-frame-pointer  -fno-optimize-sibling-calls -fno-sanitize-memory-use-after-dtor -fsanitize-recover=all -O1 ) 
  ENDIF()

  IF(ubsan) 
    #https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html
    message("UB san  sanitazer ON")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=undefined")
    SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=undefined")
    add_compile_options(-fsanitize=undefined  -fsanitize=signed-integer-overflow,null,alignment -fsanitize-recover=all) 
ENDIF()



#  SET(LINTER_ENABLED TRUE)

  #include realis build type
ELSEIF(CMAKE_BUILD_TYPE MATCHES Release)
  message("————————► RELEASE ◄————————")

  add_compile_options(-O2 -DNDEBUG -fomit-frame-pointer)
  add_compile_definitions(RELEASE)


ELSEIF (CMAKE_BUILD_TYPE MATCHES Clang-tidy)
message("————————► clang-tidy ◄————————")

  set(CMAKE_C_COMPILER clang-12 FORCE)
  set(CMAKE_CPP_COMPILER clang++-12 FORCE)
  set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-checks=*")

  # error  
ELSE()
  message( FATAL_ERROR "ALARM Build type abort")
ENDIF()

file (GLOB CPP_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/graph/graph.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

set (SOURCE_FILES ${CPP_FILES})


add_executable(test ${CPP_FILES})

target_link_libraries(test PRIVATE
    OpenGL::GL
    glfw
    ${GLFW_LIBRARIES}
    ${X11_LIBRARIES}
    ${X11_Xi_LIB}
    ${X11_Xmu_LIB}
    GLEW::GLEW
    ${OPENGL_LIBRARIES} GLU
    ${GLUT_LIBRARIES}
)
