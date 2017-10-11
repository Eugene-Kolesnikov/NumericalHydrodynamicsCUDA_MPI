#
# Generated Makefile - do not edit!
#
# Edit the Makefile in the project folder instead (../Makefile). Each target
# has a -pre and a -post target defined where you can add customized code.
#
# This makefile implements configuration specific macros and targets.


# Environment
MKDIR=mkdir
CP=cp
GREP=grep
NM=nm
CCADMIN=CCadmin
RANLIB=ranlib
CC=clang
CCC=clang++
CXX=clang++
FC=gfortran
AS=as

# Macros
CND_PLATFORM=CLang-MacOSX
CND_DLIB_EXT=dylib
CND_CONF=Release
CND_DISTDIR=dist
CND_BUILDDIR=build

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/drawSimpleField.o


# C Compiler Flags
CFLAGS=

# CC Compiler Flags
CCFLAGS=
CXXFLAGS=

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=-L../DynamicLibraryVisualization/dist/Release/CLang-MacOSX -L../../dist/Release/CLang-MacOSX -lDynamicLibraryVisualization

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/unittesttriangle
	${CP} ../../dist/Release/CLang-MacOSX/libDynamicLibraryVisualization.dylib ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	-install_name_tool -change libDynamicLibraryVisualization.dylib @executable_path/libDynamicLibraryVisualization.dylib ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/unittesttriangle

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/unittesttriangle: ../../dist/Release/CLang-MacOSX/libDynamicLibraryVisualization.dylib

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/unittesttriangle: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/unittesttriangle ${OBJECTFILES} ${LDLIBSOPTIONS}

${OBJECTDIR}/drawSimpleField.o: drawSimpleField.cpp
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/drawSimpleField.o drawSimpleField.cpp

# Subprojects
.build-subprojects:
	cd ../.. && ${MAKE}  -f Makefile CONF=Release

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}
	${RM} -r ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/libDynamicLibraryVisualization.dylib
	${RM} ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/unittesttriangle

# Subprojects
.clean-subprojects:
	cd ../.. && ${MAKE}  -f Makefile CONF=Release clean

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
