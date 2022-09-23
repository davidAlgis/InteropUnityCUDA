#pragma once
#include <stddef.h>
#include "framework.h"

#include <assert.h>
#if UNITY_IOS || UNITY_TVOS
#	include <OpenGLES/ES2/gl.h>
#elif UNITY_ANDROID || UNITY_WEBGL
#	include <GLES2/gl2.h>
#elif UNITY_OSX
#	include <OpenGL/gl3.h>
#elif UNITY_WIN
// On Windows, use gl3w to initialize and load OpenGL Core functions. In principle any other
// library (like GLEW, GLFW etc.) can be used; here we use gl3w since it's simple and
// straightforward.
#	include "gl3w.h"
#elif UNITY_LINUX
#	define GL_GLEXT_PROTOTYPES
#	include <GL/gl.h>
#elif UNITY_EMBEDDED_LINUX
#	include <GLES2/gl2.h>
#if SUPPORT_OPENGL_CORE
#	define GL_GLEXT_PROTOTYPES
#	include <GL/gl.h>
#endif
#else
#	error Unknown platform
#endif
#include <string>
#include <stdio.h>
#include <log.h>

#define GL_CHECK(){ glAssert( __FILE__, __LINE__);} 
inline VOID glAssert(const char* file, const int line)
{
	// check for error
	GLenum gl_error = glGetError();

	if (gl_error != GL_NO_ERROR)
	{
		char buffer[512];
		snprintf(buffer, 100, "0x0%x", gl_error);
		std::string fileStr(file);
		std::string errorStr(buffer);
		std::string s = "OpenGL error " + errorStr + " at : " + fileStr + " - " + std::to_string(line);
		//sprintf(buffer, "OpenGL error %08x, at %s:%i - for %s\n", gl_error, file, line);
		//std::string strError(buffer);
		Log::log().debugLog(s);
	}
}