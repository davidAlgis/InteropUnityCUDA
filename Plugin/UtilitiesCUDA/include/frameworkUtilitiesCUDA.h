#pragma once

// More at
// https://cmake.org/Wiki/BuildingWinDLL
// https://gcc.gnu.org/wiki/Visibility

#define WIN32_LEAN_AND_MEAN             // Exclure les en-têtes Windows rarement utilisés
// Fichiers d'en-tête Windows
#include <windows.h>


// Generic helper definitions for shared library support
#if defined _WIN32 || (defined _WIN64) || (defined __CYGWIN__)
	#if defined __MINGW32__ || (defined __MINGW64__)
		// GCC with MINGW
	#define UTILITIES_CUDA_DLL_IMPORT __attribute__ ((visibility ("default")))
	#define UTILITIES_CUDA_DLL_EXPORT __attribute__ ((visibility ("default")))
	#define UTILITIES_CUDA_DLL_LOCAL  __attribute__ ((visibility ("hidden")))
	#elif defined _MSC_VER
		// Microsoft compiler
	#define UTILITIES_CUDA_DLL_IMPORT __declspec(dllimport)
	#define UTILITIES_CUDA_DLL_EXPORT __declspec(dllexport)
	#define UTILITIES_CUDA_DLL_LOCAL
	#else
		// Unknown
	#define UTILITIES_CUDA_DLL_IMPORT
	#define UTILITIES_CUDA_DLL_EXPORT
	#define UTILITIES_CUDA_DLL_LOCAL
	#endif
	#else
	#if __GNUC__ >= 4 || (defined __clang__)
		// GCC
	#define UTILITIES_CUDA_DLL_IMPORT __attribute__ ((visibility ("default")))
	#define UTILITIES_CUDA_DLL_EXPORT __attribute__ ((visibility ("default")))
	#define UTILITIES_CUDA_DLL_LOCAL  __attribute__ ((visibility ("hidden")))
	#else
		// GCC does not support __attribute__ before version 4.
	#define UTILITIES_CUDA_DLL_IMPORT
	#define UTILITIES_CUDA_DLL_EXPORT
	#define UTILITIES_CUDA_DLL_LOCAL
	#endif
#endif
// Now we use the generic helper definitions above to define UTILITIES_CUDA_API and UTILITIES_CUDA_LOCAL.
// UTILITIES_CUDA_API is used for the public API symbols. It either DLL imports or DLL exports (or does nothing for static build)
// UTILITIES_CUDA_LOCAL is used for non-api symbols.
#ifdef UTILITIES_CUDA_SHARED              // Defined if UTILITIES_CUDA is compiled as a DLL
#ifdef UTILITIES_CUDA_SHARED_EXPORTS  // Defined if we are building the UTILITIES_CUDA DLL (instead of using it)
#define UTILITIES_CUDA_API UTILITIES_CUDA_DLL_EXPORT
#else
#define UTILITIES_CUDA_API UTILITIES_CUDA_DLL_IMPORT
#endif // UTILITIES_CUDA_SHARED_EXPORTS
#define UTILITIES_CUDA_LOCAL UTILITIES_CUDA_DLL_LOCAL
#else  // UTILITIES_CUDA_SHARED is not defined: this means UTILITIES_CUDA is a STATIC lib.
#define UTILITIES_CUDA_API
#define UTILITIES_CUDA_LOCAL
#endif // UTILITIES_CUDA_SHARED



// Which platform we are on?
// UNITY_WIN - Windows (regular win32)
// UNITY_OSX - Mac OS X
// UNITY_LINUX - Linux
// UNITY_IOS - iOS
// UNITY_TVOS - tvOS
// UNITY_ANDROID - Android
// UNITY_METRO - WSA or UWP
// UNITY_WEBGL - WebGL
// UNITY_EMBEDDED_LINUX - EmbeddedLinux OpenGLES
// UNITY_EMBEDDED_LINUX_GL - EmbeddedLinux OpenGLCore
#if _MSC_VER
#define UNITY_WIN 1
#elif defined(__APPLE__)
#if TARGET_OS_TV
#define UNITY_TVOS 1
#elif TARGET_OS_IOS
#define UNITY_IOS 1
#else
#define UNITY_OSX 1
#endif
#elif defined(__ANDROID__)
#define UNITY_ANDROID 1
#elif defined(UNITY_METRO) || defined(UNITY_LINUX) || defined(UNITY_WEBGL) || defined (UNITY_EMBEDDED_LINUX) || defined (UNITY_EMBEDDED_LINUX_GL)
	// these are defined externally
#elif defined(__EMSCRIPTEN__)
	// this is already defined in Unity 5.6
#define UNITY_WEBGL 1
#else
#error "Unknown platform!"
#endif


// Which graphics device APIs we possibly support?
#if UNITY_METRO
#define SUPPORT_D3D11 1
#if WINDOWS_UWP
#define SUPPORT_D3D12 1
#endif
#elif UNITY_WIN
#define SUPPORT_D3D11 1 // comment this out if you don't have D3D11 header/library files
#define SUPPORT_D3D12 1 // comment this out if you don't have D3D12 header/library files
#define SUPPORT_OPENGL_UNIFIED 1
#define SUPPORT_OPENGL_CORE 1
#define SUPPORT_VULKAN 0 // Requires Vulkan SDK to be installed
#elif UNITY_IOS || UNITY_TVOS || UNITY_ANDROID || UNITY_WEBGL
#ifndef SUPPORT_OPENGL_ES
#define SUPPORT_OPENGL_ES 1
#endif
#define SUPPORT_OPENGL_UNIFIED SUPPORT_OPENGL_ES
#ifndef SUPPORT_VULKAN
#define SUPPORT_VULKAN 0
#endif
#elif UNITY_OSX || UNITY_LINUX
#define SUPPORT_OPENGL_UNIFIED 1
#define SUPPORT_OPENGL_CORE 1
#elif UNITY_EMBEDDED_LINUX
#define SUPPORT_OPENGL_UNIFIED 1
#define SUPPORT_OPENGL_ES 1
#ifndef SUPPORT_VULKAN
#define SUPPORT_VULKAN 0
#endif
#endif

#if UNITY_IOS || UNITY_TVOS || UNITY_OSX
#define SUPPORT_METAL 1
#endif



// COM-like Release macro
#ifndef SAFE_RELEASE
#define SAFE_RELEASE(a) if (a) { a->Release(); a = NULL; }
#endif
