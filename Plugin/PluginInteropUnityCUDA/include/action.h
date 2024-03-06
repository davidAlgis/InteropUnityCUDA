#pragma once
#include "IUnityGraphics.h"
#include <functional>

/**
 * @class      Action
 *
 * @brief      Base class to derive from if you want to execute some function on
 * graphics object
 *
 *
 * @example action_sample_texture_array.cpp shows an example of use of this
 * class with a texture array.
 */
class UNITY_INTERFACE_EXPORT Action
{
    public:
    Action() = default;
    ~Action() = default;
    Action(const Action &) = default;
    Action(Action &&) = default;
    Action &operator=(const Action &) = default;
    Action &operator=(Action &&) = default;

    // The name of the function are just here to make indication about their
    // first purposes but you can use them in another purposes.
    virtual int Start() = 0;
    virtual int Update() = 0;
    virtual int OnDestroy() = 0;
    bool IsActive = true;
};