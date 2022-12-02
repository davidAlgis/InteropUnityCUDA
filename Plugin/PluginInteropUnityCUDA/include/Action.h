#pragma once
#include <functional>
//contains the macro to export and import in dll
#include "log.h"

class UNITY_INTERFACE_EXPORT Action
{
public:
	Action() = default;
	~Action() = default;
	Action(const Action&) = default;
	Action(Action&&) = default;
	Action& operator=(const Action&) = default;
	Action& operator=(Action&&) = default;

	// The name of the function are just here to make indication about their first purposes 
	// but you can use them in another purposes.
	virtual int Start() = 0;
	virtual int Update() = 0;
	virtual int OnDestroy() = 0;
};