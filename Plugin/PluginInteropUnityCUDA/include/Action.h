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

	virtual bool DoAction(const int time) = 0;
};