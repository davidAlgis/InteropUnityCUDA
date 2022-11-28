#pragma once
#include <functional>
//contains the macro to export and import in dll
#include "log.h"

class UNITY_INTERFACE_EXPORT Action
{
public:
	using Key = int;
	Action(int key);
	~Action() = default;
	Action(const Action&) = default;
	Action(Action&&) = default;
	Action& operator=(const Action&) = default;
	Action& operator=(Action&&) = default;
	Key GetKey() const;
	
	virtual bool DoAction(const int time) = 0;

	private:
		Key _key;
};