#pragma once
#include <functional>

class Data;

class Action
{
public:
	using Key = int;
	Action() = default;
	~Action() = default;
	Action(const Action&) = default;
	Action(Action&&) = default;
	Action& operator=(const Action&) = default;
	Action& operator=(Action&&) = default;

	virtual Key GetKey() const = 0;
	virtual bool DoAction(const Data& data) = 0;
};