#pragma once
#include "action.h"


Action::Action(int key) : _key{ key } {}

Action::Key Action::GetKey() const
{
	return _key;
}
