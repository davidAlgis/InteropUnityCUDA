#ifndef MaSuperDLL_ACTIONTOTO_HPP
#define MaSuperDLL_ACTIONTOTO_HPP

#include "Action.h"
#include "ActionTOTO.h"

namespace MaSuperDLL {

  ActionToto::ActionToto(int key) : m_key{key} {}

  Action::Key ActionToto::GetKey() const
  {
    return m_key;
  }

  bool ActionToto::DoAction(const int time)
  {
    constexpr auto epsilon = 1e6;

    const auto log10_10 = log10(10.0);
    const auto one = 1.0 + 0.0 * time;

    return fabs(log10_10 - one) < epsilon;
  }

  void ActionToto::setTexture(Texture *const texture)
  {
    m_texture = texture;
  }

} // namespace MaSuperDLL

#endif