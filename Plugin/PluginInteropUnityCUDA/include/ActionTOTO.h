#ifndef MaSuperDLL_ACTIONTOTO_H
#define MaSuperDLL_ACTIONTOTO_H

#include "Action.h"
#include "framework.h"
#include "texture.h"

namespace MaSuperDLL {

  class ActionToto : public Action {
   public:
    inline ActionToto(int key);
    ActionToto(const ActionToto &) = default;
    ActionToto(ActionToto &&) = default;

    // TODO: verify this
    ~ActionToto() = default;
    ActionToto &operator=(const ActionToto &) = default;
    ActionToto &operator=(ActionToto &&) = default;

    inline Key GetKey() const override;
    inline bool DoAction(const int time) override;

    inline void setTexture(Texture *const texture);

   private:
    int m_key;
    Texture *m_texture;
  };

} // namespace MaSuperDLL

extern "C" {

UNITY_INTERFACE_EXPORT MaSuperDLL::ActionToto *UNITY_INTERFACE_API createActionToto(int key);

UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API setTextureActionToto(MaSuperDLL::ActionToto *, Texture*);

}

#include "ActionTOTO.hpp"

#endif
