#include "ActionTOTO.h"

#include "Action.h"

#include <memory>

extern "C" {

UNITY_INTERFACE_EXPORT MaSuperDLL::ActionToto *UNITY_INTERFACE_API createActionToto(int key)
{
  using namespace MaSuperDLL;
  return (new ActionToto(key));
}

UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API
setTextureActionToto(MaSuperDLL::ActionToto *actionToto, Texture *texture)
{
  actionToto->setTexture(texture);
}

} // extern "C"
