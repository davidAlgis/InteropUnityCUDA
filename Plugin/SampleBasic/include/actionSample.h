#pragma once
#include "log.h"
#include "action.h"
#include "framework.h"
#include "texture.h"

namespace SampleBasic
{
    class ActionSample : public Action 
    {
    public:
        inline ActionSample(int key);
        ActionSample(const ActionSample&) = default;
        ActionSample(ActionSample&&) = default;

        // TODO: verify this
        ~ActionSample() = default;
        ActionSample& operator=(const ActionSample&) = default;
        ActionSample& operator=(ActionSample&&) = default;

        inline Key GetKey() const override;
        inline bool DoAction(const int time) override;

        inline void setTexture(Texture* const texture);

    private:
        int m_key;
        Texture* m_texture;
    };
}


extern "C" 
{

    SampleBasic::ActionSample* UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API createActionToto(int key);

    void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API setTextureActionToto(MaSuperDLL::ActionToto*, Texture*);

}

