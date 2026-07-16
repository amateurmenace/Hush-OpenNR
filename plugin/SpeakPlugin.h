// Speak — free film-reconstruction OpenFX plugin for DaVinci Resolve.
// The film counterpart to Hush (see OpenNRPlugin.h): the LAST node in a grade.
// Registered as a SECOND plugin (org.opennr.Speak) in the same .ofx bundle.
// MIT License.

#ifndef OPENNR_SPEAKPLUGIN_H
#define OPENNR_SPEAKPLUGIN_H

#include "ofxsImageEffect.h"

class SpeakPluginFactory : public OFX::PluginFactoryHelper<SpeakPluginFactory>
{
public:
    SpeakPluginFactory();
    virtual void load() {}
    virtual void unload() {}
    virtual void describe(OFX::ImageEffectDescriptor& p_Desc);
    virtual void describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum p_Context);
    virtual OFX::ImageEffect* createInstance(OfxImageEffectHandle p_Handle, OFX::ContextEnum p_Context);
};

// Registered from OpenNRPlugin.cpp's getPluginIDs (one bundle, two plugins).
namespace speakofx { void registerSpeak(OFX::PluginFactoryArray& p_FactoryArray); }

#endif
