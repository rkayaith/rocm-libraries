// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Container.hpp"
#include "device/CurrentDevicePropertyProvider.hpp"

#ifdef HIPDNN_ENGINE_HIP_MLOPS
#include "engines/hip_mlops_engine/HipMlopsEngine.hpp"
#include "engines/hip_mlops_engine/plans/RMSnorm/RMSnormBwdPlanBuilder.hpp"
#include "engines/hip_mlops_engine/plans/RMSnorm/RMSnormPlanBuilder.hpp"
#include "engines/hip_mlops_engine/plans/batchnorm/BatchnormFwdTrainingPlanBuilder.hpp"
#include "engines/hip_mlops_engine/plans/batchnorm/BatchnormPlanBuilder.hpp"
#include "engines/hip_mlops_engine/plans/layernorm/LayernormPlanBuilder.hpp"
#include "engines/hip_mlops_engine/plans/resample/ResamplePlanBuilder.hpp"
#endif

#ifdef HIPDNN_ENGINE_HIP_FLASH2
#include "engines/hip_flash2_engine/HipFlash2Engine.hpp"
#include "engines/hip_flash2_engine/HipFlash2FwdPlanBuilder_v2.hpp"
#endif

#ifdef HIPDNN_ENGINE_ASM_SDPA
#include "engines/asm_sdpa_engine/AsmSdpaEngine.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaBwdPlanBuilder.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaFwdPlanBuilder.hpp"
#endif

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#include <filesystem>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/PluginApi.h>
#include <hipdnn_plugin_sdk/ingestor/MakeEngine.hpp>

#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#endif

#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace hip_kernel_provider::core
{

using namespace hipdnn_data_sdk::utilities;

const std::vector<Container::EngineDefinition>& Container::getEngineDefinitions()
{
    static const std::vector<EngineDefinition> s_engineDefinitions = [] {
        std::vector<EngineDefinition> definitions = {
        // HIP_MLOPS_ENGINE
#ifdef HIPDNN_ENGINE_HIP_MLOPS
            {HIP_MLOPS_ENGINE_ID,
             [](const device::IDevicePropertyProvider& devicePropertyProvider)
                 -> std::unique_ptr<hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>> {
                 auto engine = std::make_unique<HipMlopsEngine>(HIP_MLOPS_ENGINE_ID);
                 const compilation::IKernelCompiler& kernelCompiler = engine->getKernelCompiler();
                 engine->addPlanBuilder(std::make_unique<batchnorm::BatchnormPlanBuilder>(
                     kernelCompiler, devicePropertyProvider));
                 engine->addPlanBuilder(
                     std::make_unique<batchnorm::BatchnormFwdTrainingPlanBuilder>(
                         kernelCompiler, devicePropertyProvider));
                 engine->addPlanBuilder(std::make_unique<rmsnorm::RMSnormPlanBuilder>(
                     kernelCompiler, devicePropertyProvider));
                 engine->addPlanBuilder(std::make_unique<rmsnorm::RMSnormBwdPlanBuilder>(
                     kernelCompiler, devicePropertyProvider));
                 engine->addPlanBuilder(std::make_unique<layernorm::LayernormPlanBuilder>(
                     kernelCompiler, devicePropertyProvider));
                 engine->addPlanBuilder(std::make_unique<resample::ResamplePlanBuilder>(
                     kernelCompiler, devicePropertyProvider));
                 return engine;
             }},
#endif
#ifdef HIPDNN_ENGINE_HIP_FLASH2
            // HIP_FLASH2_ENGINE: FP16 Flash-Attention 2 V7 (rocWMMA MFMA + causal tile skip)
            // Complements ASM_SDPA_ENGINE: handles FP16 on gfx942/gfx950.
            // Performance: 78.98 TFLOPS MI325X, 71.27 TFLOPS MI300X (seq=4096 causal D=128).
            {HIP_FLASH2_ENGINE_ID,
             [](const device::IDevicePropertyProvider& /*devicePropertyProvider*/)
                 -> std::unique_ptr<hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>> {
                 auto engine = std::make_unique<hip_flash2_engine::HipFlash2Engine>();
                 engine->addPlanBuilder(
                     std::make_unique<hip_flash2_engine::HipFlash2FwdPlanBuilder>());
                 return engine;
             }},
#endif // HIPDNN_ENGINE_HIP_FLASH2
#ifdef HIPDNN_ENGINE_ASM_SDPA
            // ASM_SDPA_ENGINE
            {ASM_SDPA_ENGINE_ID,
             [](const device::IDevicePropertyProvider& /*devicePropertyProvider*/)
                 -> std::unique_ptr<hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>> {
                 auto engine = std::make_unique<asm_sdpa_engine::AsmSdpaEngine>();
                 engine->addPlanBuilder(std::make_unique<asm_sdpa_engine::SdpaFwdPlanBuilder>());
                 engine->addPlanBuilder(std::make_unique<asm_sdpa_engine::SdpaBwdPlanBuilder>());
                 return engine;
             }},
#endif
        };

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
        // One ingestor engine per discovered descriptor set, and a set is now a file on
        // disk: adding an engine is an install, not an edit here.
        for(const auto& set : kernel_ingestor_engine::discoverDescriptorSets())
        {
            // engineNameToId is a pure FNV-1a hash of the UED name, so the id and the name
            // agree by construction -- which is exactly the invariant
            // hipdnnEnginePluginGetEngineName's contract requires the host to be able to
            // re-verify (engineNameToId(name) == engine_id, checked at load; a mismatch
            // drops the engine).
            //
            // The name is recorded here rather than registered into EngineNames.hpp's map:
            // that map is a static in-tree registry a plugin cannot reach, and writing a
            // string_view into it from here would risk a dangling view. getEngineName()
            // below serves it instead, which is the mechanism the host added for this in
            // engine plugin API 1.4.0.
            const auto engineId = engineNameToId(set.engine.name);
            definitions.push_back(
                {engineId,
                 // Aliases the memoized set's std::string, so it is valid for the process
                 // -- what the host requires of the pointer getEngineName() hands back.
                 set.engine.name,
                 // set aliases discoverDescriptorSets()'s memoized, process-lifetime vector.
                 // Capture by reference: [set] would re-copy a DescriptorSet per engine.
                 [&set](const device::IDevicePropertyProvider& /*devicePropertyProvider*/)
                     -> std::unique_ptr<hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>> {
                     try
                     {
                         // Device facts are resolved per call from the handle, not from
                         // the construction-time provider.
                         return hipdnn_plugin_sdk::ingestor::makeEngine<Handle, Settings, Context>(
                             set, kernel_ingestor_engine::deviceResolver());
                     }
                     catch(const std::exception& error)
                     {
                         // The loader validates each set, but its probe and this construction
                         // are different objects, so that's convention, not a guarantee.
                         // Return null: throwing here would cost HIP_MLOPS and ASM_SDPA too.
                         HIPDNN_PLUGIN_LOG_ERROR("ingestor: engine '"
                                                 << set.engine.name
                                                 << "' failed to construct and is excluded: "
                                                 << error.what());
                         return nullptr;
                     }
                 }});
        }
#endif

        return definitions;
    }();

    return s_engineDefinitions;
}

uint32_t Container::copyEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t& numEngines)
{
    const auto& engineDefinitions = getEngineDefinitions();
    auto totalEngines = static_cast<uint32_t>(engineDefinitions.size());

    if(maxEngines == 0)
    {
        numEngines = totalEngines;
        return totalEngines;
    }

    auto enginesToCopy = std::min(maxEngines, totalEngines);
    for(uint32_t i = 0; i < enginesToCopy; ++i)
    {
        engineIds[i] = engineDefinitions[i].id;
    }

    numEngines = enginesToCopy;

    return totalEngines;
}

hipdnnPluginStatus_t Container::getEngineName(int64_t engineId, const char** name)
{
    // The host never passes null, so this is a defect path rather than a decline -- and
    // per the contract it costs the engine its place, so it must not be used for "no name".
    if(name == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }

    for(const auto& definition : getEngineDefinitions())
    {
        if(definition.id != engineId)
        {
            continue;
        }

        // A statically registered engine carries no name here on purpose: the host's own
        // EngineNames.hpp registry already names it, and NOT_APPLICABLE is the documented
        // way to defer to it.
        if(definition.name.empty())
        {
            return HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE;
        }

        // .data() is NUL-terminated here because the view aliases a std::string owned by
        // the process-lifetime static behind discoverDescriptorSets(). That is also what
        // satisfies "must remain valid for the lifetime of the loaded library".
        *name = definition.name.data();
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }

    // An id this plugin does not recognise. NOT_APPLICABLE is the only penalty-free
    // decline; any other non-success status would drop the engine.
    return HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE;
}

Container::Container()
    : _devicePropertyProvider(std::make_unique<device::CurrentDevicePropertyProvider>())
{
    HIPDNN_PLUGIN_LOG_INFO("Creating Container");

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
    // Must run before any descriptor-backed engine below can resolve its UMD/UHD/UDD
    // symbols. Safe on every Container construction: registers exactly once per process
    // (see SharedContainerManager).
    kernel_ingestor_engine::registerNativeIngestorSymbols();
#endif

    _engineManager
        = std::make_unique<hipdnn_plugin_sdk::EngineManager<Handle, Settings, Context>>();

    for(const auto& engineDefinition : getEngineDefinitions())
    {
        // Null only when a descriptor-backed engine failed to construct (already logged).
        // Its id stays advertised but never claims a graph -- indistinguishable from an
        // engine that declines everything.
        if(auto engine = engineDefinition.createEngine(*_devicePropertyProvider))
        {
            _engineManager->addEngine(std::move(engine));
        }
    }
}

Container::~Container()
{
    HIPDNN_PLUGIN_LOG_INFO("Destroying Container");
}

hipdnn_plugin_sdk::EngineManager<Handle, Settings, Context>& Container::getEngineManager()
{
    return *_engineManager;
}

} // namespace hip_kernel_provider::core
