// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <functional>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include "Handle.hpp"
#include "compilation/IKernelCompiler.hpp"
#include "device/IDevicePropertyProvider.hpp"

namespace hip_kernel_provider::core
{

/*
 * Container class to manage the instantiation and ownership of all HIP kernel plan builders
 * and engines. The class designs use dependency injection to get the components they need
 * in order to function.
 *
 * The construction sequence should contain no logic other than the creation of various classes.
 * If logic is needed, it should be placed in a separate function that can be called after the
 * container has finished constructing all its components.
 */
class Container
{
public:
    Container();
    ~Container();

    // Copy engine IDs into a buffer.
    // If maxEngines == 0: Does not copy, only queries total count.
    // If maxEngines > 0: Copies up to maxEngines IDs into *engineIds, sets numEngines to number
    // copied. Returns: Total number of available engines (regardless of maxEngines value).
    static uint32_t copyEngineIds(int64_t* engineIds, uint32_t maxEngines, uint32_t& numEngines);

    /// Names this plugin's engines for the host (engine plugin API 1.4.0, optional --
    /// detected on the Container by SFINAE and reached through
    /// hipdnnEnginePluginGetEngineName).
    ///
    /// Without it, descriptor-backed engines have no name anywhere in the public API: the
    /// host's built-in registry only knows the static HIPDNN_REGISTER_ENGINE list, so a
    /// discovered engine falls back to a hex rendering of its id and cannot be selected by
    /// name (`--test-engine`, tooling, logs).
    ///
    /// Two contract points the host enforces at load, both satisfied by construction here:
    /// the returned pointer must outlive the loaded library -- it aliases a `std::string`
    /// in the process-lifetime static that discoverDescriptorSets() memoizes -- and
    /// `engineNameToId(name)` must equal `engine_id`, which holds because the id was
    /// derived by hashing exactly this string. A mismatch costs the engine its place in
    /// enumeration, routing and dispatch.
    ///
    /// Returns NOT_APPLICABLE for an unrecognised id, which is the only penalty-free
    /// decline; the statically registered engines (HIP_MLOPS, ASM_SDPA) take that path and
    /// are named by the host's registry instead.
    static hipdnnPluginStatus_t getEngineName(int64_t engineId, const char** name);

    hipdnn_plugin_sdk::EngineManager<Handle, Settings, Context>& getEngineManager();

private:
    struct EngineDefinition
    {
        using Factory
            = std::function<std::unique_ptr<hipdnn_plugin_sdk::IEngine<Handle, Settings, Context>>(
                const device::IDevicePropertyProvider&)>;

        /// Statically registered engine: the host's own EngineNames.hpp registry names it,
        /// so this carries none and getEngineName() answers NOT_APPLICABLE for it.
        EngineDefinition(int64_t engineId, Factory factory)
            : id(engineId)
            , createEngine(std::move(factory))
        {
        }

        /// Descriptor-backed engine: @p engineName must be the UED name @p engineId was
        /// hashed from, and must outlive the loaded library. Both hold when it aliases the
        /// memoized DescriptorSet -- see the call site.
        EngineDefinition(int64_t engineId, std::string_view engineName, Factory factory)
            : id(engineId)
            , name(engineName)
            , createEngine(std::move(factory))
        {
        }

        int64_t id;
        /// Empty for a statically registered engine.
        std::string_view name;
        Factory createEngine;
    };

    static const std::vector<EngineDefinition>& getEngineDefinitions();

    std::unique_ptr<device::IDevicePropertyProvider> _devicePropertyProvider;
    std::unique_ptr<compilation::IKernelCompiler> _kernelCompiler;
    std::unique_ptr<hipdnn_plugin_sdk::EngineManager<Handle, Settings, Context>> _engineManager;
};

} // namespace hip_kernel_provider::core
