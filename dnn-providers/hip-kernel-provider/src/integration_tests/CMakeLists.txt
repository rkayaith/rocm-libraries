# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(Threads REQUIRED)

# hipdnn SDK dependencies - use targets directly if available (superbuild),
# otherwise find installed packages (true standalone)
if(NOT TARGET hipdnn_test_sdk)
    find_package(hipdnn_test_sdk CONFIG REQUIRED)
endif()
if(NOT TARGET hipdnn_plugin_sdk)
    find_package(hipdnn_plugin_sdk CONFIG REQUIRED)
endif()

add_executable(
    hip_kernel_provider_integration_tests
    main.cpp
)

target_compile_options(hip_kernel_provider_integration_tests PRIVATE ${HIPDNN_WARNING_COMPILE_OPTIONS})

# To use the build tree plugin, we use the path relative to the test binary instead of the installed
# libhipdnn_backend.so HIPDNN_PLUGIN_ENGINE_SUBDIR is provided by the sdk
target_compile_definitions(
    hip_kernel_provider_integration_tests
    PRIVATE
        PLUGIN_PATH="../$<IF:$<PLATFORM_ID:Windows>,bin,lib>/${HIPDNN_PLUGIN_ENGINE_SUBDIR}/hip_kernel_provider"
)

target_include_directories(hip_kernel_provider_integration_tests PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
)

target_link_libraries(
    hip_kernel_provider_integration_tests PUBLIC GTest::gtest hip::host hipdnn_test_sdk hipdnn_plugin_sdk Threads::Threads
)

# Ensure that the hip_kernel_plugin is built before the integration tests
add_dependencies(hip_kernel_provider_integration_tests hip_kernel_provider)

set(HIP_KERNEL_PROVIDER_ASM_KERNEL_ENV "")
if(ENABLE_ASM_SDPA_ENGINE)
    if(NOT HIPDNN_ENABLE_SDPA)
        message(WARNING
                "ENABLE_ASM_SDPA_ENGINE is ON but hipDNN was built with HIPDNN_ENABLE_SDPA=OFF. "
                "Skipping SDPA integration tests. Rebuild hipDNN with -DHIPDNN_ENABLE_SDPA=ON to enable them.")
    else()
        target_sources(hip_kernel_provider_integration_tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../tests/engines/asm_sdpa_engine/AsmSdpaConfigHelpers.cpp
            asm_sdpa_engine/IntegrationGpuSdpaForward.cpp
        )
        # Add include path for codegen-generated headers (e.g., asm_fmha_v3_fwd_configs.hpp)
        # and the shared test helpers (AsmSdpaConfigHelpers.hpp)
        target_include_directories(hip_kernel_provider_integration_tests PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR}/../engines/asm_sdpa_engine/generated
            ${CMAKE_CURRENT_SOURCE_DIR}/../tests/engines/asm_sdpa_engine
        )
        # Ensure codegen runs before compiling the integration tests
        add_dependencies(hip_kernel_provider_integration_tests asm_kernel_codegen)
        target_compile_definitions(hip_kernel_provider_integration_tests PRIVATE
            AITER_ASM_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../engines/asm_sdpa_engine/asm/asm_kernels"
        )
        set(HIP_KERNEL_PROVIDER_ASM_KERNEL_ENV
            "HIPDNN_AITER_ASM_DIR=${CMAKE_BINARY_DIR}/hip_kernel_provider/asm_kernels")
    endif()
endif()

if(ENABLE_HIP_MLOPS_ENGINE)
    target_sources(hip_kernel_provider_integration_tests PRIVATE
        hip_mlops_engine/IntegrationGpuPassByValue.cpp
    )
endif()

if(ENABLE_HIP_FLASH2_ENGINE)
    if(NOT HIPDNN_ENABLE_SDPA)
        message(WARNING
                "ENABLE_HIP_FLASH2_ENGINE is ON but hipDNN was built with HIPDNN_ENABLE_SDPA=OFF. "
                "Rebuild hipDNN with -DHIPDNN_ENABLE_SDPA=ON to enable Flash2 integration tests.")
    else()
        add_subdirectory(hip_flash2_engine)
    endif()
endif()
if(HIPDNN_ENABLE_KERNEL_INGESTOR)
    target_sources(hip_kernel_provider_integration_tests PRIVATE
        kernel_ingestor_engine/IntegrationGpuKernelIngestor.cpp
        kernel_ingestor_engine/IntegrationGpuKernelIngestorKpack.cpp
        kernel_ingestor_engine/IntegrationGpuRockeAttentionDense.cpp
    )
    # This target links only the SDKs, not hip_kernel_provider_private, so it does not
    # inherit the provider's gate and needs the definition set directly.
    target_compile_definitions(hip_kernel_provider_integration_tests
        PRIVATE HIPDNN_ENABLE_KERNEL_INGESTOR=1
    )
    # The relative layout only, never an absolute build path: the kpack suite locates the
    # staged tree by walking from the loaded plugin exactly as the loader does, and needs
    # the same spelling of the subdirectory the provider compiled in. An absolute path
    # here would be baked into the installed test binary as well, naming a build tree it
    # cannot see.
    target_compile_definitions(hip_kernel_provider_integration_tests
        PRIVATE HIPDNN_PACKAGED_FIXTURE_SUBDIR="${HIPDNN_DESCRIPTOR_SUBDIR}"
    )
    # Guarded rather than assumed: a root skips with a warning when hipcc is absent, and
    # the suite is written to GTEST_SKIP in exactly that case. A hard dependency would
    # turn a missing compiler into a configure error. hkp_descriptor_staging is the one
    # target that owns the staged tree, whichever roots contributed to it.
    if(TARGET hkp_descriptor_staging)
        add_dependencies(hip_kernel_provider_integration_tests hkp_descriptor_staging)
    endif()
endif()

clang_tidy_check(hip_kernel_provider_integration_tests)

# ENVIRONMENT is forwarded to add_integration_test_target() rather than set
# afterward via set_tests_properties(): when YAML-driven CTest
# categorization is active (DNN_PROVIDER_CTEST_CATEGORIES_YAML), the raw
# hip_kernel_provider_integration_tests CTest test is never registered --
# only the tiered suites created later by apply_test_category_labels() in
# ../CMakeLists.txt are, and set_tests_properties() on the unregistered
# name is a hard CMake configure error. add_integration_test_target()
# publishes the merged environment as
# hip_kernel_provider_integration_tests_TEST_ENVIRONMENT so that later call
# can forward it to whichever suites actually exist.
#
# No descriptor entry: this suite loads the real plugin, and the engine resolves its
# descriptor tree relative to the loaded module -- the staged copy in a build tree, the
# installed copy in an install tree. Naming one explicitly would pin the build path into
# the install-time CTest file and stop the installed descriptors from ever being read.
add_integration_test_target(hip_kernel_provider_integration_tests ${CMAKE_CURRENT_BINARY_DIR}
    LABELS slow
    ENVIRONMENT ${HIP_KERNEL_PROVIDER_ASM_KERNEL_ENV})

# Forward the merged raw-test environment (published by add_integration_test_target)
# to the parent scope so the tiered CTest suites created in ../CMakeLists.txt can
# pick it up. Setting ENVIRONMENT directly on the raw test name here is a hard
# configure error when YAML categorization is active, hence the forwarding.
if(DEFINED hip_kernel_provider_integration_tests_TEST_ENVIRONMENT)
    set(hip_kernel_provider_integration_tests_TEST_ENVIRONMENT
        "${hip_kernel_provider_integration_tests_TEST_ENVIRONMENT}" PARENT_SCOPE)
endif()

# Register the cross-provider ASM SDPA gpu-reference integration test target. In
# standalone provider builds the harness may not already be in scope, so try to
# import it first.
if(NOT COMMAND add_external_integration_test_target)
    find_package(hipdnn_integration_tests CONFIG QUIET)
endif()

if(ENABLE_ASM_SDPA_ENGINE AND HIPDNN_ENABLE_SDPA
   AND TARGET hipdnn_integration_tests AND COMMAND add_external_integration_test_target)
    set(_synthetic_root "/__hipdnn_install_root__")
    set(_install_cwd_abs "${_synthetic_root}/${CMAKE_INSTALL_BINDIR}/hip_kernel_provider")
    set(_install_asm_abs
        "${_synthetic_root}/${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/hip_kernel_provider/asm_kernels"
    )
    file(RELATIVE_PATH _install_asm_dir "${_install_cwd_abs}" "${_install_asm_abs}")

    add_external_integration_test_target(
        TARGET_NAME hip_kernel_provider_asm_sdpa_gpu_ref_integration_tests
        PLUGIN_TARGET hip_kernel_provider
        ENGINE_NAME ASM_SDPA_ENGINE
        INSTALL_SUBDIR hip_kernel_provider
        REFERENCE_EXECUTOR gpu
        GTEST_FILTER
            Smoke/IntegrationGpuSdpaFwdBfp16.Correctness/*
        ENVIRONMENT
            ${HIP_KERNEL_PROVIDER_ASM_KERNEL_ENV}
        INSTALL_ENVIRONMENT
            "HIPDNN_AITER_ASM_DIR=${_install_asm_dir}"
    )
endif()

# The rocKE attention_dense packs, through the cross-provider gpu-reference suite.
#
# Selected by ENGINE_NAME, which only became possible once hip-kernel-provider implemented
# Container::getEngineName (engine plugin API 1.4.0, added by #10645 and adopted here).
# Before that a descriptor-backed engine published no name at all, --test-engine rejected
# it as "not loaded", and this block could not exist.
#
# One target per pack because --test-engine pins exactly one engine: a case belonging to
# the other pack reports not-applicable and skips, which each TOML documents.
#
# Gated on the packs actually being built. The descriptors only exist when the ingestor is
# on AND the rocKE producer ran over a source root; without that the engines never appear
# and every case would fail rather than skip.
if(HIPDNN_ENABLE_KERNEL_INGESTOR AND HIPDNN_ENABLE_SDPA
   AND HIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE
   AND TARGET hipdnn_integration_tests AND COMMAND add_external_integration_test_target)
    # gpu: these graphs carry no golden data in-tree, so the reference is computed live.
    # GpuSdpaFwdPlanBuilder accepts them (no dropout, paging, varlen, stats or fp8 descale)
    # and reads strides from the tensor rather than assuming a layout -- which is what
    # makes the BSHD cases checkable at all.
    add_external_integration_test_target(
        TARGET_NAME hip_kernel_provider_rocke_attention_dense_gpu_ref_integration_tests
        PLUGIN_TARGET hip_kernel_provider
        ENGINE_NAME hipkernel:AttentionDense
        INSTALL_SUBDIR hip_kernel_provider
        REFERENCE_EXECUTOR gpu
        GTEST_FILTER
            Smoke/IntegrationGpuSdpaRockeAttentionDenseBfp16.Correctness/*
    )

    add_external_integration_test_target(
        TARGET_NAME hip_kernel_provider_rocke_attention_dense_autotune_gpu_ref_integration_tests
        PLUGIN_TARGET hip_kernel_provider
        ENGINE_NAME hipkernel:AttentionDenseAutotune
        INSTALL_SUBDIR hip_kernel_provider
        REFERENCE_EXECUTOR gpu
        GTEST_FILTER
            Smoke/IntegrationGpuSdpaRockeAttentionDenseBfp16.Correctness/*
    )
endif()
