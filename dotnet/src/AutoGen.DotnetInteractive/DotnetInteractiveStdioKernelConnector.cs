﻿// Modifications made in https://github.com/autogen-ai/autogen since August 25, 2024, are licensed under the Apache License, Version 2.0 (Apache-2.0).
// Copyright (c) 2023 - 2024, Owners of https://github.com/autogen-ai
// SPDX-License-Identifier: Apache-2.0
// Original portions of this file are derived from https://github.com/microsoft/autogen under the MIT License, before release v0.2.35.
// SPDX-License-Identifier: MIT
// Copyright (c) Microsoft Corporation. All rights reserved.
// DotnetInteractiveStdioKernelConnector.cs

using AutoGen.DotnetInteractive.Extension;
using Microsoft.DotNet.Interactive;
using Microsoft.DotNet.Interactive.Commands;
using Microsoft.DotNet.Interactive.Connection;

namespace AutoGen.DotnetInteractive;

public class DotnetInteractiveStdioKernelConnector
{
    private string workingDirectory;
    private InteractiveService interactiveService;
    private string kernelName;
    private List<SubmitCode> setupCommands = new List<SubmitCode>();

    internal DotnetInteractiveStdioKernelConnector(string workingDirectory, string kernelName = "root-proxy")
    {
        this.workingDirectory = workingDirectory;
        this.interactiveService = new InteractiveService(workingDirectory);
        this.kernelName = kernelName;
    }

    public DotnetInteractiveStdioKernelConnector RestoreDotnetInteractive()
    {
        if (this.interactiveService.RestoreDotnetInteractive())
        {
            return this;
        }
        else
        {
            throw new Exception("Failed to restore dotnet interactive tool.");
        }
    }

    public DotnetInteractiveStdioKernelConnector AddPythonKernel(
        string venv,
        string kernelName = "python")
    {
        var magicCommand = $"#!connect jupyter --kernel-name {kernelName} --kernel-spec {venv}";
        var connectCommand = new SubmitCode(magicCommand);

        this.setupCommands.Add(connectCommand);

        return this;
    }

    public async Task<Kernel> BuildAsync(CancellationToken ct = default)
    {
        var compositeKernel = new CompositeKernel();
        var url = KernelHost.CreateHostUri(this.kernelName);
        var cmd = new string[]
            {
                    "dotnet",
                    "tool",
                    "run",
                    "dotnet-interactive",
                    $"[cb-{this.kernelName}]",
                    "stdio",
                    //"--default-kernel",
                    //"csharp",
                    "--working-dir",
                    $@"""{workingDirectory}""",
            };

        var connector = new StdIoKernelConnector(
            cmd,
            this.kernelName,
            url,
            new DirectoryInfo(this.workingDirectory));

        var rootProxyKernel = await connector.CreateRootProxyKernelAsync();

        rootProxyKernel.KernelInfo.SupportedKernelCommands.Add(new(nameof(SubmitCode)));

        var dotnetKernel = await connector.CreateProxyKernelAsync(".NET");
        foreach (var setupCommand in this.setupCommands)
        {
            var setupCommandResult = await rootProxyKernel.SendAsync(setupCommand, ct);
            setupCommandResult.ThrowOnCommandFailed();
        }

        return rootProxyKernel;
    }
}
