﻿// Copyright (c) 2023 - 2024, Owners of https://github.com/autogen-ai
// SPDX-License-Identifier: Apache-2.0
// Contributions to this project, i.e., https://github.com/autogen-ai/autogen, 
// are licensed under the Apache License, Version 2.0 (Apache-2.0).
// Portions derived from  https://github.com/microsoft/autogen under the MIT License, before release v0.2.35.
// SPDX-License-Identifier: MIT
// Copyright (c) Microsoft Corporation. All rights reserved.
// FunctionContractExtensionTests.cs

using ApprovalTests;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using AutoGen.Gemini.Extension;
using Google.Protobuf;
using Xunit;

namespace AutoGen.Gemini.Tests;

public class FunctionContractExtensionTests
{
    private readonly Functions functions = new Functions();
    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("ApprovalTests")]
    public void ItGenerateGetWeatherToolTest()
    {
        var contract = functions.GetWeatherAsyncFunctionContract;
        var tool = contract.ToFunctionDeclaration();
        var formatter = new JsonFormatter(JsonFormatter.Settings.Default.WithIndentation("  "));
        var json = formatter.Format(tool);
        Approvals.Verify(json);
    }
}
