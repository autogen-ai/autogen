﻿// Copyright (c) 2023 - 2024, Owners of https://github.com/autogen-ai
// SPDX-License-Identifier: Apache-2.0
// Contributions to this project, i.e., https://github.com/autogen-ai/autogen, 
// are licensed under the Apache License, Version 2.0 (Apache-2.0).
// Portions derived from  https://github.com/microsoft/autogen under the MIT License, before release v0.2.35.
// SPDX-License-Identifier: MIT
// Copyright (c) Microsoft Corporation. All rights reserved.
// MiddlewareContext.cs

using System.Collections.Generic;

namespace AutoGen.Core;

public class MiddlewareContext
{
    public MiddlewareContext(
        IEnumerable<IMessage> messages,
        GenerateReplyOptions? options)
    {
        this.Messages = messages;
        this.Options = options;
    }

    /// <summary>
    /// Messages to send to the agent
    /// </summary>
    public IEnumerable<IMessage> Messages { get; }

    /// <summary>
    /// Options to generate the reply
    /// </summary>
    public GenerateReplyOptions? Options { get; }
}
