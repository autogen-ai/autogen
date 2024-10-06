# Copyright (c) 2023 - 2024, Owners of https://github.com/autogen-ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
#!/usr/bin/env python3 -m pytest

import unittest
from unittest.mock import MagicMock

import pytest
from annotated_types import Annotated
from conftest import MOCK_OPEN_AI_API_KEY

import autogen
from autogen import AssistantAgent, ConversableAgent, UserProxyAgent

try:
    from autogen.agentchat.contrib.img_utils import get_pil_image
    from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
except ImportError:
    skip = True
else:
    skip = False

VISION_MODEL_NAME = "gpt-4-turbo"

base64_encoded_image = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4"
    "//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="
)


if skip:
    pil_image = None
else:
    pil_image = get_pil_image(base64_encoded_image)


@pytest.mark.skipif(skip, reason="dependency is not installed")
class TestMultimodalConversableAgent(unittest.TestCase):
    def setUp(self):
        self.agent = MultimodalConversableAgent(
            name="TestAgent",
            llm_config={
                "timeout": 600,
                "seed": 42,
                "config_list": [{"model": VISION_MODEL_NAME, "api_key": MOCK_OPEN_AI_API_KEY}],
            },
        )

    def test_system_message(self):
        # Test default system message
        self.assertEqual(
            self.agent.system_message,
            [
                {
                    "type": "text",
                    "text": "You are a helpful AI assistant.",
                }
            ],
        )

        # Test updating system message
        new_message = f"We will discuss <img {base64_encoded_image}> in this conversation."
        self.agent.update_system_message(new_message)
        self.assertEqual(
            self.agent.system_message,
            [
                {"type": "text", "text": "We will discuss "},
                {"type": "image_url", "image_url": {"url": pil_image}},
                {"type": "text", "text": " in this conversation."},
            ],
        )

    def test_message_to_dict(self):
        # Test string message
        message_str = "Hello"
        expected_dict = {"content": [{"type": "text", "text": "Hello"}]}
        self.assertDictEqual(self.agent._message_to_dict(message_str), expected_dict)

        # Test list message
        message_list = [{"type": "text", "text": "Hello"}]
        expected_dict = {"content": message_list}
        self.assertDictEqual(self.agent._message_to_dict(message_list), expected_dict)

        # Test dictionary message
        message_dict = {"content": [{"type": "text", "text": "Hello"}]}
        self.assertDictEqual(self.agent._message_to_dict(message_dict), message_dict)

    def test_print_received_message(self):
        sender = ConversableAgent(name="SenderAgent", llm_config=False, code_execution_config=False)
        message_str = "Hello"
        self.agent._print_received_message = MagicMock()  # Mocking print method to avoid actual print
        self.agent._print_received_message(message_str, sender)
        self.agent._print_received_message.assert_called_with(message_str, sender)


@pytest.mark.skipif(skip, reason="Dependency not installed")
def test_group_chat_with_lmm():
    """
    Tests the group chat functionality with two MultimodalConversable Agents.
    Verifies that the chat is correctly limited by the max_round parameter.
    Each agent is set to describe an image in a unique style, but the chat should not exceed the specified max_rounds.
    """

    # Configuration parameters
    max_round = 5
    max_consecutive_auto_reply = 10
    llm_config = False

    # Creating two MultimodalConversable Agents with different descriptive styles
    agent1 = MultimodalConversableAgent(
        name="image-explainer-1",
        max_consecutive_auto_reply=max_consecutive_auto_reply,
        llm_config=llm_config,
        system_message="Your image description is poetic and engaging.",
    )
    agent2 = MultimodalConversableAgent(
        name="image-explainer-2",
        max_consecutive_auto_reply=max_consecutive_auto_reply,
        llm_config=llm_config,
        system_message="Your image description is factual and to the point.",
    )

    # Creating a user proxy agent for initiating the group chat
    user_proxy = autogen.UserProxyAgent(
        name="User_proxy",
        system_message="Ask both image explainer 1 and 2 for their description.",
        human_input_mode="NEVER",  # Options: 'ALWAYS' or 'NEVER'
        max_consecutive_auto_reply=max_consecutive_auto_reply,
        code_execution_config=False,
    )

    # Setting up the group chat
    groupchat = autogen.GroupChat(agents=[agent1, agent2, user_proxy], messages=[], max_round=max_round)
    group_chat_manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Initiating the group chat and observing the number of rounds
    user_proxy.initiate_chat(group_chat_manager, message=f"What do you see? <img {base64_encoded_image}>")

    # Assertions to check if the number of rounds does not exceed max_round
    assert all(len(arr) <= max_round for arr in agent1._oai_messages.values()), "Agent 1 exceeded max rounds"
    assert all(len(arr) <= max_round for arr in agent2._oai_messages.values()), "Agent 2 exceeded max rounds"
    assert all(len(arr) <= max_round for arr in user_proxy._oai_messages.values()), "User proxy exceeded max rounds"


@pytest.mark.skipif(skip, reason="Dependency not installed")
def test_func_call_with_lmm():
    assistant = MultimodalConversableAgent(
        name="Assistant",
        system_message="Describe all the colors in the image.",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=2,
        llm_config={
            "timeout": 600,
            "seed": 42,
            "config_list": [{"model": VISION_MODEL_NAME, "api_key": MOCK_OPEN_AI_API_KEY}],
        },
    )

    coder = AssistantAgent(
        name="Coder",
        system_message="YOU MUST USE THE FUNCTION PROVIDED.",
        llm_config={
            "timeout": 600,
            "seed": 42,
            "config_list": [{"model": VISION_MODEL_NAME, "api_key": MOCK_OPEN_AI_API_KEY}],
        },
        human_input_mode="NEVER",
        code_execution_config=False,
        max_consecutive_auto_reply=2,
    )

    def count_colors(colors: list) -> int:
        return len(colors)

    coder.register_for_llm(name="count_colors", description="Count colors.")(count_colors)
    assistant.register_for_execution(name="count_colors")(count_colors)

    coder.initiate_chat(
        assistant, clear_history=True, message=f"""How many colors here: <img {base64_encoded_image}>"""
    )

    assert len(coder._oai_messages[assistant]) > 1, "Function call did not happen"


if __name__ == "__main__":
    unittest.main()
