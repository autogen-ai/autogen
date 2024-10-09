# Copyright (c) 2023 - 2024, Owners of https://github.com/autogen-ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
from typing import Dict, List, Union

import pytest

from autogen import agentchat

TAG_PARSING_TESTS = [
    {
        "message": "Hello agent, can you take a look at this image <img http://example.com/image.png>",
        "strict_filepath_match": False,
        "expected": [{"tag": "img", "attr": {"src": "http://example.com/image.png"}}],
    },
    {
        "message": "Can you transcribe this audio? <audio http://example.com/au=dio.mp3>",
        "strict_filepath_match": False,
        "expected": [{"tag": "audio", "attr": {"src": "http://example.com/au=dio.mp3"}}],
    },
    {
        "message": "Can you describe what's in this image <img url='http://example.com/=image.png'>",
        "strict_filepath_match": False,
        "expected": [{"tag": "img", "attr": {"url": "http://example.com/=image.png"}}],
    },
    {
        "message": "Can you describe what's in this image <img http://example.com/image.png> and transcribe this audio? <audio http://example.com/audio.mp3>",
        "strict_filepath_match": False,
        "expected": [
            {"tag": "img", "attr": {"src": "http://example.com/image.png"}},
            {"tag": "audio", "attr": {"src": "http://example.com/audio.mp3"}},
        ],
    },
    {
        "message": "Can you generate this audio? <audio text='Hello I'm a robot' prompt='whisper'>",
        "strict_filepath_match": False,
        "expected": [{"tag": "audio", "attr": {"text": "Hello I'm a robot", "prompt": "whisper"}}],
    },
    {
        "message": "Can you describe what's in this image <img http://example.com/image.png width='100'> and this image <img http://hello.com/image=.png>?",
        "strict_filepath_match": False,
        "expected": [
            {"tag": "img", "attr": {"src": "http://example.com/image.png", "width": "100"}},
            {"tag": "img", "attr": {"src": "http://hello.com/image=.png"}},
        ],
    },
    {
        "message": "Text with no tags",
        "strict_filepath_match": False,
        "expected": [],
    },
    {
        "message": 'Can you generate this audio? <audio text="Hello I\'m a robot" prompt="whisper">',
        "strict_filepath_match": False,
        "expected": [{"tag": "audio", "attr": {"text": "Hello I'm a robot", "prompt": "whisper"}}],
    },
    {
        "message": 'Can you generate this audio? <audio text="Hello I\'m a robot" prompt="whisper">',
        "strict_filepath_match": True,
        "expected": [],
    },
    {
        "message": 'Complex nested quotes <img src="http://example.com/image.png" alt="A "quoted" description">',
        "strict_filepath_match": False,
        "expected": [{"tag": "img", "attr": {"src": "http://example.com/image.png", "alt": 'A "quoted" description'}}],
    },
    {
        "message": 'Complex nested quotes <img src="http://example.com/image.png" alt="A "quoted" description">',
        "strict_filepath_match": True,
        "expected": [],
    },
    {
        "message": "Complex nested quotes <img http://example.com/image.png>",
        "strict_filepath_match": True,
        "expected": [{"tag": "img", "attr": {"src": "http://example.com/image.png"}}],
    },
    {
        "message": 'Can you generate this audio? <audio text="Hello I\'m a robot" prompt="whisper">',
        "strict_filepath_match": True,
        "expected": [],  # Empty because strict_filepath_match=True doesn't match tags with spaces or quotes
    },
    {
        "message": 'Complex nested quotes <img src="http://example.com/image.png" alt="A "quoted" description">',
        "strict_filepath_match": True,
        "expected": [],  # Empty because strict_filepath_match=True doesn't match tags with spaces or quotes
    },
]


def _delete_unused_keys(d: Dict) -> None:
    if "match" in d:
        del d["match"]


@pytest.mark.parametrize("test_case", TAG_PARSING_TESTS)
def test_tag_parsing(test_case: Dict[str, Union[str, bool, List[Dict[str, Union[str, Dict[str, str]]]]]]) -> None:
    """Test the strict_filepath_match feature in tag parsing."""
    message = test_case["message"]
    strict_filepath_match = test_case["strict_filepath_match"]
    expected = test_case["expected"]
    tags = ["img", "audio"]

    result = []
    for tag in tags:
        parsed_tags = agentchat.utils.parse_tags_from_content(tag, message, strict_filepath_match=strict_filepath_match)
        for item in parsed_tags:
            _delete_unused_keys(item)
        result.extend(parsed_tags)
    print(result, expected, strict_filepath_match)
    assert result == expected

    result = []
    for tag in tags:
        content = [{"type": "text", "text": message}]
        parsed_tags = agentchat.utils.parse_tags_from_content(tag, content, strict_filepath_match=strict_filepath_match)
        for item in parsed_tags:
            _delete_unused_keys(item)
        result.extend(parsed_tags)
    print(result, expected)
    assert result == expected


if __name__ == "__main__":
    for test_case in TAG_PARSING_TESTS:
        test_tag_parsing(test_case)
