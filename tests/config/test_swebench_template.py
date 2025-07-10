from dataclasses import dataclass
from pathlib import Path

import yaml
from jinja2 import Template


@dataclass
class MockOutput:
    """Mock output object for testing the template"""

    returncode: int
    stderr: str
    stdout: str


def test_action_observation_template_short_output():
    """Test that short output (< 10000 chars) is displayed in full"""
    # Load the swebench config
    config_path = Path(__file__).parent.parent.parent / "src" / "microsweagent" / "config" / "extra" / "swebench.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract the template
    template_str = config["agent"]["action_observation_template"]
    template = Template(template_str)

    # Create mock output with short content
    output = MockOutput(returncode=0, stderr="Warning: minor issue", stdout="Success! Operation completed.")

    # Render the template
    result = template.render(output=output)

    # Verify the result contains all parts and no truncation
    assert "<returncode>" in result
    assert "0" in result
    assert "<stderr>" in result
    assert "Warning: minor issue" in result
    assert "<stdout>" in result
    assert "Success! Operation completed." in result

    # Should not contain truncation elements for short output
    assert "<observation_head>" not in result
    assert "<elided_chars>" not in result
    assert "<observation_tail>" not in result
    assert "<warning>" not in result


def test_action_observation_template_long_output():
    """Test that long output (> 10000 chars) is truncated with head/tail format"""
    # Load the swebench config
    config_path = Path(__file__).parent.parent.parent / "src" / "microsweagent" / "config" / "extra" / "swebench.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract the template
    template_str = config["agent"]["action_observation_template"]
    template = Template(template_str)

    # Create mock output with long content
    long_stdout = "A" * 8000  # 8000 characters
    long_stderr = "B" * 3000  # 3000 characters
    # Total will be > 10000 chars when combined with XML tags

    output = MockOutput(returncode=1, stderr=long_stderr, stdout=long_stdout)

    # Render the template
    result = template.render(output=output)

    # Should contain truncation elements for long output
    assert "<warning>" in result
    assert "The output of your last command was too long" in result
    assert "<observation_head>" in result
    assert "<elided_chars>" in result
    assert "characters elided" in result
    assert "<observation_tail>" in result

    # Should still contain the basic structure
    assert "<returncode>" in result
    assert "1" in result
    assert "<stderr>" in result
    assert "<stdout>" in result

    # Verify the head contains first part of output
    head_start = result.find("<observation_head>")
    head_end = result.find("</observation_head>")
    head_content = result[head_start:head_end]
    assert "AAAA" in head_content  # Should contain start of stdout

    # Verify the tail contains last part of output (which would be end of stdout)
    tail_start = result.find("<observation_tail>")
    tail_end = result.find("</observation_tail>")
    tail_content = result[tail_start:tail_end]
    assert "AAAA" in tail_content  # Should contain end of stdout (last part of full output)


def test_action_observation_template_edge_case_exactly_10000_chars():
    """Test the boundary case where output is around 10000 characters"""
    # Load the swebench config
    config_path = Path(__file__).parent.parent.parent / "src" / "microsweagent" / "config" / "extra" / "swebench.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract the template
    template_str = config["agent"]["action_observation_template"]
    template = Template(template_str)

    # Use a large amount of data that will definitely exceed 10000 chars when rendered
    output = MockOutput(returncode=0, stderr="", stdout="X" * 10000)

    # Render the template
    result = template.render(output=output)

    # Should use truncated format for large output
    assert "<observation_head>" in result
    assert "<elided_chars>" in result
    assert "<observation_tail>" in result
    assert "<warning>" in result
    # The X's should still be present in head or tail
    assert "XXXX" in result


def test_action_observation_template_just_under_10000_chars():
    """Test that smaller output shows full output without truncation"""
    # Load the swebench config
    config_path = Path(__file__).parent.parent.parent / "src" / "microsweagent" / "config" / "extra" / "swebench.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract the template
    template_str = config["agent"]["action_observation_template"]
    template = Template(template_str)

    # Use a reasonably sized output that should be well under 10000 chars when rendered
    output = MockOutput(returncode=0, stderr="", stdout="Y" * 8000)

    # Render the template
    result = template.render(output=output)

    # Should show full output without truncation
    assert "<observation_head>" not in result
    assert "<elided_chars>" not in result
    assert "<observation_tail>" not in result
    assert "<warning>" not in result
    assert "Y" * 8000 in result
