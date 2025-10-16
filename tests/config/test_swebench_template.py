from dataclasses import dataclass
from pathlib import Path

import yaml
from jinja2 import StrictUndefined, Template


@dataclass
class MockOutput:
    """Mock output object for testing the template"""

    returncode: int
    output: str


def test_action_observation_template_short_output():
    """Test that short output (< 10000 chars) is displayed in full"""
    # Load the swebench config
    config_path = Path(__file__).parent.parent.parent / "src" / "minisweagent" / "config" / "extra" / "swebench.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract the template
    template_str = config["agent"]["action_observation_template"]
    template = Template(template_str, undefined=StrictUndefined)

    # Create mock output with short content
    output = MockOutput(returncode=0, output="Success! Operation completed.\nWarning: minor issue")

    # Render the template
    result = template.render(output=output)

    # Verify the result contains all parts and no truncation
    assert "<returncode>" in result
    assert "0" in result
    assert "<output>" in result
    assert "Success! Operation completed." in result
    assert "Warning: minor issue" in result

    # Should not contain truncation elements for short output
    assert "<output_head>" not in result
    assert "<elided_chars>" not in result
    assert "<output_tail>" not in result
    assert "<warning>" not in result


def test_action_observation_template_long_output():
    """Test that long output (> 10000 chars) is truncated with head/tail format"""
    # Load the swebench config
    config_path = Path(__file__).parent.parent.parent / "src" / "minisweagent" / "config" / "extra" / "swebench.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract the template
    template_str = config["agent"]["action_observation_template"]
    template = Template(template_str, undefined=StrictUndefined)

    # Create mock output with long content
    long_output = "A" * 8000 + "B" * 3000  # 11000 characters total
    # Total will be > 10000 chars

    output = MockOutput(returncode=1, output=long_output)

    # Render the template
    result = template.render(output=output)

    # Should contain truncation elements for long output
    assert "<warning>" in result
    assert "The output of your last command was too long" in result
    assert "<output_head>" in result
    assert "<elided_chars>" in result
    assert "characters elided" in result
    assert "<output_tail>" in result

    # Should still contain the basic structure
    assert "<returncode>" in result
    assert "1" in result

    # Verify the head contains first part of output
    head_start = result.find("<output_head>")
    head_end = result.find("</output_head>")
    head_content = result[head_start:head_end]
    assert "AAAA" in head_content  # Should contain start of output

    # Verify the tail contains last part of output
    tail_start = result.find("<output_tail>")
    tail_end = result.find("</output_tail>")
    tail_content = result[tail_start:tail_end]
    assert "BBBB" in tail_content  # Should contain end of output


def test_action_observation_template_edge_case_exactly_10000_chars():
    """Test the boundary case where output is around 10000 characters"""
    # Load the swebench config
    config_path = Path(__file__).parent.parent.parent / "src" / "minisweagent" / "config" / "extra" / "swebench.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract the template
    template_str = config["agent"]["action_observation_template"]
    template = Template(template_str, undefined=StrictUndefined)

    # Use a large amount of data that will definitely exceed 10000 chars when rendered
    output = MockOutput(returncode=0, output="X" * 10000)

    # Render the template
    result = template.render(output=output)

    # Should use truncated format for large output
    assert "<output_head>" in result
    assert "<elided_chars>" in result
    assert "<output_tail>" in result
    assert "<warning>" in result
    # The X's should still be present in head or tail
    assert "XXXX" in result


def test_action_observation_template_just_under_10000_chars():
    """Test that smaller output shows full output without truncation"""
    # Load the swebench config
    config_path = Path(__file__).parent.parent.parent / "src" / "minisweagent" / "config" / "extra" / "swebench.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract the template
    template_str = config["agent"]["action_observation_template"]
    template = Template(template_str, undefined=StrictUndefined)

    # Use a reasonably sized output that should be well under 10000 chars when rendered
    output = MockOutput(returncode=0, output="Y" * 8000)

    # Render the template
    result = template.render(output=output)

    # Should show full output without truncation
    assert "<output_head>" not in result
    assert "<elided_chars>" not in result
    assert "<output_tail>" not in result
    assert "<warning>" not in result
    assert "Y" * 8000 in result
