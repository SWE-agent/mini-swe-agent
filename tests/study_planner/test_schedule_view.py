from study_planner.ui.mock_data import mock_schedule, mock_tasks, empty_schedule
from study_planner.ui.schedule_view import create_weekly_schedule
from study_planner.ui.mock_data import (
    mock_schedule,
    mock_tasks,
    empty_schedule,
    overnight_schedule,
)


def test_weekly_schedule_displays_all_seven_days():
    html = create_weekly_schedule(mock_schedule, mock_tasks)

    assert "Monday" in html
    assert "Tuesday" in html
    assert "Wednesday" in html
    assert "Thursday" in html
    assert "Friday" in html
    assert "Saturday" in html
    assert "Sunday" in html


def test_weekly_schedule_displays_multiple_blocks_same_day():
    html = create_weekly_schedule(mock_schedule, mock_tasks)

    assert "Study for CS Midterm" in html
    assert "Study for Math Midterm" in html


def test_empty_schedule_displays_message():
    html = create_weekly_schedule(empty_schedule, mock_tasks)

    assert "No study sessions scheduled" in html
    assert "currently empty" in html


def test_task_split_across_multiple_days():
    html = create_weekly_schedule(mock_schedule, mock_tasks)

    # CS midterm appears on both monday and tuesday
    assert html.count("<strong>Study for CS Midterm</strong>") == 2

def test_overnight_session_is_split_across_two_days():
    html = create_weekly_schedule(overnight_schedule, mock_tasks)

    # the session appears as two separate blocks
    assert html.count('class="study-block"') == 2

    # monday: 11 PM to midnight
    assert "11:00 PM - 12:00 AM" in html

    # tuesday: midnight to 1 AM
    assert "12:00 AM - 01:00 AM" in html


def test_study_block_contains_popup_details():
    html = create_weekly_schedule(mock_schedule, mock_tasks)

    assert "study-popup" in html
    assert "Course:" in html
    assert "Start:" in html
    assert "End:" in html
    assert "Duration:" in html


def test_study_blocks_have_background_colors():
    html = create_weekly_schedule(mock_schedule, mock_tasks)

    assert "background-color:" in html
