from datetime import timedelta
from html import escape
from study_planner.schemas import Schedule

def create_weekly_schedule(schedule: Schedule, tasks: dict) -> str:
    if not schedule.blocks:
        return """
        <div class="empty-schedule">
            <h2>No study sessions scheduled</h2>
            <p>Your weekly schedule is currently empty.</p>
        </div>
        """

    # find the beginning of the week containing the earliest study block
    earliest_start = min(block.start for block in schedule.blocks)

    week_start = earliest_start.replace(
        hour=0, minute=0, second=0, microsecond=0
    ) - timedelta(days=earliest_start.weekday())

    # display the full 24 hours of each day
    start_hour = 0
    end_hour = 24

    hour_height = 40
    timeline_height = (end_hour - start_hour) * hour_height

    # assign a different color to each task
    colors = [
        "#DCEBFA",
        "#FCE4D6",
        "#E4F1DF",
        "#F2E3F7",
        "#FFF1C9",
        "#DDF2F0"
    ]

    task_ids = sorted(set(block.task_id for block in schedule.blocks))

    task_colors = {}

    for i, task_id in enumerate(task_ids):
        task_colors[task_id] = colors[i % len(colors)]

    # generate the time labels
    time_labels = ""

    for hour in range(start_hour, end_hour):
        time_labels += f"""
        <div class="time-label" style="top: {hour * hour_height}px">
            {datetime_time(hour)}
        </div>
        """

    days = []

    for day_number in range(7):
        day_start = week_start + timedelta(days=day_number)
        day_end = day_start + timedelta(days=1)
        current_date = day_start.date()

        block_html = ""

        # find every block that overlaps with this day
        day_blocks = []

        for block in schedule.blocks:
            if block.start < day_end and block.end > day_start:
                day_blocks.append(block)

        day_blocks.sort(key=lambda block: block.start)

        for block in day_blocks:
            task = tasks.get(block.task_id, {})

            task_name = escape(task.get("name", block.task_id))
            course = escape(task.get("course") or "")

            # split the block at midnight if necessary.``
            segment_start = max(block.start, day_start)
            segment_end = min(block.end, day_end)

            start_time = segment_start.strftime("%I:%M %p")
            end_time = segment_end.strftime("%I:%M %p")

            # calculate the block's position and height
            start_minutes = (
                segment_start - day_start
            ).total_seconds() / 60

            duration_minutes = (
                segment_end - segment_start
            ).total_seconds() / 60

            top = start_minutes * hour_height / 60
            height = duration_minutes * hour_height / 60

            task_color = task_colors[block.task_id]

            block_html += f"""
            <div class="study-block"
                 style="top: {top}px;
                        height: {height}px;
                        background-color: {task_color};">

                <strong>{task_name}</strong>
                <small>{start_time} - {end_time}</small>

                <div class="study-popup" hidden>
                    <h3>{task_name}</h3>
                    <p><strong>Course:</strong> {course or "N/A"}</p>
                    <p><strong>Start:</strong> {start_time}</p>
                    <p><strong>End:</strong> {end_time}</p>
                    <p><strong>Duration:</strong> {int(duration_minutes)} minutes</p>
                </div>

            </div>
            """

        if not block_html:
            block_html = '<p class="no-blocks">No study sessions</p>'

        days.append(f"""
        <div class="day-column">
            <div class="day-header">
                <h3>{current_date.strftime("%A")}</h3>
                <p>{current_date.strftime("%b %d")}</p>
            </div>

            <div class="day-timeline" style="height: {timeline_height}px;">
                {block_html}
            </div>
        </div>
        """)

    return f"""
    <div class="calendar-scroll">
        <div class="weekly-calendar"
             style="--timeline-height: {timeline_height}px;">

            <div class="time-axis">
                <div class="axis-header"></div>
                <div class="time-labels" style="height: {timeline_height}px;">
                    {time_labels}
                </div>
            </div>

            {"".join(days)}
        </div>
    </div>
    """


def datetime_time(hour: int) -> str:
    """Format an hour as a 12-hour clock label."""
    suffix = "AM" if hour < 12 else "PM"
    display_hour = hour % 12 or 12
    return f"{display_hour} {suffix}"
