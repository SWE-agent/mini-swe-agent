from pathlib import Path

from study_planner.ui.mock_data import mock_schedule, mock_tasks
from study_planner.ui.schedule_view import create_weekly_schedule


def main():
    schedule_html = create_weekly_schedule(
        mock_schedule,
        mock_tasks,
    )

    css_path = Path(__file__).parent / "schedule.css"
    css = css_path.read_text(encoding="utf-8")

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Weekly Study Schedule</title>
        <style>
            {css}
        </style>
    </head>
    <body>
        <h1 class="page-title">My Weekly Study Schedule</h1>
        {schedule_html}

        <script>
        document.addEventListener("click", function(event) {{{{
            const clickedBlock = event.target.closest(".study-block");
            const clickedPopup = event.target.closest(".study-popup");

            // Clicking inside the popup does nothing.
            if (clickedPopup) {{{{
                return;
            }}}}

            // Check whether the clicked task's popup is already open.
            let alreadyOpen = false;

            if (clickedBlock) {{{{
                const popup = clickedBlock.querySelector(".study-popup");
                alreadyOpen = popup && !popup.hidden;
            }}}}

            // Close all popups.
            document.querySelectorAll(".study-popup").forEach(popup => {{{{
                popup.hidden = true;
            }}}});

            // Open the clicked task's popup only if it wasn't already open.
            if (clickedBlock && !alreadyOpen) {{{{
                const popup = clickedBlock.querySelector(".study-popup");

                if (popup) {{{{
                    popup.hidden = false;
                }}}}
            }}}}
        }}}});
        </script>
    </body>
    </html>
    """

    output_path = Path(__file__).parent / "weekly_schedule.html"
    output_path.write_text(html, encoding="utf-8")

    print(f"Schedule preview created: {output_path}")


if __name__ == "__main__":
    main()
