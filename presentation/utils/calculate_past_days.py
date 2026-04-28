from datetime import date, datetime, timezone

def calculate_past_days(diary_entries: list[dict], min_days: int = 1, max_days = 90) -> int:
    if not diary_entries:
        return min_days

    dates = []

    for entry in diary_entries:
        local_date = entry.get("localDate")

        if isinstance(local_date, str):
            dates.append(date.fromisoformat(local_date))
        elif isinstance(local_date, date):
            dates.append(local_date)

    if not dates:
        return min_days

    oldest_date = min(dates)
    today = datetime.now(timezone.utc).date()

    return max((today - oldest_date).days + 1, max_days)