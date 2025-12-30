import argparse
import csv
from datetime import datetime
from database import connect_db


def query_attendance(date=None, enrollment=None):
    conn = connect_db()
    cursor = conn.cursor()

    base_query = "SELECT id, enrollment, name, date, time, status FROM attendance"
    params = []
    clauses = []

    if date:
        clauses.append("date = ?")
        params.append(date)
    if enrollment:
        clauses.append("enrollment = ?")
        params.append(enrollment)

    if clauses:
        q = base_query + " WHERE " + " AND ".join(clauses) + " ORDER BY date, time"
    else:
        q = base_query + " ORDER BY date, time"

    cursor.execute(q, tuple(params))
    rows = cursor.fetchall()
    conn.close()
    return rows


def print_rows(rows):
    if not rows:
        print("No attendance records found.")
        return

    # simple pretty print
    print(f"Found {len(rows)} records:\n")
    print(f"{'ID':<4} {'Enrollment':<16} {'Name':<30} {'Date':<12} {'Time':<10} {'Status'}")
    print('-' * 80)
    for r in rows:
        id_, enroll, name, date, time, status = r
        print(f"{id_:<4} {enroll:<16} {name:<30} {date:<12} {time:<10} {status}")


def export_csv(rows, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['id', 'enrollment', 'name', 'date', 'time', 'status'])
        w.writerows(rows)
    print(f"Exported {len(rows)} rows to {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check attendance records')
    parser.add_argument('--date', '-d', help='Date in YYYY-MM-DD format (default: today)')
    parser.add_argument('--enrollment', '-e', help='Filter by enrollment number')
    parser.add_argument('--export', '-x', help='Export results to CSV file path')

    args = parser.parse_args()

    if args.date:
        # validate format
        try:
            datetime.strptime(args.date, '%Y-%m-%d')
        except ValueError:
            print('Invalid date format. Use YYYY-MM-DD')
            raise SystemExit(1)
    else:
        args.date = datetime.now().strftime('%Y-%m-%d')

    rows = query_attendance(date=args.date, enrollment=args.enrollment)
    print_rows(rows)

    if args.export:
        export_csv(rows, args.export)
