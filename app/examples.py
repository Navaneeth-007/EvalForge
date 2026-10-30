"""Authored walkthrough fixtures, never presented as model predictions."""

SCHEMA = "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary REAL);"
EXAMPLES = [
    {
        "question": "What is the average salary by department?",
        "sql": "SELECT department, AVG(salary) AS average_salary FROM employees GROUP BY department;",
    },
    {
        "question": "Who are the five highest paid employees?",
        "sql": "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 5;",
    },
    {
        "question": "How many employees work in each department?",
        "sql": "SELECT department, COUNT(*) AS employee_count FROM employees GROUP BY department;",
    },
]
for example in EXAMPLES:
    example["schema"] = SCHEMA
