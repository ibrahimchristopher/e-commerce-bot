#python script to test out the database created

import sqlite3

# Connect to the database
db_file = "orders.db"
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Example 1: Count the number of orders
cursor.execute("SELECT COUNT(*) FROM orders")
total_orders = cursor.fetchone()[0]
print(f"Total Orders: {total_orders}")

# Example 2: Fetch all orders with the status 'Completed'
cursor.execute("SELECT * FROM orders WHERE status = 'Completed'")
completed_orders = cursor.fetchall()
print("\nCompleted Orders:")
for order in completed_orders[:10]:  # Limit output to the first 10 rows
    print(order)

# Example 3: Fetch orders placed on a specific date (e.g., '2025-01-15')
specific_date = "2025-01-15"
cursor.execute("SELECT * FROM orders WHERE order_date = ?", (specific_date,))
orders_on_date = cursor.fetchall()
print(f"\nOrders on {specific_date}:")
for order in orders_on_date:
    print(order)

# Example 4: Display distinct statuses in the table
cursor.execute("SELECT DISTINCT status FROM orders")
statuses = cursor.fetchall()
print("\nOrder Statuses:")
for status in statuses:
    print(status[0])

# Close the connection
conn.close()
