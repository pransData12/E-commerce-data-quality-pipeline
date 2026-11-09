-- Business questions answered on the CLEAN warehouse.
-- Format: each query starts with "-- name: <query_name>". "Revenue" = completed + shipped orders.

-- name: monthly_revenue_last_12
SELECT strftime('%Y-%m', order_date)  AS month,
       COUNT(*)                        AS orders,
       ROUND(SUM(total_amount), 2)     AS revenue,
       ROUND(AVG(total_amount), 2)     AS avg_order_value
FROM orders
WHERE status IN ('completed', 'shipped')
GROUP BY month
ORDER BY month DESC
LIMIT 12;

-- name: revenue_growth_mom
WITH monthly AS (
    SELECT strftime('%Y-%m', order_date) AS month, SUM(total_amount) AS revenue
    FROM orders
    WHERE status IN ('completed', 'shipped')
    GROUP BY month
)
SELECT month,
       ROUND(revenue, 2) AS revenue,
       ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month))
             / LAG(revenue) OVER (ORDER BY month), 1) AS mom_growth_pct
FROM monthly
ORDER BY month DESC
LIMIT 12;

-- name: revenue_by_category
SELECT p.category,
       SUM(oi.quantity)                               AS units_sold,
       ROUND(SUM(oi.quantity * oi.unit_price), 2)     AS revenue
FROM order_items oi
JOIN orders   o ON o.order_id   = oi.order_id
JOIN products p ON p.product_id = oi.product_id
WHERE o.status IN ('completed', 'shipped')
GROUP BY p.category
ORDER BY revenue DESC;

-- name: top_10_customers_by_lifetime_value
SELECT RANK() OVER (ORDER BY SUM(o.total_amount) DESC) AS rank,
       c.customer_id,
       c.name,
       c.country,
       COUNT(*)                        AS orders,
       ROUND(SUM(o.total_amount), 2)   AS lifetime_value
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
WHERE o.status IN ('completed', 'shipped')
GROUP BY c.customer_id
ORDER BY rank
LIMIT 10;

-- name: repeat_customer_rate
WITH per_customer AS (
    SELECT customer_id, COUNT(*) AS n_orders FROM orders GROUP BY customer_id
)
SELECT COUNT(*)                                                      AS customers_with_orders,
       SUM(CASE WHEN n_orders >= 2 THEN 1 ELSE 0 END)                AS repeat_customers,
       ROUND(100.0 * SUM(CASE WHEN n_orders >= 2 THEN 1 ELSE 0 END)
             / COUNT(*), 1)                                          AS repeat_rate_pct
FROM per_customer;

-- name: cancel_and_return_rate_by_country
SELECT c.country,
       COUNT(*)                                                  AS orders,
       ROUND(100.0 * SUM(o.status = 'cancelled') / COUNT(*), 1)  AS cancel_pct,
       ROUND(100.0 * SUM(o.status = 'returned')  / COUNT(*), 1)  AS return_pct
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
GROUP BY c.country
ORDER BY orders DESC;

-- name: dq_failures_by_rule
SELECT rule_id, severity, rows_failed, fail_rate_pct
FROM dq_rule_results
WHERE rows_failed > 0
ORDER BY rows_failed DESC;
