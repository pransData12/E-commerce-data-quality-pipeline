# Rules reference

Rules live in [`config/rules.yaml`](../config/rules.yaml). Every rule needs four common keys plus the parameters of its `type`.

| Common key | Meaning |
|---|---|
| `id` | unique name, shown in the report and in `dq_violations` |
| `type` | which check to run (table below) |
| `table` | the table the rule is evaluated on |
| `severity` | `error` = offending rows are quarantined; `warning` = reported only |
| `description` | one line shown in the report |

## Check types

| `type` | Parameters | Fails a row when |
|---|---|---|
| `not_null` | `column` | value is missing or blank |
| `unique` | `column` | value already appeared in an earlier row (the first occurrence passes) |
| `regex` | `column`, `pattern` | a present value does not fully match the pattern |
| `range` | `column`, `min`?, `max`? | value is non-numeric or outside `[min, max]` |
| `allowed_values` | `column`, `values` | a present value is not in the list |
| `date_not_future` | `column` | date is unparseable or after `as_of` |
| `foreign_key` | `column`, `ref_table`, `ref_column` | value does not exist in the parent table |
| `date_order` | `column`, `key`, `ref_table`, `ref_key`, `ref_column` | child date is earlier than the parent's date (e.g. order before signup) |
| `price_consistency` | `column`, `product_column`, `ref_table`, `ref_key`, `ref_price`, `tolerance` | price differs from the product's reference price by more than `tolerance` (0.15 = 15%) |
| `sum_match` | `key`, `column`, `child_table`, `child_key`, `child_qty`, `child_price`, `tolerance` | parent total differs from the sum of child `qty * price` by more than `tolerance` |

`as_of` at the top of the YAML is the fixed "today" used by date checks, so runs are reproducible.

## Adding a new check type

Write one function in `dq_pipeline/validators.py` and register it. It returns the labels of failing rows and the number of rows it could test:

```python
@register("min_length")
def min_length(tables, rule, as_of):
    df = tables[rule["table"]]
    s = df[rule["column"]].astype("string")
    bad = s.notna() & (s.str.len() < rule["length"])
    return _failed(df, bad), int(s.notna().sum())
```

Use it in the YAML immediately:

```yaml
- id: short_name
  type: min_length
  table: customers
  column: name
  length: 3
  severity: warning
  description: Name is suspiciously short
```

Add a unit test in `tests/test_validators.py` with a tiny hand-made DataFrame whose right answer you know.
