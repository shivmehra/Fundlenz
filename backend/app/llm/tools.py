TOOL_COMPUTE_METRIC = {
    "type": "function",
    "function": {
        "name": "compute_metric",
        "description": (
            "Use for AGGREGATIONS over a tabular file (CSV/Excel) — one number or chart. "
            "Examples: average NAV, sum of AUM, max return, top 5 by AUM, trend over time. "
            "Do NOT use for filtering rows (use query_table) or for PDFs/Word documents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Filename of the tabular file (CSV/XLSX) the user is asking about.",
                },
                "op": {
                    "type": "string",
                    "enum": ["mean", "sum", "count", "min", "max", "top_n", "trend"],
                },
                "column": {
                    "type": "string",
                    "description": "The numeric column to aggregate.",
                },
                "group_by": {
                    "type": "string",
                    "description": "Optional column to group by.",
                },
                "n": {
                    "type": "integer",
                    "description": "For top_n: how many rows to return.",
                },
            },
            "required": ["filename", "op", "column"],
        },
    },
}


TOOL_QUERY_TABLE = {
    "type": "function",
    "function": {
        "name": "query_table",
        "description": (
            "Use to filter, sort, and slice ROWS from a tabular file (CSV/Excel). "
            "Use when the user wants to see a subset of rows. "
            "Examples: 'list all debt funds', 'show funds with AUM > 100', "
            "'top 10 funds sorted by return descending', 'find rows where category = Equity'. "
            "Do NOT use for aggregations like average/sum/max (use compute_metric) or for PDFs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Filename of the tabular file (CSV/XLSX) the user is asking about.",
                },
                "filters": {
                    "type": "array",
                    "description": (
                        "Optional list of filter conditions. Each item is "
                        "{column, op, value}. op ∈ {==, !=, >, >=, <, <=, contains, in}. "
                        "For 'contains', value is matched case-insensitively as a substring. "
                        "For 'in', value is a list."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string"},
                            "op": {
                                "type": "string",
                                "enum": ["==", "!=", ">", ">=", "<", "<=", "contains", "in"],
                            },
                            "value": {
                                "description": "String, number, or array (for 'in' op).",
                            },
                        },
                        "required": ["column", "op", "value"],
                    },
                },
                "sort_by": {
                    "type": "string",
                    "description": "Optional column name to sort the result by.",
                },
                "sort_desc": {
                    "type": "boolean",
                    "description": "Sort descending. Defaults to true.",
                },
                "select_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional subset of columns to include. Defaults to all.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum rows to return. Defaults to 50.",
                },
            },
            "required": ["filename"],
        },
    },
}


TOOLS = [TOOL_COMPUTE_METRIC, TOOL_QUERY_TABLE]


SYSTEM_PROMPT = """You are Fundlenz, an expert analyst that answers questions about fund documents.

Answer questions using the retrieved document context provided below. Cite the source filename and page number when possible.

OUTPUT RULES — read carefully:
- NEVER write Python, pandas, SQL, or any executable code in your replies. You cannot execute code, and the user does not want code — they want the answer.
- NEVER copy raw retrieved chunk text into your reply. The chunks are inputs TO YOU, not output to the user. In particular, do not echo pipe-separated table rows like `| col1 | col2 | val1 | val2 |` — those are summary fragments meant for your reading. Synthesize a clean prose answer or render a proper markdown table from the values you choose to include.
- For ANY aggregation over a CSV/Excel file (max, min, average, mean, median, sum, count, top-N, trend, breakdown by group), you MUST call compute_metric. Tabular summary chunks contain only a SAMPLE of the data — picking a max or sum from those chunks gives the wrong answer.
- For ROW-LEVEL queries that filter, sort, or slice (e.g. "list all debt funds", "show funds with AUM > 100", "top 10 funds sorted by return", "find rows where category = X"), call query_table. It returns a filtered/sorted subset of rows as a clean markdown table.
- compute_metric vs. query_table: compute_metric collapses many rows to ONE number/chart; query_table returns MANY rows. If the user wants a number → compute_metric. If the user wants rows → query_table.
- For point lookups about a specific entity ("expense ratio of Fund X", "AUM of Y"), prefer query_table with a filter on the matching column. Or, if the row chunk for that entity is in your retrieved context (formatted `Row from filename: col=val; col=val; ...`), you may read the value directly. Don't guess.
- For "list all distinct X" queries on a single column, the retrieved enumeration chunks (beginning with `All distinct values in column ...`) are exhaustive — use them.
- For summaries, explanations, and comparisons: answer in clean prose. Use markdown tables only when comparing structured items, and only with values you have actually seen in context.
- If the context is insufficient to answer, say so clearly. Do not invent figures, column names, fund names, or any other facts.
"""
