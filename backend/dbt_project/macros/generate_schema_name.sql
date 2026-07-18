{#
    Use custom schema names verbatim instead of dbt's default
    "<target_schema>_<custom_schema>" concatenation.

    Without this override, a model configured with +schema: public would land in
    a Postgres schema called "main_public" (target.schema is "main", the DuckDB
    default). We want +schema: public to mean exactly the `public` schema the API
    and Alembic use. Models with no +schema fall back to target.schema.
#}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
