.. py:attribute:: {{ name }}
    :type: {{ klass }}
    :value: {{ initial_value }}

    {% if label %}
        {% if label_contains_latex %}
    **Label:** :math:`{{ label }}`
        {% else %}
    **Label:** {{ label }}
        {% endif %}
    {% endif %}

    {% if unit %}
    **Unit:** {{ unit }}
    {% endif %}

    {% if vals %}
    **Vals:** {{ vals }}
    {% endif %}

    {{ docstring }}
