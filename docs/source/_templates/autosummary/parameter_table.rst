.. rubric:: Instrument parameters

.. list-table::
   :header-rows: 1

   {% block header %}
   * - Name
     - Description
   {% endblock %}

   {% block content %}
   {% for name, param in parameters.items() %}
   * - :attr:`{{ name }}`
     - {{ param["docstring"] }}
   {%- endfor %}
   {% endblock %}
