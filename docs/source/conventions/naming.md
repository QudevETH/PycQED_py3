# Python naming conventions

Please refer to [Google's Python styleguide](https://google.github.io/styleguide/pyguide.html#s3.16-naming)
for naming your Python entities.

Here is a short summary of the naming conventions:
* `module_name`
* `package_name`
* `ClassName`
* `method_name`
* `ExceptionName`
* `function_name`
* `GLOBAL_CONSTANT_NAME`
* `global_var_name`
* `instance_var_name`
* `function_parameter_name`
* `local_var_name`

Avoid:
* Using dashes (-) in any package/module name
* Needlessly including the type of the variable in its name (e.g.
`id_to_name_dict`), use type hints instead.

## Working on existing files

If you have to work on an existing file that do not comply with our conventions
(e.g. legacy file or file copied from an external source), the rule is to follow
the existing conventions within this file.

## New files

When creating new files, make sure to follow the established coding conventions
from the start!
