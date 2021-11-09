# Python coding conventions

## Naming conventions

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

### Working on existing files

If you have to work on an existing file that do not comply with our conventions
(e.g. legacy file or file copied from an external source), the rule is to follow
the existing conventions within this file.

### New files

When creating new files, make sure to follow the established coding conventions
from the start!

## Coding conventions

### Use 4 spaces

**Always use 4 spaces for indentation, and never use tabs.** Mixing spaces and
tabs will lead to Python throwing a lot of `TabError`. Your IDE should be
configured to never place any tabs characters.

### Max line width

Configure your IDE to display a ruler at 80 characters. The lines of codes (or
text in documentation page) should be kept before this limit as much as
possible.

This 80 character limit is not a hard limit, but it should be followed to
guarantee the readability of the code. The rule can of course be broken if the
line cannot be broken into multiple lines for language syntax reason, e.g. a
link in Markdown:
```markdown
The link is too long to fit within 80 characters:
[Google's Python styleguide](https://google.github.io/styleguide/pyguide.html#s3.16-naming)
```
