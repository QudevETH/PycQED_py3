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

### Avoid magic numbers

A magic number is a value that should be given a symbolic name, but is instead
used directly in the code, usually in more than one place. This is problematic
for two reasons:
* Difficult to understand: understanding what the value represents without a
name usually requires reading through the code in depth, and sometimes having
domain specific knowledge.
* Difficult to maintain: magic numbers are hardcoded values, that need to be
changed in more than one place. This must usually be done (semi-)manually, and
can lead to errors if not all occurences are updated.

Do
```python
BASE_ADDRESS = 185600
DELAY_REGISTER_ADDRESS = BASE_ADDRESS + 15
set_value(DELAY_REGISTER_ADDRESS, 15e-9) # ns
# ...
set_value(DELAY_REGISTER_ADDRESS, 0)
```

Don't
```python
set_value(185615, 15e-9)
# ...
set_value(185615, 0)
```

### Strings

* Use double quotes:
```python
do = "use double quotes"
dont = 'do not use single quotes'
```

* Exception: if the string contains a lot of `"` or `'`, you can use the other
quotes to encompass the string:
```python
ok = 'This "string" contains a lot of "quotes", so it is "okay".'
```

* When you need to include values inside a string, favor using
[f-strings](https://docs.python.org/3/tutorial/inputoutput.html#formatted-string-literals)
rather than `string.format()` or adding strings:
```python
a = 2
b = 3
best = f"{a} + {b} = {a + b}"
ok = "{} + {} = {}".format(a, b, a + b)
avoid = str(a) + " + " + str(b) + " = " + str(a + b)
```

* Break your strings in multiple lines when needed:
```python
long_string = "One possible way to break lines in string " \
              "is using a backslash"
```

### Comments

* Always write comments in (reasonable) english.
* Don't hesitate to add links if you copied a complex formula or code snippet
from a website, if this helps for understanding.
* Always flag any part of the code that would need to be cleaned, corrected or
double checked. Use `TODO` or `FIXME` prefixes for such comments. This way, one
can simply search for these terms to get a reasonable list of items that need to
be cleaned or reviewed.
* Avoid commenting out code, and always write a comment specifing a meaninful
reason for the reason behing this code being commented. If a substantial
modification needs to be made that justifies this commenting-out, you should
rather work on a branch.
* Do not comment out code with triple quotes, this syntax is used primarily for
docstrings. All IDEs have a shortcut for commenting selected lines (e.g.
`Ctrl+K, Ctrl+C` in VS code).
