# Path Parameters { #path-parameters }

You can declare path "parameters" or "variables" with the same syntax used by Python format strings:

{* ../../docs_src/path_params/tutorial001_py310.py hl[6:7] *}

The value of the path parameter `item_id` will be passed to your function as the argument `item_id`.

## Predefined values { #predefined-values }

If you have a *path operation* that receives a *path parameter*, but you want the possible valid *path parameter* values to be predefined, you can use a standard Python <abbr title="Enumeration">`Enum`</abbr>.

/// warning

Path operations are evaluated in order, so `/users/me` must be declared before `/users/{user_id}`.

///
