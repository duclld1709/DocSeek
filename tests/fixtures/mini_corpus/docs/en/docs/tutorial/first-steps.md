# First Steps { #first-steps }

The simplest FastAPI file could look like this:

{* ../../docs_src/first_steps/tutorial001_py310.py *}

Copy that to a file `main.py`.

Run the live server:

<div class="termy">

```console
# Install FastAPI first
$ pip install "fastapi[standard]"
$ <font color="#4E9A06">fastapi</font> dev main.py
```

</div>

## Interactive API docs { #interactive-api-docs }

Now go to <a href="http://127.0.0.1:8000/docs" class="external-link" target="_blank">http://127.0.0.1:8000/docs</a>.

You will see the automatic interactive API documentation (provided by Swagger UI):

<img src="/img/index/index-01-swagger-ui-simple.png">

### Step 1: import `FastAPI` { #step-1-import-fastapi }

```Python
# This line is a comment, not a heading
from fastapi import FastAPI
```

`FastAPI` is a Python class that provides all the functionality for your API.

/// note | Technical Details

`FastAPI` is a class that inherits directly from `Starlette`.

///
