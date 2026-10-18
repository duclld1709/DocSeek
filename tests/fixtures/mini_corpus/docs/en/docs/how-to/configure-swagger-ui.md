# Configure Swagger UI { #configure-swagger-ui }

You can configure some extra Swagger UI parameters by passing the argument `swagger_ui_parameters` when creating the `FastAPI()` app object.

## Disable Syntax Highlighting { #disable-syntax-highlighting }

```Python
from fastapi import FastAPI

app = FastAPI(swagger_ui_parameters={"syntaxHighlight": False})
```

...and then Swagger UI won't show the syntax highlighting anymore:

<img src="/img/tutorial/extending-openapi/image03.png">
