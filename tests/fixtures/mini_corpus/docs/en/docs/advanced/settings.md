# Settings and Environment Variables { #settings-and-environment-variables }

In many cases your application could need some external settings or configurations, for example secret keys, database credentials, credentials for email services, etc.

## Pydantic `Settings` { #pydantic-settings }

{* ../../docs_src/settings/tutorial001_py310.py hl[2,5:8,11] *}

### Run the server { #run-the-server }

Next, you would run the server passing the configurations as environment variables:

//// tab | Linux, macOS, Windows Bash

<div class="termy">

```console
$ ADMIN_EMAIL="deadpool@example.com" APP_NAME="ChimichangApp" fastapi run main.py
```

</div>

////

//// tab | Windows PowerShell

<div class="termy">

```console
$ $Env:ADMIN_EMAIL = "deadpool@example.com"
$ fastapi run main.py
```

</div>

////

/// tip

To set multiple env vars for a single command just separate them with a space.

///
