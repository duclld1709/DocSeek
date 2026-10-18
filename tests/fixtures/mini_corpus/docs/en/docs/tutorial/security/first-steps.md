# Security - First Steps { #security-first-steps }

Let's imagine that you have your **backend** API in some domain, and you want the frontend to authenticate using a **username** and **password**.

## Create `main.py` { #create-main-py }

Copy the example in a file `main.py`:

{* ../../docs_src/security/tutorial001_an_py310.py *}

## `OAuth2PasswordBearer` { #oauth2passwordbearer }

When we create an instance of the `OAuth2PasswordBearer` class we pass in the `tokenUrl` parameter.

/// tip | Authorize button!

You already have a shiny new "Authorize" button in the interactive docs.

///

The `oauth2_scheme` variable is an instance of `OAuth2PasswordBearer`, but it is also a "callable", so it can be used with `Depends`. A database session for the current user can be provided with a [dependency with yield](../dependencies/dependencies-with-yield.md#a-database-dependency-with-yield).
