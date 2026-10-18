# Dependencies with yield { #dependencies-with-yield }

FastAPI supports dependencies that do some <dfn title="sometimes also called 'exit code', 'cleanup code', 'teardown code'">extra steps after finishing</dfn>.

To do this, use `yield` instead of `return`, and write the extra steps after.

## A database dependency with `yield` { #a-database-dependency-with-yield }

For example, you could use this to create a database session and close it after finishing.

Only the code prior to and including the `yield` statement is executed before creating a response:

{* ../../docs_src/dependencies/tutorial007_an_py310.py ln[1:4] hl[2:3] *}

The code following the `yield` statement is executed after the response:

{* ../../docs_src/dependencies/tutorial007_an_py310.py ln[5:6] hl[5:6] *}

/// note | Technical Details

Any function that is valid to use with `@contextlib.contextmanager` or `@contextlib.asynccontextmanager` would be valid to use as a **FastAPI** dependency.

///

## Dependencies with `yield` and `HTTPException` { #dependencies-with-yield-and-httpexception }

You can raise an `HTTPException` after the `yield`, see [Handling Errors](../handling-errors.md#use-httpexception).

/// danger

Do not swallow exceptions in the exit code of a dependency with `yield` without re-raising them.

///
