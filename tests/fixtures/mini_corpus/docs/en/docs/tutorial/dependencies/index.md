# Dependencies { #dependencies }

**FastAPI** has a very powerful but intuitive **<dfn title="also known as components, resources, providers, services, injectables">Dependency Injection</dfn>** system.

## First Steps { #first-steps }

Let's see a very simple example.

### Create a dependency, or "dependable" { #create-a-dependency-or-dependable }

It is just a function that can take all the same parameters that a *path operation function* can take:

{* ../../docs_src/dependencies/tutorial001_an_py310.py hl[8:9] *}

### Declare the dependency, in the "dependant" { #declare-the-dependency-in-the-dependant }

The same way you use `Body`, `Query`, etc. with your *path operation function* parameters, use `Depends` with a new parameter.

/// tip

You only give `Depends` a single parameter. It must be something like a function. You **don't call it** directly.

///

See also [Dependencies with yield](dependencies-with-yield.md).
