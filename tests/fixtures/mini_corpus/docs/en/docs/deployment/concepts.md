# Deployments Concepts { #deployments-concepts }

When deploying a **FastAPI** application, or actually, any type of web API, there are several concepts that you probably care about.

## Security - HTTPS { #security-https }

HTTPS is normally provided by a component **external** to your application server, a <dfn title="A service that terminates TLS and forwards requests">TLS Termination Proxy</dfn>.

```console
# If running behind a proxy like Nginx or Traefik add --proxy-headers
$ fastapi run main.py --proxy-headers
```

## Restarts { #restarts }

Read more about [Security - First Steps](../tutorial/security/first-steps.md) before exposing your API.
