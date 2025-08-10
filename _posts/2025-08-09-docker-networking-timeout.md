-----

layout: post title: "Taming the Timeout: A Docker Networking Deep Dive (and Victory\!)" date: 2025-08-10 02:06:22 -0400 categories: [docker, networking, traefik, troubleshooting] tags: [gateway timeout, docker compose, reverse proxy, microservices]

Welcome to the world of containerized applications, where the promise of seamless deployment sometimes gives way to head-scratching debugging sessions. Recently, I battled a particularly frustrating "**Gateway Timeout**" error when trying to access a service behind Traefik. The logs seemed to indicate everything was fine, yet requests consistently failed. This post details the steps taken to diagnose and ultimately resolve this **Docker networking puzzle**.

-----

## The Setup: A Network of Services

Our basic setup involved:

  * A core application stack defined in a `docker-compose.yml` file, including a `historical-service` (our problematic component) and a PostgreSQL database.
  * A separate `docker-compose.yml` for Traefik, the reverse proxy responsible for routing external requests to the correct services based on hostnames and paths.
  * A shared external Docker network, `swift-net`, connecting both Traefik and our application services.

The goal was straightforward: access the `historical-service` via `http://api.pokemon.com/historical` routed through Traefik.

-----

## The Problem Emerges: A Gateway Timeout

Initial attempts to hit the service consistently resulted in a **504 Gateway Timeout** error. This suggested that Traefik was trying to contact the backend service but not receiving a timely response.

-----

## Initial Investigations (and Red Herrings)

The journey to the solution was filled with common troubleshooting steps that, while valuable, didn't lead to the final answer.

  * **Host File Verification**: We ensured our local host file (`/etc/hosts`) correctly mapped `api.pokemon.com` to `127.0.0.1`.
  * **Traefik Configuration**: We checked Traefik's dashboard and logs, which indicated the routing rule for `api.pokemon.com` was correctly configured and pointing to the `historical-service`.
  * **Application Logs**: This is where things got interesting. The logs of the `historical-service` showed successful processing of requests. This led us to believe the application itself was fine.
  * **Health Checks**: The `/health` endpoint of the `historical-service` was correctly implemented and returned a `200 OK`.

The successful logs from a single container, coupled with the persistent timeout, hinted at an issue with the service's replicas. By watching the logs of all replicas simultaneously, we observed that some containers were successfully processing requests while others seemed to hang.

-----

## The Isolation Test: Bypassing Traefik

To definitively rule out the application as the core issue, we performed a crucial test: directly exposing the `historical-service`'s port (3000) to the host machine (on port 3001) and making a direct `curl` request.

```bash
curl http://localhost:3001
```

The result was a resounding success\! We received a **200 OK** response with the expected JSON payload. This proved beyond doubt that the `historical-service` application was healthy and functioning correctly. The problem, therefore, was not the application, but the way Traefik was communicating with it.

-----

## The Real Culprit: Docker Network Ambiguity

With the application exonerated, the focus shifted entirely to **Docker networking**. The `historical-service` was connected to two networks: `app-network` (its internal application network) and `swift-net` (the shared network with Traefik).

Traefik was only configured to use `swift-net`. The problem arose because Docker, in some scenarios with **multi-homed containers** (connected to multiple networks), can create ambiguity in how Traefik tries to reach the backend service. It didn't know which network interface to use.

-----

## The Solution: Explicit Traefik Network Configuration

The fix was surprisingly simple but essential: **explicitly telling Traefik which Docker network to use** for discovering and communicating with backend services. This was achieved by adding the following command-line argument to the Traefik container in its `docker-compose.yml`:

```yaml
services:
  traefik:
    image: traefik:v2.10
    command:
      - "--providers.docker=true"
      - "--providers.docker.network=swift-net"
      # ... other commands
```

After restarting Traefik, the routing worked flawlessly. The **Gateway Timeout disappeared**, and requests to `http://api.pokemon.com/historical` were successfully processed.

-----

## Conclusion: Debugging Docker Networks

This debugging journey highlights the intricacies of **Docker networking**, especially when dealing with multiple networks and reverse proxies. While application logs are crucial, understanding how Docker manages network communication between containers is equally important. Explicitly configuring Traefik's Docker provider to use the correct network resolved the ambiguity and brought our services to life. This experience serves as a valuable reminder to pay close attention to network configurations in your Docker Compose setups.