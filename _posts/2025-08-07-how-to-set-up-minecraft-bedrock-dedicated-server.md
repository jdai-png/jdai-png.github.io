---
layout: post
title: "How to Set Up Your Own Minecraft Bedrock Dedicated Server"
date: 2025-08-07 13:55:10 -0400
author: Gemini AI
categories: [gaming, tutorials]
tags: [minecraft, bedrock, dedicated-server, self-hosting, windows, linux]
description: "A complete step-by-step guide to downloading, configuring, and running your own private Minecraft Bedrock Dedicated Server on Windows or Linux."
image: /assets/img/minecraft/server-banner.webp
---

Running your own **Minecraft Bedrock Dedicated Server** 🎮 gives you complete control over your multiplayer world. You can create a persistent space for you and your friends to build, explore, and adventure together, without relying on Realms or third-party hosting. This guide, based on the original tutorial by ProfessorValko, will walk you through the entire process, from initial setup to ongoing maintenance.

> *Disclaimer: This guide is based on the alpha version of the official server software. Some details may have changed in newer releases. It is not officially endorsed by or affiliated with Minecraft/Mojang.*

***

### Getting Started: What You'll Need

Before you can build your new world, you need to make sure your hardware and software are up to the task. Here are the minimum requirements.

#### System & Hardware Requirements 🖥️

* **Operating System:**
    * Windows 10 (version 1703 or later) or Windows Server 2016 (or later).
    * Ubuntu 18 (Other Linux distributions might work but are not officially supported).
    * **Note for Windows users:** You may need to install the [Visual C++ Redistributable for Visual Studio 2015](https://www.microsoft.com/en-us/download/details.aspx?id=48145) to get all the necessary components.
* **Hardware:**
    * A 64-bit (`x86_64`) Intel or AMD processor.
    * A dual-core processor is recommended.
    * At least **1GB of RAM**.

***

### The Initial Setup: Bringing Your Server to Life

With the requirements met, it's time to get your server configured and running.

#### Step 1: Download and Configure

1.  **Download the Server Software:** Grab the latest version from the official [Minecraft server download page](https://minecraft.net/en-us/download/server/bedrock/).
2.  **Extract the Files:** Unzip the downloaded file into a dedicated folder where you want your server to live.
3.  **Configure Server Properties:** Open the `server.properties` file with a plain text editor (like Notepad++ or VS Code). This file controls your world's name, game mode, difficulty, and more. Review the `bedrock_server_how_to.html` file included in the download for a full list of options and their allowed values.

#### Step 2: Set Up Player Access (Whitelist & Permissions)

You'll need to manually create two important files to manage who can join and what they can do.

* **Create `whitelist.json`:**
    1.  In your server folder, create a new file named `whitelist.json`.
    2.  Open it and type `[]` inside.
    3.  Save the file.
    4.  If you set `white-list=true` in `server.properties`, you must manually add the gamertags of allowed players to this file. Otherwise, players will be added automatically when they join.

* **Create `permissions.json`:**
    1.  Create another new file named `permissions.json`.
    2.  Open it and, just like before, type `[]` inside.
    3.  Save the file.
    4.  This file is used to assign permissions, like making a player an **operator**. You'll need a player's XUID (which appears in the server console when they connect) to grant them permissions here.

#### Step 3: Launch the Server!

* **On Windows:** Double-click the `bedrock_server.exe` file. A console window will appear and start the server process.
* **On Ubuntu Linux:** Open a terminal, navigate to your server directory, and run the command:

    ```bash
    LD_LIBRARY_PATH=. ./bedrock_server
    ```

Keep the console window open—closing it will shut down your server. If you see errors, check the troubleshooting section below.

***

### Connecting to Your New World

Now that your server is running, players need a way to join. The method depends on whether they are on the same local network as the server.

#### Connecting Over LAN

Players on the same Wi-Fi or home network can join easily. The server will automatically appear under the **Friends** tab in their Minecraft client.

#### Connecting with an IP Address

For a more direct connection, or for friends joining over the internet, you'll need the server's IP address.

* **Private IP (LAN):** For players on your local network. You can find your server's private IPv4 address by typing `ipconfig` (on Windows) or `ifconfig` (on Linux) in a command prompt or terminal.
* **Public IP (External):** For players outside your network. You can find this by searching "what is my IP address" on Google from the computer hosting the server.

Once you have the IP, players can go to the **Servers** tab, click "Add Server," and enter the server's name, IP address, and port (the default is `19132`).

#### Essential Steps for Connectivity

* **Port Forwarding:** To allow players from the internet to connect, you must configure **port forwarding** on your router. You'll need to forward UDP/TCP port `19132` (for IPv4) to the private IP address of your server computer. The process varies by router, so consult your router's manual or manufacturer's website.
* **UWP Loopback (Windows 10 Only):** If you are hosting the server and playing Minecraft on the *same* Windows 10 PC, you must enable a loopback exemption. Open Command Prompt as an administrator and run this exact command:
    ```bash
    CheckNetIsolation.exe LoopbackExempt –a –p=S-1-15-2-1958404141-86561845-1752920682-3514627264-368642714-62675701-733520436
    ```

***

### Customizing and Managing Your Server

A dedicated server offers deep customization, from player roles to using your favorite worlds.

#### Assigning Operator Permissions

Operators (or "ops") can use in-game commands like `/gamemode`, `/teleport`, and `/kick`. You can assign this role by editing the **`permissions.json`** file.

1.  Have the player join the server once to generate their XUID in the console.
2.  Open `permissions.json`.
3.  Add an entry for the player, setting their permission level to `operator` and pasting their `xuid`.
4.  Save the file and type `permissions reload` in the server console to apply the changes.

#### Using Existing Worlds and Add-Ons

You can import a world you've already started or downloaded.

1.  Stop the server.
2.  Copy the entire world save folder into the `worlds` folder in your server directory.
3.  Open `server.properties` and set the `level-name` property to match the exact name of your world's folder.
4.  Restart the server.

To use **add-ons** (resource or behavior packs), apply them to your world in-game *before* you move the world folder to your server. This ensures all the necessary configuration files are generated correctly. Note that add-ons from the Marketplace or those using experimental features are not supported.

***

### Server Maintenance and Updates

Keeping your server healthy and current is key to a smooth experience.

#### Updating the Server Software

1.  **Backup first!** In the server console, use the `save hold`, `save query`, and `save resume` commands to safely prepare your world. Then, stop the server and make a copy of your entire server folder.
2.  Download the new server version and extract it to a new folder.
3.  Copy your old `server.properties`, `whitelist.json`, `permissions.json` files, and your entire `worlds` folder into the new server directory, overwriting the defaults.
4.  Start the new `bedrock_server.exe`.

#### Troubleshooting Common Issues

* **Error: `Chakra.dll` or other `.dll` not found:** Your operating system is not supported, or you need to install the Visual C++ Redistributable linked at the beginning of this guide.
* **Error: "NO LOG FILE!":** This is normal. The alpha server software does not yet create a log file.
* **Error: "Error opening ops file / whitelist file":** This typically means the `permissions.json` or `whitelist.json` file is missing. You must create them manually, even if they are empty (`[]`).
* **Player permissions don't save:** Permissions set with in-game commands are temporary. For persistent operator status, you must edit the `permissions.json` file.