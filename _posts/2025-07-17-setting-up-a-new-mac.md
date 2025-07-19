---
layout: post
title: "My Blueprint for Setting Up a New Mac"
date: 2025-07-19 16:41:09 -0400
categories: macos setup development
---

Setting up a new Mac can be a chore, but with a solid plan, you can get a powerful development environment up and running quickly. Here's my personal checklist for getting a fresh macOS machine ready, from dotfiles to development tools.

***

## Step 1: Install Matthias's Dotfiles

First things first, I lay down a strong foundation with [Mathias Bynens's dotfiles](https://github.com/mathiasbynens/dotfiles). They provide a ton of sensible defaults, aliases, and functions that make the command line a much nicer place to be.

I skip the bootstrap script and pull them in directly with this command:

```bash
cd; curl -#L [https://github.com/mathiasbynens/dotfiles/tarball/main](https://github.com/mathiasbynens/dotfiles/tarball/main) | tar -xzv --strip-components 1 --exclude={README.md,bootstrap.sh,.osx,LICENSE-MIT.txt}
````

This downloads the repository and places the files directly in my home directory, ready to go.

-----

## Step 2: Create a Personal `.extra` File

To keep my personal settings separate from the main dotfiles repository, I create a `~/.extra` file. This is the perfect place for secrets and machine-specific configurations, like Git credentials. The dotfiles automatically source this file if it exists.

My `~/.extra` looks something like this:

```bash
# Git credentials
# Not in the repository, to prevent people from accidentally committing under my name
GIT_AUTHOR_NAME="Mathias Bynens"
GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME"
git config --global user.name "$GIT_AUTHOR_NAME"
GIT_AUTHOR_EMAIL="mathias@mailinator.com"
GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
git config --global user.email "$GIT_AUTHOR_EMAIL"
```

This ensures all my commits are correctly authored without hardcoding my details into the main configuration.

-----

## Step 3: Install Homebrew, Ruby & Jekyll

macOS comes with a system version of Ruby, but it's best to leave it alone. For web development with tools like Jekyll, you'll want a modern version managed separately.

### Install Homebrew

[Homebrew](https://brew.sh/) is the essential package manager for macOS. If it's not already installed, a single command will take care of it:

```bash
/bin/bash -c "$(curl -fsSL [https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh](https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh))"
```

### Install Ruby with chruby

Next, I use Homebrew to install `chruby` and `ruby-install`, which are a lightweight and effective combination for managing Ruby versions.

1.  **Install the tools:**

    ```bash
    brew install chruby ruby-install
    ```

2.  **Install the latest stable Ruby version:**

    ```bash
    ruby-install ruby 3.4.1
    ```

    This can take a few minutes as it compiles from source.

3.  **Configure your shell:** Add the following lines to your `~/.bash_profile` (or `~/.zshrc` if you're using Zsh) to auto-load `chruby` and select your new Ruby version in new terminal sessions.

    ```bash
    echo "source $(brew --prefix)/opt/chruby/share/chruby/chruby.sh" >> ~/.bash_profile
    echo "source $(brew --prefix)/opt/chruby/share/chruby/auto.sh" >> ~/.bash_profile
    echo "chruby ruby-3.4.1" >> ~/.bash_profile
    ```

4.  **Restart your terminal** and verify the installation:

    ```bash
    ruby -v
    ```

    You should see `ruby 3.4.1` or whatever version you installed.

### Install Jekyll

With a modern Ruby environment ready, installing Jekyll is just one command away:

```bash
gem install jekyll
```

-----

## Step 4: Set Up GitHub SSH Keys

To securely connect to GitHub without entering a password every time, an SSH key is a must.

1.  **Generate a new key:** I use the `ed25519` algorithm, which is modern and secure.

    ```bash
    ssh-keygen -t ed25519 -C "your_email@example.com"
    ```

    Follow the prompts, accepting the default file location.

2.  **Copy the public key to your clipboard:**

    ```bash
    pbcopy < ~/.ssh/id_ed25519.pub
    ```

3.  **Add it to GitHub:** Go to your [GitHub account settings](https://github.com/settings/keys), click "New SSH key," give it a descriptive title, and paste the key from your clipboard.

-----

## Step 5: Add a Quality-of-Life Alias

One of the best parts of having a custom setup is adding little helpers. After installing `yt-dlp` (`brew install yt-dlp`), I add this alias to my `.extra` file (or `.aliases` in the dotfiles structure) for easily downloading videos.

```bash
# yt-dlp: Download the best quality video (capped at 1080p/60fps) as an MP4 with embedded subtitles.
# Usage: yt "URL"
alias yt='yt-dlp -f "bestvideo[height<=1080][fps<=60]+bestaudio/best" --merge-output-format mp4 --embed-subs --write-sub'
```

Now, `yt "some-video-url"` is all I need to save a video for offline viewing.

-----

## Bonus: Troubleshooting GPG Commit Signing ✍️

If you've configured Git to sign commits, you might run into the dreaded `error: gpg failed to sign the data`. This means Git can't find or use your GPG key. Here’s how to fix it.

### 1\. Check for a GPG Key

First, check if you have a GPG key on your machine.

```bash
gpg --list-secret-keys --keyid-format=long
```

If you have a key, the output will look something like this. Copy the GPG key ID that starts after `sec`. In this example, it's `3AA5C34371567BD2`.

```
/Users/hubot/.gnupg/pubring.kbx
------------------------------------
sec   rsa4096/3AA5C34371567BD2 2024-01-01 [SC]
      A1234567890B1234567890C1234567890D1234E5F
uid                 [ultimate] Octocat <octocat@github.com>
ssb   rsa4096/4A12345B123456C1 2024-01-01 [E]
```

If you don't have a key, you'll need to [generate one](https://docs.github.com/en/authentication/managing-commit-signature-verification/generating-a-new-gpg-key).

### 2\. Configure Git and Your Shell

Once you have your key ID, you need to tell Git to use it and ensure your shell is configured correctly.

1.  **Tell Git about your key:** Replace `YOUR_KEY_ID` with the key ID you copied.

    ```bash
    git config --global user.signingkey YOUR_KEY_ID
    ```

2.  **Configure your shell:** This is the most common fix. GPG needs to know where to ask for your passphrase. Add the following line to your shell's startup file (e.g., `~/.bash_profile` or `~/.zshrc`).

    ```bash
    echo 'export GPG_TTY=$(tty)' >> ~/.bash_profile
    ```

3.  **Apply the changes:** Either restart your terminal or run `source ~/.bash_profile` (or your corresponding shell file).

Now, try your `git commit` command again. It should prompt you for your GPG key passphrase and then succeed.

```
