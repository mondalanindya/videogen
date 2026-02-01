# GitHub Upload Instructions

Your videogen project is now ready to upload to GitHub! Follow these steps:

## Step 1: Create a GitHub Repository

1. Go to [GitHub](https://github.com) and log in
2. Click the **+** icon in the top right, then select **New repository**
3. Fill in the repository details:
   - **Repository name**: `videogen` (or your preferred name)
   - **Description**: "Video generation pipeline with trajectory planning and frame synthesis"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
4. Click **Create repository**

## Step 2: Connect Your Local Repository to GitHub

After creating the repository on GitHub, you'll see instructions. Run these commands:

```bash
cd /projects/u6bl/myprojects/videogen

# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/videogen.git

# Push your code to GitHub
git push -u origin main
```

## Step 3: Verify Upload

Go to your GitHub repository page and verify that all files are uploaded:
- ✅ All Python scripts (.py files)
- ✅ Documentation files (.md files)
- ✅ Configuration files (requirements.txt, .gitignore)
- ✅ Input samples
- ✅ Shell scripts

## What's Included

The following files have been committed and are ready to push:
- **Core Scripts**: 5 Python scripts (pipeline.py, validate.py, scripts/*.py)
- **Documentation**: 8 markdown/text files (README.md, INSTALL.md, EXAMPLES.md, etc.)
- **Configuration**: requirements.txt, .gitignore, SETUP_SUMMARY.sh
- **Sample Inputs**: first_frame.png, prompt.txt

## What's Excluded (via .gitignore)

The following are automatically excluded and won't be uploaded:
- ❌ Virtual environments (.venv/)
- ❌ Cache directories (cache/, __pycache__/)
- ❌ Output files (outputs/, err/)
- ❌ Large model files
- ❌ SLURM job files (.sub)

## Update Git User Information (Optional)

If you want to use your actual name and email:

```bash
cd /projects/u6bl/myprojects/videogen

# Update for this repository only
git config user.name "Your Actual Name"
git config user.email "your-actual-email@example.com"

# Or update globally for all repositories
git config --global user.name "Your Actual Name"
git config --global user.email "your-actual-email@example.com"

# Amend the commit with new author information
git commit --amend --reset-author --no-edit
git push -f origin main  # Force push to update the commit
```

## Authentication

When pushing to GitHub, you may need to authenticate. GitHub no longer supports password authentication for Git operations. Use one of these methods:

### Option 1: Personal Access Token (Recommended)
1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate a new token with `repo` scope
3. Use the token as your password when prompted

### Option 2: SSH Key
1. Generate SSH key: `ssh-keygen -t ed25519 -C "your-email@example.com"`
2. Add key to GitHub: Settings → SSH and GPG keys
3. Change remote URL: `git remote set-url origin git@github.com:YOUR_USERNAME/videogen.git`

## Next Steps After Upload

1. **Add a license**: Consider adding a LICENSE file (MIT, Apache 2.0, etc.)
2. **Create releases**: Tag versions of your project
3. **Enable GitHub Actions**: Set up CI/CD for automated testing
4. **Add badges**: Display build status, license, etc. in README
5. **Write CONTRIBUTING.md**: Guide others on how to contribute

## Troubleshooting

**Error: "remote origin already exists"**
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/videogen.git
```

**Error: "failed to push some refs"**
```bash
git pull origin main --rebase
git push -u origin main
```

**Large files warning**
If you accidentally added large files, use:
```bash
git rm --cached path/to/large/file
git commit --amend
```

---

**Your repository is ready!** Just follow Step 2 above to connect to GitHub and push your code.
