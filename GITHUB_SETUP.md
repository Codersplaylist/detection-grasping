# GitHub Setup Guide

This guide will help you set up and publish this project to GitHub.

## Step 1: Prepare Your Repository

### Rename the Project Directory (Optional but Recommended)

```bash
# Navigate to parent directory
cd /Users/mac/Documents/

# Rename project folder
mv "appa project all files" "fruit-detection-grasping"

# Navigate into renamed folder
cd fruit-detection-grasping
```

### Create .gitkeep Files for Empty Directories

```bash
# Create directory markers to preserve structure
touch models/.gitkeep
touch dataset/train/images/.gitkeep
touch dataset/train/labels/.gitkeep
touch dataset/val/images/.gitkeep
touch dataset/val/labels/.gitkeep
```

## Step 2: Initialize Git Repository

```bash
# Initialize git
git init

# Add all files
git add .

# Make initial commit
git commit -m "Initial commit: Fruit Detection & Robotic Grasping System

- YOLO11-based object detection for fruits
- Real-time grasp coordinate calculation
- Complete training pipeline
- 90-95% mAP accuracy on validation set
- Includes dataset conversion, training, and inference"
```

## Step 3: Create GitHub Repository

### Option A: Using GitHub Website

1. Go to https://github.com/new
2. Repository name: `fruit-detection-grasping`
3. Description: `Real-time fruit detection and robotic grasping using YOLOv11`
4. Choose **Public** (for portfolio) or **Private**
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

### Option B: Using GitHub CLI (if installed)

```bash
gh repo create fruit-detection-grasping --public --source=. --description="Real-time fruit detection and robotic grasping using YOLOv11"
```

## Step 4: Connect and Push to GitHub

After creating the repository on GitHub, you'll see instructions. Follow these:

```bash
# Add remote origin (replace 'yourusername' with your GitHub username)
git remote add origin https://github.com/yourusername/fruit-detection-grasping.git

# Verify remote
git remote -v

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 5: Add Model File (Important!)

The trained model (`models/my_model.pt`) is excluded from git (too large). You have two options:

### Option A: GitHub Releases (Recommended)

1. Go to your repository on GitHub
2. Click "Releases" → "Create a new release"
3. Tag version: `v1.0.0`
4. Release title: `Initial Release - Trained Model`
5. Drag and drop `models/my_model.pt` to the release assets
6. Publish release

Update README with download instructions:
```markdown
## Download Trained Model

Download the pre-trained model from [Releases](https://github.com/yourusername/fruit-detection-grasping/releases/latest):

1. Download `my_model.pt`
2. Place it in the `models/` directory
3. Run the application: `python3 camera_yolo_integrated.py`
```

### Option B: Git LFS (Large File Storage)

```bash
# Install Git LFS
brew install git-lfs  # macOS
# Or download from: https://git-lfs.github.com/

# Initialize Git LFS
git lfs install

# Track .pt files
git lfs track "*.pt"

# Add .gitattributes
git add .gitattributes

# Now add and commit the model
git add models/my_model.pt
git commit -m "Add trained model weights"
git push
```

## Step 6: Add Demo Images/Video (Optional but Recommended)

```bash
# Create docs directory
mkdir -p docs

# Add screenshots or demo video
# Take a screenshot of the application running
# Create a GIF of detection in action using tools like:
# - macOS: QuickTime Player + Gifski
# - https://ezgif.com/ (online converter)

# Add to git
git add docs/
git commit -m "Add demo images and documentation"
git push
```

## Step 7: Configure Repository Settings

On GitHub, go to your repository settings:

### Repository Topics
Add topics for discoverability:
- `computer-vision`
- `yolo`
- `object-detection`
- `robotics`
- `deep-learning`
- `grasp-detection`
- `pytorch`
- `opencv`

### About Section
Edit the "About" section:
- ✅ Description: "Real-time fruit detection and robotic grasping using YOLOv11"
- ✅ Website: Your portfolio or demo link (if available)
- ✅ Topics: (as listed above)

### Social Preview
Settings → General → Social Preview
- Upload an image (1280x640px) showing your system in action

## Step 8: Add GitHub Actions (Optional - CI/CD)

Create `.github/workflows/python-app.yml`:

```yaml
name: Python application

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Lint with flake8
      run: |
        pip install flake8
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

## Step 9: Create a Great README

Your README should include:
- ✅ Project description
- ✅ Features and use cases
- ✅ Installation instructions
- ✅ Usage examples
- ✅ Performance metrics
- ✅ Screenshots/GIFs
- ✅ API documentation
- ✅ Contributing guidelines
- ✅ License information
- ✅ Contact information

All of this is already in your README.md!

## Step 10: Promote Your Repository

### Update Your GitHub Profile
Add to your pinned repositories:
1. Go to your GitHub profile
2. Click "Customize your pins"
3. Select this repository

### Share on LinkedIn
Post about your project:
```
🤖 Excited to share my latest project: Fruit Detection & Robotic Grasping System!

✨ Features:
- Real-time object detection using YOLOv11
- Intelligent grasp point calculation for robotics
- 90-95% detection accuracy
- Complete end-to-end ML pipeline

Built with: Python, PyTorch, OpenCV, YOLO
Application: Agricultural robotics, warehouse automation

Check it out on GitHub: [link]
#ComputerVision #Robotics #MachineLearning #AI
```

## Maintenance

### Regular Updates

```bash
# Pull latest changes
git pull origin main

# Make changes
# ... edit files ...

# Commit and push
git add .
git commit -m "Descriptive commit message"
git push origin main
```

### Create Branches for Features

```bash
# Create a new feature branch
git checkout -b feature/improve-accuracy

# Make changes
# ... 

# Push feature branch
git push origin feature/improve-accuracy

# Create Pull Request on GitHub
```

### Add Collaborators

Settings → Collaborators and teams → Add people

## Common Git Commands

```bash
# Check status
git status

# View changes
git diff

# View commit history
git log --oneline --graph

# Create new branch
git checkout -b branch-name

# Switch branches
git checkout main

# Merge branch
git merge feature-branch

# Delete local branch
git branch -d branch-name

# Delete remote branch
git push origin --delete branch-name

# Undo last commit (keep changes)
git reset --soft HEAD^

# Discard local changes
git checkout -- filename
```

## Troubleshooting

### Large Files Error

If you get an error about file size:
```bash
# Remove from git (keep local file)
git rm --cached models/my_model.pt

# Add to .gitignore
echo "models/*.pt" >> .gitignore

# Commit
git add .gitignore
git commit -m "Exclude large model files from git"
git push
```

Then use GitHub Releases for the model file.

### Authentication Required

```bash
# Set up SSH key (recommended)
ssh-keygen -t ed25519 -C "your.email@example.com"

# Add to GitHub: Settings → SSH and GPG keys → New SSH key
cat ~/.ssh/id_ed25519.pub

# Change remote to SSH
git remote set-url origin git@github.com:yourusername/fruit-detection-grasping.git
```

Or use Personal Access Token:
- GitHub Settings → Developer settings → Personal access tokens
- Use token as password when pushing

## Final Checklist

Before sharing your repository, verify:

- [ ] README.md is complete and professional
- [ ] .gitignore excludes sensitive/large files
- [ ] LICENSE file is present
- [ ] Code is well-commented
- [ ] Requirements.txt is up-to-date
- [ ] Setup instructions are clear
- [ ] Demo images/video included
- [ ] Repository topics added
- [ ] Model weights available (via Releases or LFS)
- [ ] All commits have meaningful messages
- [ ] No API keys or secrets in code
- [ ] Project builds successfully from repo clone

## Example Companies To Share With

When sharing with potential employers:

**Email Template:**
```
Subject: Portfolio Project - Computer Vision & Robotics

Hi [Name],

I wanted to share a computer vision project I developed that demonstrates skills relevant to [Company/Position]:

Fruit Detection & Robotic Grasping System
GitHub: https://github.com/yourusername/fruit-detection-grasping

Key highlights:
• Real-time object detection (90-95% accuracy)
• End-to-end ML pipeline (data → training → deployment)
• Robot integration with grasp planning
• Production-ready code with comprehensive documentation

The system combines YOLOv11 for detection with custom grasp point calculation, designed for robotic pick-and-place applications.

I'd be happy to discuss the technical implementation or answer any questions.

Best regards,
[Your Name]
```

---

Congratulations! Your project is now professionally presented on GitHub! 🎉
