# evo-control-score

This repository contains code for project - Evolutionary Search for Toxicity Control in LLMs using Context-aware Scoring.

## Setup Instructions

##### To clone the github repositories
```
git clone https://github.com/Onkar2102/evo-control-score.git
cd evo-tox-llm
```

##### To work with the latest dev code:
```
git checkout develop
git pull origin develop
```

##### Setting up environment
```
pip3 install -r requirements.txt
```

##### To explore a specific feature branch:
```
[Onkar]
git checkout feature/evolution-core

[Roopikaa]
git checkout feature/embedding

[Bhaskar]
git checkout feature/metric-engine
```

##### To merge with develop branch
```
git merge develop
```

##### Commit and Push to feature branch
```
git add .
git commit -m "Implement new feature XYZ"
git push origin feature/feature branch name
```

##### Merge changes from feature to develop
```
git checkout develop
git pull origin develop
git merge feature/my-feature
git push origin develop
```

##### Merge changes from develop to main
[please discuss before merging develop and main always]
```
git checkout main
git pull origin main
git merge develop
git push origin main
```