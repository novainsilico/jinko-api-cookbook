## Instructions to run the Trial Design Optimization demo using the AD model

  1. Log in Nova Demo
  2. AD demo trial: https://jinko.ai/tr-4bHo-BMO3
  3. If not done already, create a Jinko API key for the Demo project https://jinko.ai/project/a9593bfe-0abb-416b-8f93-4654a2b47960 
  4. If not done already, clone the cookbooks repo. The demo should run out of the box on the `main` branch:
```
git clone git@git.novadiscovery.net:jinko/api/jinko-api-cookbook.git
```
  5. Edit the “.envrc” file at the root of the repo (or create a new one if it does not exist)
```
export JINKO_API_KEY="<your Jinko API key for the Nova Demo project>"
export JINKO_PROJECT_ID="a9593bfe-0abb-416b-8f93-4654a2b47960"
export JINKO_BASE_URL="https://api.jinko.ai"
```
  6. You may have to run “direnv allow” afterwards 
  7. Still at the root of the repo, run the command
```
jinko-sdk-shell
```
this should open a Nix shell with all the dependencies needed to run the cookbook 
  8. Open VSCode with 
```
code .
```
  9. Navigate to “cookbooks/r_and_d/trial_design_optimization/continuous_or_binary_outcome.ipynb”
 10. Click on the “Run All” button in the Jupyter command bar. VSCode may ask you which kernel you want to use, just pick the first proposed one, should be something along the lines of “Python 13.8 …”  
 11. On my machine in runs in under 20s but please note that it may be slower if you are sharing your screen during a video call.
