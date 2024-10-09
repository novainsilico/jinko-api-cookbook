# Jinko API Cookbooks

Welcome to the Jinko API Cookbooks repository. This project contains practical examples and tutorials for using the Jinko API, aimed at helping users integrate and leverage Jinko's features programmatically.

## Project Overview

The Jinko API Cookbooks provide a series of Jupyter Notebooks demonstrating real-life use cases and practical applications of the Jinko API. These cookbooks are designed to help users navigate common tasks and advanced scenarios using the Jinko platform.

Jinko is an innovative SaaS and CaaS platform focused on trial simulation and design optimization. The Jinko API offers programmatic access to a wide range of functionalities, facilitating integration with various tools and enhancing collaboration.

Find out more at [Jinko Doc](https://doc.jinko.ai)

## Structure

- **[Quick start](#quick-start)**
  - Cloning and installing the project
  - How to register a token
  - Environment setup
- **[Tutorials](/tutorial)**
  - Sequential walkthroughs of available services
    - [01-getting-started](/tutorial/01-getting-started.ipynb): Setup and user information
    - 02-navigate: List project resources
    - 03-knowledge: Manage knowledge sources, extracts, and documents
    - 04-modeling: Create and edit a model
    - 05-trial: Create a trial, protocol, and VPOP
    - 06-simulate: Run a simulation and read progress
    - 07-analysis: Retrieve and download results
- **[Cookbooks](/cookbooks)** - Real-life examples and practical use cases
  - [template](/cookbooks/template.ipynb) a model to start a new cookbook with.
  - [basic_stat_analysis](/cookbooks/basic_stat_analysis.ipynb) a cookbook to perform basic statistical analysis on trial results.
  - [producing_data_summary](/cookbooks/producing_data_summary.ipynb) a cookbook to produce a summary table for simulated data.
  - [quantifying_uncertainty](/cookbooks/quantifying_uncertainty.ipynb) assess the degree of uncertainty in a trial.
  - [reviewing_model_versions](/cookbooks/reviewing_model_versions.ipynb) a cookbook to show the differences across model versions.
  - [run_a_trial](/cookbooks/run_a_trial.ipynb) a cookbook to run a trial in jinko from scratch.
  - [sensitivity_analysis](/cookbooks/sensitivity_analysis.ipynb) a cookbook to perform sensitivity analysis on trial results via Lasso and Random Forest.
  - [visualizing_scalar_results](/cookbooks/visualizing_scalar_results.ipynb) a cookbook to visualize scalar results from an existing trial in jinko.
  - [visualizing_timeseries](/cookbooks/visualizing_timeseries.ipynb) the creation of a simple visualization from an existing trial in jinko. 


## Quick-Start

To get started, follow these steps:

1. **Clone the Repository**:
    ```sh
    git clone git@git.novadiscovery.net:jinko/api/jinko-api-cookbook.git
    cd jinko-api-cookbook
    ```
2. **Register a Token**: 
   - Open the admin section of a project and go to the API Access tab and click on "New Token" (tutorial [here](https://doc.jinko.ai/docs/quick-start))
   - Copy `.envrc.sample` to `.envrc` and adjust the variable in it. The project id can be found in the url (e.g. `https://jinko.ai/project/<project-id>`) 
   - Source `.envrc` 
  
3. **Run cookbooks**:
   - With [nix](https://nixos.org):
     ```sh
     # Simple nix shell with core requirements (poetry)
     nix develop
    
     # Open a poetry shell with installed requirements
     nix develop .#poetry

     # Open jupyter-lab
     nix develop .#lab
     ```
   - With python and poetry
     ```
     poetry install
     poetry shell
     jupyter-lab
     ```
   - With vscode: see [Official Jupyter integration](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)



## Contributing

Thank you for your interest in contributing to the Jinko API Cookbooks! Your contributions help make our documentation and examples better for everyone.


To maintain a high standard and ensure consistency across all cookbooks, please follow these guidelines when contributing:

### 1. Use the basic model

-  Copy this basic [template](./cookbooks/template.ipynb) to initialize a new cookbook.

### 2. Create an MR Per Cookbook

- Each contribution should be made as a separate Merge Request (MR).
- This allows for easier review and ensures that changes are focused and isolated.

### 3. One Cookbook = One Use Case

- Each Jupyter Notebook (cookbook) should focus on a single, well-defined use case.
- Avoid combining multiple use cases into one notebook.

### 4. Keep It Light and Simple

- Aim to keep the examples straightforward and easy to understand.
- Avoid adding unnecessary complexity.

### 5. Comment and Illustrate with Real Use Cases

- Provide clear and concise comments within the code to explain the steps and logic.
- Illustrate each use case with real examples using the Jinko API.

### 6. Focus on Illustrating API Usage

- The primary goal is to demonstrate the use of the Jinko API.
- Avoid adding too many additional functions or helpers that detract from the main purpose.


## Configuration & advanced initialization

The default configuration is coming from your `.envrc` (see `./.envrc.sample`), but if you need so, here the complete usage of `jinko.initialize()` helper.

```python
# Configuration

# Fill your API key (ex: 'd74cc07e-4a86-4aab-952a-a5814ae6e1ee')
# If not set, it will be retrieved from the JINKO_API_KEY environment variable.
# If environment variable is not set, you will be asked for it interactively
apiKey = None

# Fill your Project Id (ex: '14495063-918c-441a-a898-3131d70b02b0')
# If not set, it will be retrieved from the JINKO_PROJECT_ID environment variable.
# If environment variable is not set, you will be asked for it interactively
projectId = None

# This function ensures that authentication is correct
# It it also possible to override the base url by passing baseUrl=...
jinko.initialize(projectId, apiKey = apiKey)

```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For support or inquiries, please contact us at support@jinko.ai