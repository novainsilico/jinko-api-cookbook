# Jinko API Cookbooks

Jinko API Cookbooks: practical notebooks and tutorials to use the Jinko API programmatically.

**Who this is for**
- **External/public users**: this repo is public for reference. You can browse notebooks and run those that don’t depend on internal assets; others will need your own Jinko project/data.
- **Nova internal users**: run everything as-is. Prerequisites: `jinko-sdk-shell` Python env and an API key on the `Cookbooks` project inside group `Api Access Cookbooks` (org `Nova Internal`).

## Project Overview

The Jinko API Cookbooks provide a series of Jupyter Notebooks demonstrating real-life use cases and practical applications of the Jinko API. These cookbooks are designed to help users navigate common tasks and advanced scenarios using the Jinko platform.

Jinko is an innovative SaaS and CaaS platform focused on trial simulation and design optimization. The Jinko API offers programmatic access to a wide range of functionalities, facilitating integration with various tools and enhancing collaboration.

Find out more at [Jinko Doc](https://doc.jinko.ai)

## Structure

- **Tutorials** (`/tutorials`): best starting point for new/external users; sequential walkthroughs of API usage.
- **Cookbooks** (`/cookbooks`): practical examples organized by product area (each folder has its own `resources/` when needed); these often depend on internal assets/access
  - Basics & execution: `cookbooks/basics` - Introductory notebooks.
  - Modeling, calibration & uncertainty: `cookbooks/modeling` - Modeling tools (model calibratopn, virtual population generation).
  - Trial simulation & analytics: `cookbooks/trial_simulation_and_analytics` - Run trials and perform advanced analytics on trial results. 
  - AI agents: `cookbooks/ai-agents` - AI agents as an accelerator.
  - R&D / advanced: `cookbooks/r_and_d` - trial design optimization, vpop calibration with SAEM and deep learning techniques.


## Quick-Start

### Public/external users
1. **Get access**: create a Jinko account at [jinko.ai](https://www.jinko.ai), then create a project and generate an API key for it.
2. **Run the tutorials**: open `tutorials/` and follow the notebook instructions—start with the first two notebooks to confirm your setup works end-to-end. You can run them locally or launch them directly in Google Colab.
3. **Local setup (optional)**: clone the repo, copy `.envrc.sample` → `.envrc`, fill in your project ID + API key, and install the Jinko SDK in your Python env (`pip install jinko-sdk` or use `poetry` with the provided `pyproject.toml`).
Feel free to use any environment manager; the key requirement is having `jinko-sdk` installed.

### Nova internal users
1. **Clone**:
    ```sh
    git clone git@git.novadiscovery.net:jinko/api/jinko-api-cookbook.git
    cd jinko-api-cookbook
    ```
2. **API key**: From org `Nova Internal` → group `Api Access Cookbooks` → project `Cookbooks`, create/reuse a token (Admin → API Access → New Token).
3. **Environment**:
    ```sh
    # ensure Nova Python collection is installed locally
    doctor_manage_collections
    # load the legacy env for _deprecated notebooks
    jinko-sdk-shell --refresh
    # load the env for current notebooks (typed jinko-sdk)
    jwb-shell --refresh
    cp .envrc.sample .envrc    # fill in your key + project id
    direnv allow
    ```
4. **Run cookbooks**:
    - Current notebooks (typed `jinko-sdk`): from inside `jwb-shell`, run `code .` or `jupyter-lab`, then pick the `jwb-shell` kernel.
    - `_deprecated` notebooks (legacy `jinko_helpers`): from inside `jinko-sdk-shell`, run `code .` or `jupyter-lab`, then pick the `jinko-sdk-shell` kernel. These are kept for reference only — prefer the current notebook with the same base name when one exists.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For support or inquiries, please contact us at support@jinko.ai
