# My Shiny Python App

## Overview
This project is a Shiny application built using Python, designed to explore and visualize data related to chemical discovery trends. The application allows users to interactively select countries, filter data, and view various plots and tables that illustrate key findings from the data.

## Project Structure
The project consists of the following files and directories:

- `app.py`: The main entry point of the Shiny Python application, responsible for setting up the server and UI components.
- `requirements.txt`: A list of dependencies required to run the application.
- `data/data.parquet`: The data file in Parquet format used for analysis and visualization.
- `utils/__init__.py`: Marks the `utils` directory as a Python package.
- `utils/functions.py`: Contains utility functions for data processing and visualization.
- `www/styles.css`: Custom CSS styles for the application's UI.
- `www/original_article.pdf`: A PDF document providing access to the original article referenced in the app.

## Installation
To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd my-shiny-python-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
To run the application, execute the following command in your terminal:
```
python app.py
```
Once the application is running, open your web browser and navigate to `http://localhost:5000` to access the app.

## Contributing
Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.