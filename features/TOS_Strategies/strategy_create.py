#%%
import json
import requests
from bs4 import BeautifulSoup
import re

# List of URLs to process
# Read URLs from a text file
with open('strategy_urls.txt', 'r') as file:
    urls = [line.strip() for line in file if line.strip()]

#%%
# Function to extract strategy details from a URL
# Function to extract strategy details from a URL
def extract_strategy_details(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract the strategy name (assuming it's within an <h1> tag)
    strategy_name = soup.find('h1').get_text(strip=True) if soup.find('h1') else "Unknown Strategy"

    # Extract all text inside the <article> tag
    article = soup.find('article')
    if article:
        raw_text = article.get_text(separator=" ")  # Extract text with spaces in between
    else:
        raw_text = "No article content found."
    
    # Preprocessing: Remove extra whitespace, newlines, and other unwanted characters
    processed_text = re.sub(r'\s+', ' ', raw_text).strip()

    return {
        'strategy_name': strategy_name,
        'url': url,
        'description': processed_text
    }

#%%
# List to hold all strategy details
strategies = []

# Process each URL
for url in urls:
    try:
        strategy_details = extract_strategy_details(url)
        strategies.append(strategy_details)
    except Exception as e:
        print(f"Error processing {url}: {e}")

#%%
# Save the data to a JSON file
with open('strategies.json', 'w') as json_file:
    json.dump(strategies, json_file, indent=4)

# %%
# Use Openai to create python strategy functions
import openai
import json
import os

def load_strategies(json_file):
    """Loads strategies from a JSON file."""
    with open(json_file, 'r', encoding='utf-8') as file:
        return json.load(file)

def generate_prompt(strategy_name, description, example_function):
    """Creates a prompt for OpenAI API."""
    return f"""
    Create a Python function that outputs signals for the {strategy_name} strategy.
    Use the following description to guide your implementation:
    {description}
    Ensure the function is efficient for large OHLCV datasets of stocks and currencies.
    Use this example functions as a template:
    {example_function}
    """

def get_openai_response(prompt, api_key):
    """Sends a request to OpenAI API and retrieves the response."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful Alogrithmic trading developer assistant."},
                  {"role": "user", "content": prompt}],
        api_key=api_key
    )
    print(response['choices'][0]['message']['content'])
    return response['choices'][0]['message']['content']

def write_to_file(filename, content):
    """Writes generated function to a .py file."""
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)

def main():
    json_file = 'strategies.json'  # Path to JSON file
    output_directory = 'generated_strategies'  # Output directory
    api_key = os.getenv("OPENAI_KEY")  # Load API key from environment variable

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    strategies = load_strategies(json_file)

    example_function = """
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume

# True Strength Index (TSI) Strategy
def tsi_signals(stock_df, window_slow=25, window_fast=13):
    signals = pd.DataFrame(index=stock_df.index)
    tsi = momentum.TSIIndicator(stock_df['Close'], window_slow, window_fast)
    signals['TSI'] = tsi.tsi()
    signals['tsi_signal'] = 'neutral'
    signals.loc[(signals['TSI'] > 0) & (signals['TSI'].shift(1) <= 0), 'tsi_signal'] = 'long'
    signals.loc[(signals['TSI'] < 0) & (signals['TSI'].shift(1) >= 0), 'tsi_signal'] = 'short'
    signals.drop(['TSI'], axis=1, inplace=True)
    return signals
    """

    for strategy in strategies:
        strategy_name = strategy['strategy_name']
        description = strategy.get('description', 'No description available.')

        if not description.strip():
            print(f"Skipping {strategy_name} due to missing description.")
            continue

        prompt = generate_prompt(strategy_name, description, example_function)
        print(prompt)
        generated_code = get_openai_response(prompt, api_key)

        filename = os.path.join(output_directory, f"{strategy_name.lower().replace(' ', '_')}.py")
        write_to_file(filename, generated_code)
        print(f"Generated {filename}")

#%%
if __name__ == "__main__":
    main()

# %%
import os
import re

def clean_python_scripts(directory, output_file):
    """Cleans Python script files by removing non-code text and combines them into one file."""
    combined_code = ""
    
    for filename in os.listdir(directory):
        if filename.endswith(".py"):  # Process only Python files
            file_path = os.path.join(directory, filename)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
                # Extract Python code by removing non-code text
                code_match = re.search(r"```python\n(.*?)```", content, re.DOTALL)
                if code_match:
                    clean_code = code_match.group(1)
                else:
                    clean_code = content  # If no markdown formatting, keep original content
                
                # Append cleaned code to the combined file
                combined_code += f"\n# File: {filename}\n" + clean_code + "\n"
                
                # Overwrite file with cleaned code
                with open(file_path, 'w', encoding='utf-8') as clean_file:
                    clean_file.write(clean_code)
    
    # Write combined scripts to a single output file
    with open(output_file, 'w', encoding='utf-8') as output:
        output.write(combined_code)
    
    print(f"Processed all scripts. Combined file saved as {output_file}.")

# Example usage
directory = "generated_strategies"  # Directory containing Python scripts
output_file = "all_strategies.py"    # Output file with combined scripts
clean_python_scripts(directory, output_file)

#%%
import os
import re

# Directory containing the Python files
directory = "generated_strategies"
output_file = "combined_functions.py"

# Regular expression to match function definitions
function_pattern = re.compile(r"^def .+?:", re.MULTILINE)

# Open the output file for writing
with open(output_file, "w", encoding="utf-8") as outfile:
    outfile.write("# Combined functions from multiple files\n\n")

    # Iterate through all Python files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".py") and filename != output_file:
            file_path = os.path.join(directory, filename)

            with open(file_path, "r", encoding="utf-8") as infile:
                content = infile.read()
                
                # Extract function definitions
                functions = function_pattern.findall(content)
                
                if functions:
                    outfile.write(f"# Functions from {filename}\n")
                    outfile.write(content + "\n\n")

print(f"Combined functions saved in {output_file}")

# %%
