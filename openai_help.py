# %%
# Use Openai to create python strategy functions
import openai
import json
import os

def generate_prompt(value):
    """Creates a prompt for OpenAI API."""
    return f"""
    Use this example functions as a template and replace value with a variable:
    {value}
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
    api_key = os.getenv("OPENAI_KEY")  # Load API key from environment variable

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

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

    prompt = generate_prompt(example_function)
    print(prompt)
    generated_code = get_openai_response(prompt, api_key)

    filename = 'openai_output'
    write_to_file(filename, generated_code)
    print(f"Generated {filename}")

#%%
if __name__ == "__main__":
    main()

# %%
